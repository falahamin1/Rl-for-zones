"""Zone-graph Q-learning agent for the LB-TRM Frozen Lake task.

Since the TRM structure is known ahead of time, we precompute the zone graph
from the TRM and use its nodes directly as Q-learning states.

For guard  x >= c_min AND y <= D  the zone graph has exactly 4 clock-state nodes:

  x-node : 0  if x < c_min   (pre-guard: must accumulate more reaction time)
            1  if x >= c_min  (guard-satisfiable: can visit next goal)

  y-node : 0  if y <= D      (global budget remaining)
            1  if y > D       (global deadline exceeded)

State key : (grid_pos, rm_state, x_node, y_node)
Action    : (delay, grid_action)  delay ∈ {0,...,d_max}, grid_action ∈ {0..3}

No pyudbm required — zone graph nodes are determined by threshold comparisons
derived directly from the guard constants c_min and D.

Q-learning update
-----------------
  Q[(s, a)] += α * (r + γ^(d+1) * max_a' Q[(s', a')] - Q[(s, a)])
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid import GridEnv, ACTIONS
from .trm import TRMInstance, TRMState, U_TERMINAL


class TRMZoneEnv:
    """TRM Frozen Lake simulation with zone-graph state keys.

    State key uses 2-bit clock abstraction derived from TRM guard thresholds:
      x_node = 0 if x < c_min else 1
      y_node = 0 if y <= D    else 1
    """

    def __init__(
        self,
        trm: TRMInstance,
        d_max: int = 0,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm   = trm
        self.d_max = d_max
        self._grid  = GridEnv(slip_prob=slip_prob, seed=seed)
        self._trm_s = TRMState()
        self._x: int = 0
        self._y: int = 0

    def reset(self) -> Tuple:
        pos = self._grid.reset()
        self._trm_s.reset()
        self._x = 0
        self._y = 0
        return pos, self._trm_s.u

    def step(self, delay: int, grid_action: int):
        dt = delay + 1
        pos, event, env_done = self._grid.step(grid_action)

        self._x += dt
        self._y += dt
        self._trm_s.clocks["x"] = self._x
        self._trm_s.clocks["y"] = self._y

        rm_reward, done = self._trm_s.step(event, self.trm)
        done = done or env_done

        if event in ("a", "b", "c") and self._trm_s.u != U_TERMINAL:
            self._x = 0

        return pos, self._trm_s.u, rm_reward, done

    def state_key(self) -> tuple:
        x_node = 0 if self._x < self.trm.c_min else 1
        y_node = 0 if self._y <= self.trm.D    else 1
        return (self._grid.pos(), self._trm_s.u, x_node, y_node)


class TRMZoneAgent:
    """Tabular Q-learning over zone-graph states × (delay, grid_action).

    Parameters
    ----------
    trm           TRM instance (provides c_min, D, d_max)
    d_max         Maximum delay (overrides trm.d_max if provided explicitly)
    alpha         Learning rate
    gamma         Discount factor
    epsilon_start Initial ε for ε-greedy
    epsilon_min   Floor for ε decay
    slip_prob     Grid slip probability
    seed          RNG seed
    """

    def __init__(
        self,
        trm: TRMInstance,
        d_max: Optional[int] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm    = trm
        self.d_max  = d_max if d_max is not None else trm.d_max
        self.alpha  = alpha
        self.gamma  = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon = epsilon_start
        self._rng    = random.Random(seed)
        self._seed   = seed
        self.slip_prob = slip_prob

        self._actions: List[Tuple[int, int]] = [
            (d, a) for d in range(self.d_max + 1) for a in ACTIONS
        ]
        self.Q: Dict[tuple, Dict[tuple, float]] = {}

    def _get_q(self, s: tuple) -> Dict[tuple, float]:
        if s not in self.Q:
            self.Q[s] = {a: 0.0 for a in self._actions}
        return self.Q[s]

    def _best_action(self, s: tuple) -> tuple:
        q = self._get_q(s)
        return max(q, key=q.get)

    def _eps_greedy(self, s: tuple) -> tuple:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self._actions)
        return self._best_action(s)

    def train(
        self,
        episodes: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        eval_seed: Optional[int] = None,
    ) -> dict:
        decay = (self.epsilon_start - self.epsilon_min) / max(1, episodes - 1)
        self.epsilon = self.epsilon_start
        checkpoint_evals: List[Tuple[int, float]] = []

        for ep in range(episodes):
            self._run_episode(exploring=True)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay)

            if eval_interval > 0 and (ep + 1) % eval_interval == 0:
                saved = self.epsilon
                self.epsilon = 0.0
                res = self.evaluate(n_episodes=eval_episodes, seed=eval_seed)
                self.epsilon = saved
                checkpoint_evals.append((ep + 1, res["mean"]))

        return {"checkpoint_evals": checkpoint_evals}

    def evaluate(self, n_episodes: int = 100, seed: Optional[int] = None) -> dict:
        saved = self.epsilon
        self.epsilon = 0.0
        base_seed = seed if seed is not None else self._seed
        eval_rng = random.Random(base_seed)
        rewards = [
            self._run_episode(exploring=False, seed=eval_rng.randint(0, 2**31))
            for _ in range(n_episodes)
        ]
        self.epsilon = saved
        arr = np.array(rewards)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    def n_states(self) -> int:
        return len(self.Q)

    def _run_episode(self, exploring: bool, seed: Optional[int] = None) -> float:
        ep_seed = seed if seed is not None else self._rng.randint(0, 2**31)
        env = TRMZoneEnv(
            self.trm, d_max=self.d_max,
            slip_prob=self.slip_prob, seed=ep_seed,
        )
        pos, rm = env.reset()
        total_r = 0.0
        max_steps = 400

        for _ in range(max_steps):
            if rm == U_TERMINAL:
                break
            s = env.state_key()
            action = self._eps_greedy(s) if exploring else self._best_action(s)
            delay, grid_action = action
            _, rm, reward, done = env.step(delay, grid_action)
            total_r += reward

            if exploring:
                s_new = env.state_key()
                q = self._get_q(s)
                q_new = self._get_q(s_new)
                best_next = max(q_new.values()) if not done else 0.0
                q[action] += self.alpha * (
                    reward + (self.gamma ** (delay + 1)) * best_next - q[action]
                )
            if done:
                break

        return total_r
