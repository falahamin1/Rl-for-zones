"""
RegionRLAgent  —  Tabular Q-learning over (simplified_loc, clock_tuple) states.

Q[s][task_id] = estimated expected total remaining makespan from state s
                when dispatching task_id next.

Bellman update (cost-minimisation, γ = 1.0):

    target = elapsed + γ · min_{a'} Q[s'][a']     (non-terminal)
    target = elapsed                                (terminal)
    Q[s][a] += lr · (target − Q[s][a])

Initialisation: Q[s][a] = 0 for all newly-seen (s, a) pairs.  This is optimistic
for a cost problem (the agent assumes no future cost until it observes otherwise),
which encourages exploration similarly to the zone-based agent's ∞-initialisation.

Fallback: states not seen during training use SPT (Shortest Processing Time) — the
task with the smallest expected duration — identical in spirit to the zone agent's
SEPT fallback.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .region_env import RegionJobShopEnv
from .instance import ProbJobShopInstance

_INF = float("inf")


class RegionRLAgent:
    """Tabular Q-learning agent for job-shop scheduling via region abstraction.

    Parameters
    ----------
    env             RegionJobShopEnv wrapping the instance
    epsilon_start   initial ε for ε-greedy exploration
    epsilon_min     floor for ε (reached by the final training episode)
    lr              Q-learning step size
    gamma           discount factor (1.0 = undiscounted makespan minimisation)
    seed            RNG seed
    """

    def __init__(
        self,
        env: RegionJobShopEnv,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        lr: float = 0.1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.lr = lr
        self.gamma = gamma
        self._rng = random.Random(seed)
        self._np_seed = seed

        self.Q: Dict[tuple, Dict[str, float]] = {}

        self._episode_costs: List[float] = []
        self._epsilon_history: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        episodes: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        eval_seed: Optional[int] = None,
    ) -> Dict:
        """Train for `episodes` episodes with linearly-decaying ε-greedy.

        Parameters
        ----------
        eval_interval   evaluate greedy policy every N episodes (0 = disabled)
        eval_episodes   number of real-sim episodes per checkpoint
        eval_seed       seed for checkpoint evaluations
        """
        self.epsilon = self.epsilon_start
        decay_per_ep = (self.epsilon_start - self.epsilon_min) / max(1, episodes - 1)
        checkpoint_evals: List[Tuple[int, float]] = []

        for ep in range(episodes):
            self._epsilon_history.append(self.epsilon)
            cost = self._run_episode()
            self._episode_costs.append(cost)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_per_ep)

            if eval_interval > 0 and (ep + 1) % eval_interval == 0:
                saved_eps = self.epsilon
                self.epsilon = 0.0
                res = self.evaluate_real(n_episodes=eval_episodes, seed=eval_seed)
                self.epsilon = saved_eps
                checkpoint_evals.append((ep + 1, res["mean"]))

        return {
            "episode_costs": self._episode_costs,
            "epsilon_history": self._epsilon_history,
            "checkpoint_evals": checkpoint_evals,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_real(self, n_episodes: int = 100, seed: Optional[int] = None) -> Dict:
        """Run greedy policy on fresh simulation episodes; return makespan stats."""
        instance = self.env.instance
        makespans = []
        for i in tqdm(range(n_episodes), desc="  region_eval", leave=False):
            ep_seed = (seed + i) if seed is not None else None
            eval_env = RegionJobShopEnv(instance, seed=ep_seed)
            eval_env.reset()
            makespans.append(self._run_greedy(eval_env))

        arr = np.array(makespans)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "raw":  makespans,
        }

    def n_states(self) -> int:
        """Number of distinct (loc, clock) states visited during training."""
        return len(self.Q)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spt_action(self, enabled: List[str]) -> str:
        """Shortest Processing Time fallback (expected duration)."""
        return min(
            enabled,
            key=lambda tid: self.env.instance.task_by_id(tid).distribution.expected_duration(),
        )

    def _greedy_action(self, s: tuple, enabled: List[str]) -> str:
        """Pick action with lowest Q-value; fall back to SPT for unseen states."""
        if s in self.Q:
            seen = [t for t in enabled if t in self.Q[s]]
            if seen:
                return min(seen, key=lambda t: self.Q[s][t])
        return self._spt_action(enabled)

    def _ensure_q(self, s: tuple, tasks: List[str]) -> None:
        """Lazily initialise Q entries to 0 for newly seen (state, task) pairs."""
        if s not in self.Q:
            self.Q[s] = {}
        for tid in tasks:
            if tid not in self.Q[s]:
                self.Q[s][tid] = 0.0

    def _run_episode(self) -> float:
        """One ε-greedy training episode.  Returns total makespan."""
        self.env.reset()
        total_cost = 0.0

        while True:
            enabled = self.env.enabled_tasks()
            if not enabled:
                break

            s = self.env.state_key()
            self._ensure_q(s, enabled)

            # ε-greedy action selection
            if self._rng.random() < self.epsilon:
                action = self._rng.choice(enabled)
            else:
                action = min(enabled, key=lambda t: self.Q[s][t])

            s_new, elapsed, done = self.env.step(action)
            total_cost += elapsed

            if done:
                target = float(elapsed)
            else:
                enabled_new = self.env.enabled_tasks()
                self._ensure_q(s_new, enabled_new)
                best_next = min(self.Q[s_new].values()) if self.Q[s_new] else 0.0
                target = elapsed + self.gamma * best_next

            self.Q[s][action] += self.lr * (target - self.Q[s][action])

            if done:
                break

        return total_cost

    def _run_greedy(self, env: RegionJobShopEnv) -> float:
        """Run a single greedy episode on the given env.  Returns makespan."""
        total_cost = 0.0
        while True:
            enabled = env.enabled_tasks()
            if not enabled:
                break
            s = env.state_key()
            action = self._greedy_action(s, enabled)
            _, elapsed, done = env.step(action)
            total_cost += elapsed
            if done:
                break
        return total_cost
