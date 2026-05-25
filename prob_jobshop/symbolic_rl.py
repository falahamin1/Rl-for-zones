"""Tabular Q-learning over the symbolic zone graph of a PTA job-shop instance.

Algorithm (model-free Q-learning):
  After dispatching action a at state s = (loc, zone_key) and observing
  sampled duration d, the zone is propagated through that single branch to
  reach s' = (loc', zone_key'):

    r          = min_cost(Z') - min_cost(Z)      # incremental elapsed time
    Q[s][a]   += α * (r + γ * min_{a'} Q[s'][a'] - Q[s][a])
    V[s]       = min_{a} Q[s][a]

  The underlying transition probabilities are never used — only the single
  sampled outcome per step drives the update, exactly as in the region-based
  Q-learning agent.

Zone propagation for the sampled branch d of an action:
  1. apply_guard  — no timing guard for dispatch
  2. reset_clocks — reset job clock to 0
  3. apply_delay  — time passes; invariant = c_J <= d (exact upper bound)
  4. Per completion step in branch["path"]:
       apply c_J >= exact_d  — combined with invariant c_J <= exact_d → c_J = exact_d
       apply_delay at next intermediate location
  5. normalize

The delta clock accumulates total real elapsed time and is never reset;
min_cost(Z) gives the minimum delta value in the zone, i.e. the earliest
possible current time.  The incremental reward per step is therefore
min_cost(Z_next) - min_cost(Z_prev).
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .symbolic_env import SymbolicJobShopEnv
from .zone_ops import (
    ZoneContext,
    init_zone,
    apply_guard,
    reset_clocks,
    apply_delay,
    normalize,
    min_cost,
)

_INF = float("inf")


class SymbolicRLAgent:
    """Tabular RL agent over (location, zone) pairs with probabilistic branching.

    Parameters
    ----------
    env           SymbolicJobShopEnv providing location tuples and branched actions
    zctx          ZoneContext wrapping pyudbm for the same instance
    M             normalization constant (typically instance.worst_case_makespan_upper_bound())
    epsilon_start initial exploration rate for ε-greedy (decays linearly to epsilon_min)
    epsilon_min   floor for exploration rate
    alpha         Q-learning step size
    gamma         discount factor (1.0 for cost-optimal job-shop)
    seed          random seed for reproducibility
    """

    def __init__(
        self,
        env: SymbolicJobShopEnv,
        zctx: ZoneContext,
        M: int,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        alpha: float = 0.1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.zctx = zctx
        self.M = M
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.alpha = alpha
        self.gamma = gamma
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.Q: Dict[tuple, Dict[str, float]] = {}
        self.V_rem: Dict[tuple, float] = {}

        self._episode_costs: List[float] = []
        self._epsilon_history: List[float] = []

    # ------------------------------------------------------------------
    def train(
        self,
        episodes: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        eval_seed: Optional[int] = None,
    ) -> Dict:
        """Run `episodes` training episodes with linearly decaying ε-greedy exploration.

        Parameters
        ----------
        eval_interval   Evaluate greedy policy every N episodes (0 = disabled).
        eval_episodes   Number of real-sim episodes per checkpoint evaluation.
        eval_seed       Seed for the checkpoint simulator (for reproducibility).
        """
        self.epsilon = self.epsilon_start
        decay_per_ep = (self.epsilon_start - self.epsilon_min) / max(1, episodes - 1)
        checkpoint_evals: List[Tuple[int, float]] = []  # (episode, mean real makespan)

        for ep in range(episodes):
            self._epsilon_history.append(self.epsilon)
            cost = self._run_episode(exploring=True)
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

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """Run greedy policy (epsilon=0); return symbolic min_cost stats (lower bounds)."""
        old_eps = self.epsilon
        self.epsilon = 0.0
        costs = [self._run_episode(exploring=False) for _ in range(n_episodes)]
        self.epsilon = old_eps
        arr = np.array(costs)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "raw":  costs,
        }

    def evaluate_real(self, n_episodes: int = 100, seed: Optional[int] = None) -> Dict:
        """Evaluate with exact zone tracking.

        Maintains the zone Z alongside each simulator episode by stepping the
        symbolic env with the exact sampled duration at each dispatch.  This
        allows direct (loc, zone_str) lookups in the Q-table — no aggregation,
        no information loss.  Falls back to SEPT for (loc, zone) pairs not seen
        during training.
        """
        from .simulator import PTASimulator
        from .strategies import sept_strategy
        from tqdm import tqdm

        instance = self.env.instance
        sept_fn  = sept_strategy(instance)
        sim      = PTASimulator(instance, seed=seed)

        def _run_one() -> float:
            state = sim.initial_state()
            self.env.reset()
            Z = init_zone(self.zctx, self.env.invariant(self.env._loc))
            Z = apply_delay(Z, self.zctx, self.env.invariant(self.env._loc), 1)
            Z = normalize(Z, self.zctx, self.M)

            while not state.is_terminal():
                enabled = state.enabled_tasks(instance)
                if not enabled:
                    state = sim.advance_time(state)
                    continue

                # ── Q-table lookup with exact (loc, zone) key ──
                loc = self.env._loc
                key = (loc, str(Z))
                q   = self.Q.get(key, {})
                best = min(enabled, key=lambda t: q.get(t, _INF))
                chosen = best if q.get(best, _INF) < _INF else sept_fn(state)

                # ── Real simulator dispatch — captures sampled d ──
                state = sim.start_task(state, chosen)
                d = state.task_states[chosen].duration_if_active

                # ── Find matching action + branch ──
                actions = self.env._compute_actions(loc)
                action  = next((a for a in actions if a["action_id"] == chosen), None)
                branch  = next((b for b in action["branches"] if b["d"] == d), None) \
                          if action else None

                if action is None or branch is None:
                    # Shouldn't happen if model matches simulator distributions
                    break

                # ── Propagate zone through the exact branch ──
                Z_g = apply_guard(Z, self.zctx, action["guards"])
                if Z_g is None:
                    break
                Z_curr = reset_clocks(Z_g, self.zctx, action["resets"])
                Z_curr = apply_delay(
                    Z_curr, self.zctx, self.env.invariant(branch["next_loc"]), 1
                )
                valid = True
                for step in branch["path"]:
                    for cname, exact_d in step["completion_guards"]:
                        Z_curr = Z_curr & (self.zctx.clocks[cname] >= exact_d)
                        if Z_curr.is_empty():
                            valid = False
                            break
                    if not valid:
                        break
                    Z_curr = apply_delay(
                        Z_curr, self.zctx, self.env.invariant(step["next_loc"]), 1
                    )
                if not valid:
                    break
                Z = normalize(Z_curr, self.zctx, self.M)

                # ── Advance symbolic env to match (updates _loc + _virtual_clocks) ──
                self.env.step_deterministic(chosen, d)

                # ── Advance real simulator to next decision point ──
                while not state.is_terminal() and not state.enabled_tasks(instance):
                    state = sim.advance_time(state)

            return state.current_time

        makespans = [_run_one() for _ in tqdm(range(n_episodes), desc="  sym_rl_real", leave=False)]
        arr = np.array(makespans)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "raw":  makespans,
        }

    def n_states(self) -> int:
        return len(self.Q)

    # ------------------------------------------------------------------
    def _propagate_zone(self, Z, action) -> List[Tuple]:
        """Compute zone branches for one action, one per possible duration.

        For duration d with probability p:
          1. apply_guard (empty)
          2. reset_clocks (job clock → 0)
          3. apply_delay with invariant c_J <= d (from branch["next_loc"])
          4. Per path step: c_J >= exact_d (combined with inv → c_J = exact_d exactly)
          5. normalize

        Returns list of (d, prob, Z_branch, next_key, decision_loc).
        Empty list if all branches produce empty zones.
        """
        Z_g = apply_guard(Z, self.zctx, action["guards"])
        if Z_g is None:
            return []
        Z_r = reset_clocks(Z_g, self.zctx, action["resets"])

        result = []
        for branch in action["branches"]:
            d, prob = branch["d"], branch["prob"]

            # Invariant at immediate next_loc uses exact_d: c_J <= d
            Z_curr = apply_delay(
                Z_r, self.zctx,
                self.env.invariant(branch["next_loc"]),
                1,
            )

            valid = True
            for step in branch["path"]:
                for cname, exact_d in step["completion_guards"]:
                    # Completion guard: c_J >= exact_d
                    # Combined with invariant c_J <= exact_d → c_J = exact_d
                    Z_curr = Z_curr & (self.zctx.clocks[cname] >= exact_d)
                    if Z_curr.is_empty():
                        valid = False
                        break
                if not valid:
                    break
                Z_curr = apply_delay(
                    Z_curr, self.zctx,
                    self.env.invariant(step["next_loc"]),
                    1,
                )

            if not valid or Z_curr.is_empty():
                continue

            Z_b = normalize(Z_curr, self.zctx, self.M)
            next_key = (branch["decision_loc"], str(Z_b))
            result.append((d, prob, Z_b, next_key, branch["decision_loc"]))

        return result

    def _run_episode(self, exploring: bool) -> float:
        loc, actions = self.env.reset()

        Z = init_zone(self.zctx, self.env.invariant(loc))
        Z = apply_delay(Z, self.zctx, self.env.invariant(loc), self.env.rate(loc))
        Z = normalize(Z, self.zctx, self.M)

        Z_nx = Z   # fallback if no valid action found

        while True:
            key = (loc, str(Z))
            if key not in self.Q:
                self.Q[key] = {}
            if key not in self.V_rem:
                self.V_rem[key] = 0.0

            # ------ Find valid actions (zone propagation for reachability only) ------
            valid: List[Tuple[dict, list]] = []
            for action in actions:
                branch_list = self._propagate_zone(Z, action)
                if branch_list:
                    valid.append((action, branch_list))

            if not valid:
                break

            # ------ ε-greedy selection using stored Q-values ------
            if exploring and self._rng.random() < self.epsilon:
                idx = self._rng.randrange(len(valid))
            else:
                idx = min(
                    range(len(valid)),
                    key=lambda i: self.Q[key].get(valid[i][0]["action_id"], 0.0),
                )

            chosen_action, chosen_branches = valid[idx]
            prev_delta = min_cost(Z, self.zctx)

            # ------ Execute action — env samples concrete duration ------
            loc_new, actions_new, done, info = self.env.step(chosen_action["action_id"])
            sampled_d = info.get("sampled_d")

            # ------ Propagate zone through the sampled branch only ------
            matching = [b for b in chosen_branches if b[0] == sampled_d]
            _, _, Z_nx, next_key, _ = matching[0] if matching else chosen_branches[0]

            # ------ Incremental reward = elapsed real time this step ------
            r = min_cost(Z_nx, self.zctx) - prev_delta

            # ------ Bootstrap value for next state ------
            if done:
                v_next = 0.0
            else:
                if next_key not in self.Q:
                    self.Q[next_key] = {}
                if next_key not in self.V_rem:
                    self.V_rem[next_key] = 0.0
                v_next = self.V_rem[next_key]

            # ------ Q-learning update (model-free: single sampled transition) ------
            action_id = chosen_action["action_id"]
            q_old = self.Q[key].get(action_id, 0.0)
            self.Q[key][action_id] = q_old + self.alpha * (r + self.gamma * v_next - q_old)

            # ------ V is the minimum Q over available actions ------
            self.V_rem[key] = min(self.Q[key].values())

            loc = loc_new
            actions = actions_new

            if done:
                return min_cost(Z_nx, self.zctx)

            Z = Z_nx

        return min_cost(Z_nx, self.zctx)
