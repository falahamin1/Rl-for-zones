"""Optimal policy via exhaustive zone-graph exploration + backward DP.

Algorithm:
  1. DFS through all reachable (loc, zone) states, branching over every
     possible task duration at each dispatch action.
  2. Backward DP on the resulting DAG:
       V[terminal]  = 0
       V[s]         = min_a Q[s][a]
       Q[s][a]      = Σ_d P(d) * (min_cost(Z_d) + V[next_key_d])
  3. Evaluate the optimal Q-table on real PTASimulator episodes.

Because the zone graph is finite under k-normalization and the job-shop
DAG has no cycles, value iteration converges in one backward pass.
"""
from __future__ import annotations
from collections import deque
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


class OptimalSymbolicSolver:
    """Exhaustive zone-graph explorer + DP solver for the optimal symbolic policy.

    Parameters
    ----------
    env   SymbolicJobShopEnv (will have its state mutated and restored)
    zctx  ZoneContext wrapping pyudbm
    M     k-normalization constant
    """

    def __init__(self, env: SymbolicJobShopEnv, zctx: ZoneContext, M: int):
        self.env = env
        self.zctx = zctx
        self.M = M

        # Zone graph
        self._states: Dict[tuple, Tuple[tuple, object, List[dict], bool]] = {}
        # key -> (loc, Z, actions, is_terminal)
        self._transitions: Dict[tuple, List[Tuple[str, List]]] = {}
        # key -> [(action_id, [(d, prob, next_key, mc), ...]), ...]

        # Optimal tables (filled after solve())
        self.Q: Dict[tuple, Dict[str, float]] = {}
        self.V: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    def solve(self, verbose: bool = False, state_limit: int = 50_000) -> "OptimalSymbolicSolver":
        """Explore zone graph then solve with backward DP.

        state_limit caps the BFS to avoid combinatorial explosion on large instances.
        If truncated, the DP over the partial graph gives an optimistic lower bound
        (unexplored successors treated as V=0).
        """
        self._explore(verbose=verbose, state_limit=state_limit)
        self._backward_dp()
        return self

    def is_truncated(self) -> bool:
        return getattr(self, "_truncated", False)

    def n_states(self) -> int:
        return len(self._states)

    # ------------------------------------------------------------------
    def evaluate_real(self, n_episodes: int = 100, seed: Optional[int] = None) -> Dict:
        """Evaluate the optimal Q-table on actual PTASimulator episodes.

        Maintains the zone Z alongside each simulator episode by stepping the
        symbolic env with the exact sampled duration at each dispatch.  This
        allows direct (loc, zone_str) lookups in the Q-table.  Falls back to
        SEPT for (loc, zone) pairs not seen during BFS exploration.
        """
        from .simulator import PTASimulator
        from .strategies import sept_strategy
        from tqdm import tqdm
        from .zone_ops import init_zone, apply_guard, reset_clocks, apply_delay, normalize

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

                loc = self.env._loc
                key = (loc, str(Z))
                q   = self.Q.get(key, {})
                best = min(enabled, key=lambda t: q.get(t, _INF))
                chosen = best if q.get(best, _INF) < _INF else sept_fn(state)

                state = sim.start_task(state, chosen)
                d = state.task_states[chosen].duration_if_active

                actions = self.env._compute_actions(loc)
                action  = next((a for a in actions if a["action_id"] == chosen), None)
                branch  = next((b for b in action["branches"] if b["d"] == d), None) \
                          if action else None

                if action is None or branch is None:
                    break

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

                self.env.step_deterministic(chosen, d)

                while not state.is_terminal() and not state.enabled_tasks(instance):
                    state = sim.advance_time(state)

            return state.current_time

        makespans = [_run_one() for _ in tqdm(range(n_episodes), desc="  opt_dp_real", leave=False)]
        arr = np.array(makespans)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "raw":  makespans,
        }

    # ------------------------------------------------------------------
    # Private: zone propagation (same logic as SymbolicRLAgent)
    # ------------------------------------------------------------------

    def _propagate_zone(self, Z, action) -> List[Tuple]:
        """Returns list of (d, prob, Z_branch, next_key, decision_loc)."""
        Z_g = apply_guard(Z, self.zctx, action["guards"])
        if Z_g is None:
            return []
        Z_r = reset_clocks(Z_g, self.zctx, action["resets"])

        result = []
        for branch in action["branches"]:
            d, prob = branch["d"], branch["prob"]
            Z_curr = apply_delay(
                Z_r, self.zctx, self.env.invariant(branch["next_loc"]), 1
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
            if not valid or Z_curr.is_empty():
                continue
            Z_b = normalize(Z_curr, self.zctx, self.M)
            next_key = (branch["decision_loc"], str(Z_b))
            result.append((d, prob, Z_b, next_key, branch["decision_loc"]))
        return result

    # ------------------------------------------------------------------
    # Phase 1: Exhaustive BFS exploration
    # ------------------------------------------------------------------

    def _explore(self, verbose: bool = False, state_limit: int = 50_000) -> None:
        """BFS over all reachable (loc, zone) pairs using deterministic stepping.

        Stops early if `state_limit` states are discovered — the DP will then
        solve the partial graph (remaining unexplored successors treated as
        having V = 0, i.e., no future cost, giving an optimistic lower bound).
        """
        loc0, actions0 = self.env.reset()
        Z0 = init_zone(self.zctx, self.env.invariant(loc0))
        Z0 = apply_delay(Z0, self.zctx, self.env.invariant(loc0), 1)
        Z0 = normalize(Z0, self.zctx, self.M)

        init_key = (loc0, str(Z0))
        init_snap = self.env.save_state()

        self._states[init_key] = (loc0, Z0, actions0, False)
        # Queue entries: (loc, zone, actions, key, env_snapshot)
        queue: deque = deque([(loc0, Z0, actions0, init_key, init_snap)])
        self._truncated = False

        while queue:
            if len(self._states) >= state_limit:
                self._truncated = True
                break

            loc, Z, actions, key, snap = queue.popleft()

            if key in self._transitions:
                continue

            self._transitions[key] = []

            for action in actions:
                branch_results = self._propagate_zone(Z, action)
                if not branch_results:
                    continue

                action_trans = []
                for d, prob, Z_b, next_key, dec_loc in branch_results:
                    mc = min_cost(Z_b, self.zctx)
                    action_trans.append((d, prob, next_key, mc))

                    if next_key not in self._states and len(self._states) < state_limit:
                        self.env.restore_state(snap)
                        next_loc, next_actions, done = self.env.step_deterministic(
                            action["action_id"], d
                        )
                        next_snap = self.env.save_state()
                        self._states[next_key] = (dec_loc, Z_b, next_actions, done)
                        if not done:
                            queue.append((dec_loc, Z_b, next_actions, next_key, next_snap))

                self._transitions[key].append((action["action_id"], action_trans))

            if verbose and len(self._states) % 500 == 0:
                print(f"    [BFS] {len(self._states)} states …")

        if verbose:
            status = " (TRUNCATED)" if self._truncated else ""
            print(f"    [BFS] done — {len(self._states)} states{status}")

    # ------------------------------------------------------------------
    # Phase 2: Backward DP
    # ------------------------------------------------------------------

    def _backward_dp(self) -> None:
        """Compute optimal V and Q via iterative DP on the discovered DAG.

        Because the zone graph is a DAG (makespan strictly increases), value
        iteration converges in at most |states| passes.  In practice a few
        passes suffice.
        """
        # Initialise: terminals = 0, others = INF
        for key, (loc, Z, actions, done) in self._states.items():
            self.V[key] = 0.0 if done else _INF
            self.Q[key] = {}

        # Iterate until convergence
        for _pass in range(len(self._states) + 1):
            changed = False
            for key, action_list in self._transitions.items():
                if not action_list:
                    continue
                best_v = _INF
                for action_id, trans in action_list:
                    q = sum(
                        prob * (mc + (self.V.get(nk, 0.0) if self.V.get(nk, _INF) < _INF else 0.0))
                        for _, prob, nk, mc in trans
                    )
                    self.Q[key][action_id] = q
                    best_v = min(best_v, q)
                if best_v < self.V.get(key, _INF):
                    self.V[key] = best_v
                    changed = True
            if not changed:
                break
