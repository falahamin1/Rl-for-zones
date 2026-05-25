"""PTADirectEnv — DDPG environment that operates directly on the PTA state.

State
-----
The raw PTA state is (location, clock_valuation).  No zone abstraction, no
region grouping.  Encoded as a flat vector:

  Per task t  (4 values):
    is_waiting_t       : 1 if task is waiting to be dispatched
    is_active_t        : 1 if task is currently running
    is_done_t          : 1 if task has completed
    active_dur_norm_t  : d / max_dur_t  if active (encodes sampled duration),
                         0 otherwise

  Per job j  (1 value):
    clock_norm_j       : c_J / M  where M = worst-case makespan upper bound

  state_dim = 4 * total_tasks + n_jobs

Action
------
  action = [delta_raw, p_0, ..., p_{J-1}]     (J+1 dimensional, tanh output)

  delta  = (delta_raw + 1) / 2  *  delta_max   in [0, delta_max]
  p_j                                           priority score for job j

At each step:
  1. Advance simulation by delta (tasks run, some may complete).
  2. Dispatch the highest-priority enabled task (if any available).
  3. Advance to the next decision point (next dispatch opportunity).
  4. Return new PTA state, reward = -(total elapsed time), done flag.

Reward: −(elapsed time this step)  →  cumulative = −makespan.

Why delta matters
-----------------
delta = 0 recovers the immediate-dispatch behaviour of the current tabular
agents.  delta > 0 lets the agent time its dispatches — e.g. wait for a
preferred job to become free before committing a machine.  This continuous
timing is what makes the action space genuinely richer than the discrete
dispatch-only formulation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import ProbJobShopInstance

_EPS = 1e-9


class PTADirectEnv:
    """DDPG environment operating directly on the PTA state.

    Parameters
    ----------
    instance    ProbJobShopInstance
    seed        RNG seed for duration sampling
    """

    def __init__(self, instance: ProbJobShopInstance, seed: Optional[int] = None):
        self.instance = instance
        self._rng = np.random.default_rng(seed)

        self._task_ids: List[str] = [
            t.task_id for j in instance.jobs for t in j.tasks
        ]
        self._task_to_idx: Dict[str, int] = {
            tid: i for i, tid in enumerate(self._task_ids)
        }
        self._task_job: Dict[str, str] = {
            t.task_id: j.job_id for j in instance.jobs for t in j.tasks
        }
        self._job_ids: List[str] = [j.job_id for j in instance.jobs]

        # Normalisation constants
        self._M: float = float(instance.worst_case_makespan_upper_bound())
        self._max_dur_task: Dict[str, float] = {
            t.task_id: float(t.distribution.max_duration())
            for j in instance.jobs for t in j.tasks
        }
        # delta_max: maximum useful delay = longest single task duration
        self.delta_max: float = float(
            max(t.distribution.max_duration() for j in instance.jobs for t in j.tasks)
        )

        self.state_dim:  int = 4 * len(self._task_ids) + len(self._job_ids)
        self.action_dim: int = 1 + len(self._job_ids)   # [delta, p_0…p_{J-1}]

        # Simulation state
        self._loc:    Optional[tuple] = None
        self._clocks: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._loc    = tuple("waiting" for _ in self._task_ids)
        self._clocks = {jid: 0.0 for jid in self._job_ids}
        return self._state_vector()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute one timed action: delay δ then dispatch best enabled task.

        Parameters
        ----------
        action  (J+1)-dim vector  [delta_raw, p_0, ..., p_{J-1}]
                delta_raw ∈ [-1, 1]  →  delta = (v+1)/2 * delta_max
                p_j ∈ [-1, 1]       →  priority for job j

        Returns
        -------
        next_state, reward = -(total elapsed this step), done
        """
        delta          = float((action[0] + 1.0) / 2.0 * self.delta_max)
        job_priorities = {jid: float(action[1 + i])
                          for i, jid in enumerate(self._job_ids)}

        # 1. Advance by delta (tasks run, some may complete)
        elapsed = self._advance_time_limited(delta)

        # 2. Dispatch highest-priority enabled task
        enabled = self._enabled_tasks()
        if enabled:
            best = max(enabled,
                       key=lambda tid: job_priorities[self._task_job[tid]])
            self._dispatch(best)

        # 3. Advance to next decision point
        elapsed += self._advance_to_dispatch()

        done   = all(s == "done" for s in self._loc)
        reward = -float(elapsed)
        return self._state_vector(), reward, done

    # Deterministic dispatch used during evaluation (duration supplied externally)
    def dispatch_det(self, tid: str, d: int) -> None:
        loc = list(self._loc)
        loc[self._task_to_idx[tid]] = f"active_d{d}"
        self._clocks[self._task_job[tid]] = 0.0
        self._loc = tuple(loc)

    def advance_to_dispatch(self) -> float:
        return self._advance_to_dispatch()

    def enabled_tasks(self) -> List[str]:
        return self._enabled_tasks()

    def state_vector(self) -> np.ndarray:
        return self._state_vector()

    # ------------------------------------------------------------------
    # State encoding (raw PTA state)
    # ------------------------------------------------------------------

    def _state_vector(self) -> np.ndarray:
        vec = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0

        for tid in self._task_ids:
            status = self._loc[self._task_to_idx[tid]]
            if status == "waiting":
                vec[idx], vec[idx+1], vec[idx+2], vec[idx+3] = 1, 0, 0, 0
            elif status == "done":
                vec[idx], vec[idx+1], vec[idx+2], vec[idx+3] = 0, 0, 1, 0
            else:  # "active_d{d}"
                d = float(int(status[len("active_d"):]))
                vec[idx]   = 0
                vec[idx+1] = 1
                vec[idx+2] = 0
                vec[idx+3] = d / max(self._max_dur_task[tid], 1.0)
            idx += 4

        for jid in self._job_ids:
            vec[idx] = self._clocks[jid] / self._M
            idx += 1

        return vec

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _enabled_tasks(self) -> List[str]:
        active_machines = {
            self.instance.task_by_id(tid).machine
            for tid, s in zip(self._task_ids, self._loc)
            if s.startswith("active")
        }
        result = []
        for tid, s in zip(self._task_ids, self._loc):
            if s != "waiting":
                continue
            pred = self.instance.predecessor_of(tid)
            if pred and self._loc[self._task_to_idx[pred]] != "done":
                continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                result.append(tid)
        return result

    def _has_dispatch(self, loc: tuple) -> bool:
        active_machines = {
            self.instance.task_by_id(tid).machine
            for tid, s in zip(self._task_ids, loc)
            if s.startswith("active")
        }
        for tid, s in zip(self._task_ids, loc):
            if s != "waiting":
                continue
            pred = self.instance.predecessor_of(tid)
            if pred and loc[self._task_to_idx[pred]] != "done":
                continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                return True
        return False

    def _dispatch(self, tid: str) -> None:
        dist = self.instance.task_by_id(tid).distribution
        d    = int(self._rng.choice(dist.durations, p=dist.probabilities))
        self.dispatch_det(tid, d)

    def _advance_time_limited(self, budget: float) -> float:
        """Advance clocks by at most `budget` time units; return elapsed time."""
        loc       = list(self._loc)
        elapsed   = 0.0
        remaining = budget

        while remaining > _EPS:
            # Compute time to next task completion
            task_remaining: Dict[str, float] = {}
            for tid, s in zip(self._task_ids, loc):
                if s.startswith("active_d"):
                    jid    = self._task_job[tid]
                    exact_d = float(int(s[len("active_d"):]))
                    rem    = exact_d - self._clocks.get(jid, 0.0)
                    task_remaining[tid] = max(0.0, rem)

            if not task_remaining:
                # No active tasks — just burn the remaining budget
                for jid in self._clocks:
                    self._clocks[jid] += remaining
                elapsed   += remaining
                remaining  = 0.0
                break

            dt_next = min(task_remaining.values())
            dt      = min(dt_next, remaining)

            # Advance clocks by dt
            for jid in self._clocks:
                self._clocks[jid] += dt
            elapsed   += dt
            remaining -= dt

            # Complete tasks that finish exactly at dt_next (only if we reached it)
            if abs(dt - dt_next) < _EPS:
                for tid, rem in task_remaining.items():
                    if abs(rem - dt_next) < _EPS:
                        loc[self._task_to_idx[tid]] = "done"

        self._loc = tuple(loc)
        return elapsed

    def _advance_to_dispatch(self) -> float:
        """Advance until a dispatch is possible or all tasks done. Return elapsed."""
        loc     = list(self._loc)
        elapsed = 0.0

        while not (self._has_dispatch(tuple(loc)) or all(s == "done" for s in loc)):
            task_remaining: Dict[str, float] = {}
            for tid, s in zip(self._task_ids, loc):
                if s.startswith("active_d"):
                    jid    = self._task_job[tid]
                    exact_d = float(int(s[len("active_d"):]))
                    rem    = exact_d - self._clocks.get(jid, 0.0)
                    task_remaining[tid] = max(0.0, rem)

            if not task_remaining:
                break

            dt = min(task_remaining.values())
            for jid in self._clocks:
                self._clocks[jid] += dt
            elapsed += dt

            for tid, rem in task_remaining.items():
                if abs(rem - dt) < _EPS:
                    loc[self._task_to_idx[tid]] = "done"

        self._loc = tuple(loc)
        return elapsed
