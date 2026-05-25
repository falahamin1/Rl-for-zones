"""
RegionJobShopEnv  —  Stochastic scheduling environment for region-based RL.

State representation
--------------------
All task durations are integers, so clock values at every decision point are
integers.  The region (I, Z, L) therefore collapses to its integral part I; the
zero-set and ordering carry no extra information.  The state key is:

    (full_loc, capped_clock_tuple)

where:
  - full_loc: tuple of "waiting" | "active_d{d}" | "done" per task, using the
      same location encoding as the zone-based approach (exact duration kept)
  - capped_clock_tuple: per-job elapsed clock capped at max_constant_dict[job]
      (= max task duration for that job + 1)

The only difference from the zone state is the clock component: here each job
clock is an integer capped at max_constant_j (classical region equivalence class
for integer clocks), while the zone approach uses a DBM normalised at M =
worst-case makespan, which is a much larger bound and therefore keeps waiting-job
clocks distinct longer.  This makes the zone graph strictly larger than the
region graph for instances where jobs spend significant time waiting.

Reward convention
-----------------
step() returns elapsed_time (non-negative integer cost).
Summing over all steps gives the total makespan.
The agent minimises the expected sum → minimises expected makespan.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import ProbJobShopInstance

_EPS = 1e-9


class RegionJobShopEnv:
    """Thin simulation wrapper that exposes (simplified_loc, clock_tuple) states.

    Parameters
    ----------
    instance    ProbJobShopInstance to simulate
    seed        RNG seed for reproducible duration sampling
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

        # Cap clocks one above the maximum guard constant so region comparisons
        # treat all values at or above max as equivalent (same as Timed-RM).
        self.max_constant_dict: Dict[str, int] = {
            j.job_id: max(t.distribution.max_duration() for t in j.tasks) + 1
            for j in instance.jobs
        }

        self._loc: Optional[tuple] = None
        self._clocks: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset to initial state; return state key."""
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._clocks = {jid: 0 for jid in self._job_ids}
        return self.state_key()

    def step(self, task_id: str) -> Tuple[tuple, int, bool]:
        """Dispatch task_id, sample a duration, advance to next decision point.

        Returns
        -------
        state_key   hashable key for the new (simplified_loc, clock_tuple)
        elapsed     integer time elapsed until the next decision point (the cost)
        done        True when all tasks are finished
        """
        idx = self._task_to_idx[task_id]
        assert self._loc[idx] == "waiting", f"Task {task_id} is not waiting"

        task = self.instance.task_by_id(task_id)
        d = int(
            self._rng.choice(task.distribution.durations, p=task.distribution.probabilities)
        )

        loc = list(self._loc)
        loc[idx] = f"active_d{d}"
        jid = self._task_job[task_id]
        self._clocks[jid] = 0

        elapsed = self._advance(loc)
        self._loc = tuple(loc)
        done = all(s == "done" for s in self._loc)
        return self.state_key(), elapsed, done

    def enabled_tasks(self) -> List[str]:
        """Return task IDs that can be dispatched at the current state."""
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
            if self.instance.task_by_id(tid).machine in active_machines:
                continue
            result.append(tid)
        return result

    def state_key(self) -> tuple:
        """Return hashable (full_loc, capped_clock_tuple).

        full_loc keeps the exact duration embedded in each active task's status
        (e.g. "active_d4"), matching the location used by the zone-based approach.
        The only difference from the zone state is in the clock component: here
        each job clock is capped at max_constant_dict[job] (the classical region
        equivalence class for integer clocks), while the zone uses a DBM
        normalised at M = worst-case makespan.
        """
        clk = tuple(
            min(self._clocks[jid], self.max_constant_dict[jid])
            for jid in self._job_ids
        )
        return (self._loc, clk)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _has_dispatch(self, loc: list) -> bool:
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

    def _advance(self, loc: list) -> int:
        """Advance simulation until a dispatch is possible or all tasks are done.

        All durations are integers so dt is always an integer.
        Returns total elapsed time.
        """
        elapsed = 0
        while not (self._has_dispatch(loc) or all(s == "done" for s in loc)):
            remaining: Dict[str, float] = {}
            for tid, s in zip(self._task_ids, loc):
                if s.startswith("active_d"):
                    jid = self._task_job[tid]
                    exact_d = int(s[len("active_d"):])
                    rem = exact_d - self._clocks.get(jid, 0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                break

            dt = int(round(min(remaining.values())))
            elapsed += dt
            for jid in self._clocks:
                self._clocks[jid] += dt

            for tid, rem in remaining.items():
                if abs(rem - dt) < _EPS:
                    loc[self._task_to_idx[tid]] = "done"

        return elapsed
