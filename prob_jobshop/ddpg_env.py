"""DDPGSchedulingEnv — continuous-feature environment for deep RL.

State vector
------------
Per job j (6 features each):
  is_done_j        : 1 if all tasks of job j finished
  is_active_j      : 1 if job j has a running task
  is_waiting_j     : 1 if job j's next task is enabled (machine free, pred done)
  clock_norm_j     : elapsed clock / max_task_duration_j  ∈ [0, 1]
  remaining_norm_j : remaining time / active task duration  ∈ [0, 1]  (0 if idle)
  progress_j       : tasks completed / total tasks  ∈ [0, 1]

Per machine m (2 features each):
  is_busy_m        : 1 if machine m has a running task
  remaining_norm_m : remaining / max task duration on m  ∈ [0, 1]

Total: 6J + 2M  (fixed for a given instance)

Action
------
J-dimensional continuous vector (one priority score per job).  At each
dispatch decision point the agent is called once; the enabled job with the
highest score is dispatched.

Reward
------
Each dispatch step returns reward = -(time elapsed since the previous
dispatch point).  The cumulative reward over an episode equals -makespan.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import ProbJobShopInstance

_EPS = 1e-9


class DDPGSchedulingEnv:
    """Continuous-feature job-shop environment for DDPG.

    Parameters
    ----------
    instance    ProbJobShopInstance to simulate
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
        self._machine_ids: List[str] = instance.machines

        # Per-job max task duration (for normalisation)
        self._max_dur_job: Dict[str, int] = {
            j.job_id: max(t.distribution.max_duration() for t in j.tasks)
            for j in instance.jobs
        }
        # Per-machine max task duration (for normalisation)
        self._max_dur_machine: Dict[str, int] = {
            m: max(
                t.distribution.max_duration()
                for j in instance.jobs for t in j.tasks
                if t.machine == m
            )
            for m in instance.machines
        }
        # Total tasks per job (for progress feature)
        self._total_tasks: Dict[str, int] = {
            j.job_id: len(j.tasks) for j in instance.jobs
        }

        self.state_dim: int = 6 * len(instance.jobs) + 2 * len(instance.machines)
        self.action_dim: int = len(instance.jobs)

        # Simulation state
        self._loc: Optional[tuple] = None
        self._clocks: Dict[str, int] = {}
        self._last_event_time: int = 0
        self._current_time: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._clocks = {jid: 0 for jid in self._job_ids}
        self._last_event_time = 0
        self._current_time = 0
        return self._state_vector()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Dispatch the highest-priority enabled task; advance to next event.

        Parameters
        ----------
        action  J-dimensional priority vector (higher = prefer that job)

        Returns
        -------
        next_state   feature vector at next decision point
        reward       -(time elapsed since last decision)
        done         True when all tasks have finished
        """
        enabled = self._enabled_tasks()
        if enabled:
            # Score each enabled task by its job's action component
            best_tid = max(
                enabled,
                key=lambda tid: action[self._job_ids.index(self._task_job[tid])],
            )
            self._dispatch(best_tid)

        # Advance simulation to next dispatch point
        elapsed = self._advance()
        self._last_event_time = self._current_time

        done = all(s == "done" for s in self._loc)
        reward = -float(elapsed)
        return self._state_vector(), reward, done

    def enabled_tasks(self) -> List[str]:
        return self._enabled_tasks()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _state_vector(self) -> np.ndarray:
        vec = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0

        active_machines: Dict[str, Tuple[str, int]] = {}  # machine → (jid, remaining)
        for tid, status in zip(self._task_ids, self._loc):
            if status.startswith("active_d"):
                jid = self._task_job[tid]
                exact_d = int(status[len("active_d"):])
                rem = max(0, exact_d - self._clocks.get(jid, 0))
                machine = self.instance.task_by_id(tid).machine
                active_machines[machine] = (jid, rem)

        # ── Per-job features ──
        done_counts: Dict[str, int] = {jid: 0 for jid in self._job_ids}
        for tid, status in zip(self._task_ids, self._loc):
            if status == "done":
                done_counts[self._task_job[tid]] += 1

        enabled_set = set(self._enabled_tasks())

        for jid in self._job_ids:
            # Find first non-done task to get active/waiting status
            active_status = None
            for j in self.instance.jobs:
                if j.job_id != jid:
                    continue
                for t in j.tasks:
                    s = self._loc[self._task_to_idx[t.task_id]]
                    if s != "done":
                        active_status = s
                        break
                break

            is_done_j    = 1.0 if done_counts[jid] == self._total_tasks[jid] else 0.0
            is_active_j  = 1.0 if (active_status and active_status.startswith("active_d")) else 0.0
            # "waiting" means the next task for this job is in enabled_set
            is_waiting_j = 1.0 if any(
                self._task_job[tid] == jid for tid in enabled_set
            ) else 0.0

            # Clock features
            c = self._clocks.get(jid, 0)
            max_d = self._max_dur_job[jid]
            clock_norm = min(c / max(max_d, 1), 1.0)

            remaining_norm = 0.0
            if active_status and active_status.startswith("active_d"):
                exact_d = int(active_status[len("active_d"):])
                rem = max(0, exact_d - c)
                remaining_norm = rem / max(exact_d, 1)

            progress = done_counts[jid] / self._total_tasks[jid]

            vec[idx:idx + 6] = [
                is_done_j, is_active_j, is_waiting_j,
                clock_norm, remaining_norm, progress,
            ]
            idx += 6

        # ── Per-machine features ──
        for m in self._machine_ids:
            if m in active_machines:
                _, rem = active_machines[m]
                max_d = self._max_dur_machine[m]
                vec[idx]     = 1.0
                vec[idx + 1] = rem / max(max_d, 1)
            else:
                vec[idx]     = 0.0
                vec[idx + 1] = 0.0
            idx += 2

        return vec

    # ------------------------------------------------------------------
    # Simulation internals
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

    def _dispatch(self, tid: str) -> None:
        dist = self.instance.task_by_id(tid).distribution
        d = int(self._rng.choice(dist.durations, p=dist.probabilities))
        self._dispatch_det(tid, d)

    def _dispatch_det(self, tid: str, d: int) -> None:
        """Dispatch task tid with exact (externally sampled) duration d."""
        loc = list(self._loc)
        loc[self._task_to_idx[tid]] = f"active_d{d}"
        self._clocks[self._task_job[tid]] = 0
        self._loc = tuple(loc)

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

    def _advance(self) -> int:
        """Advance clocks until the next dispatch opportunity. Returns elapsed time."""
        loc = list(self._loc)
        elapsed = 0
        while not (self._has_dispatch(tuple(loc)) or all(s == "done" for s in loc)):
            remaining = {}
            for tid, s in zip(self._task_ids, loc):
                if s.startswith("active_d"):
                    jid = self._task_job[tid]
                    exact_d = int(s[len("active_d"):])
                    rem = exact_d - self._clocks.get(jid, 0)
                    remaining[tid] = max(0, rem)
            if not remaining:
                break
            dt = int(round(min(remaining.values())))
            elapsed += dt
            self._current_time += dt
            for jid in self._clocks:
                self._clocks[jid] += dt
            for tid, rem in remaining.items():
                if abs(rem - dt) < _EPS:
                    loc[self._task_to_idx[tid]] = "done"
        self._loc = tuple(loc)
        return elapsed
