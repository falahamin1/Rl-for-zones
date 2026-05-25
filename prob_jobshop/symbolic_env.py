from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance


class SymbolicJobShopEnv:
    """UPTA interface for the job-shop scheduling problem.

    Location: tuple of task statuses in flat job-order.
      "waiting"    — task not yet dispatched
      "active_d{d}" — task dispatched with exact duration d (distinct state per d)
      "done"       — task completed

    action_space item:
        {
          "action_id": task_id,          # e.g. "J1_T1"
          "guards":    [],               # no timing guards for dispatch
          "resets":    ["c_J1"],         # job clock to reset
          "price":     0,
          "branches":  [                 # one per possible duration
            {
              "d":           int,        # exact sampled duration
              "prob":        float,      # P(duration = d)
              "next_loc":    tuple,      # immediate location after dispatch (active_d{d})
              "path":        [...],      # completion steps to next decision point
              "decision_loc": tuple,     # final location where agent decides next
            }, ...
          ],
        }

    path item:
        {
          "completion_guards": [(clock_name, exact_d), ...],  # exact c_J = exact_d
          "next_loc":          loc_tuple,
        }

    invariant(loc) returns [(clock_name, exact_d)] for each active_d{X} task —
    the exact upper bound equals the sampled duration, so combined with the
    completion guard c_J >= exact_d we get c_J = exact_d in the zone.

    step() samples a concrete duration from the task's distribution and encodes
    it in the location.  The returned info dict contains {"sampled_d": d} so
    the symbolic RL agent can select the matching zone branch.
    """

    def __init__(self, instance: "ProbJobShopInstance", seed: Optional[int] = None):
        self.instance = instance
        self._task_ids: List[str] = [
            t.task_id for j in instance.jobs for t in j.tasks
        ]
        self._task_to_idx = {tid: i for i, tid in enumerate(self._task_ids)}

        self._task_job: dict = {
            t.task_id: j.job_id
            for j in instance.jobs for t in j.tasks
        }

        self._max_dur: dict = {
            t.task_id: t.distribution.max_duration()
            for j in instance.jobs for t in j.tasks
        }

        self._rng = np.random.default_rng(seed)

        self._loc: Optional[tuple] = None
        self._virtual_clocks: dict = {}   # job_id -> elapsed since last task start
        self._active_d: dict = {}         # job_id -> exact duration of current task

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[tuple, List[dict]]:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._virtual_clocks = {j.job_id: 0.0 for j in self.instance.jobs}
        self._active_d = {}
        return self._loc, self._compute_actions(self._loc)

    def step(self, task_id: str) -> Tuple[tuple, List[dict], bool, dict]:
        """Execute one dispatch action, sampling a concrete duration."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        dist = self.instance.task_by_id(task_id).distribution
        d = int(self._rng.choice(dist.durations, p=dist.probabilities))

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d

        loc = self._advance_to_decision(loc)
        self._loc = tuple(loc)

        done = all(s == "done" for s in self._loc)
        actions = [] if done else self._compute_actions(self._loc)
        return self._loc, actions, done, {"sampled_d": d}

    def step_deterministic(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool]:
        """Dispatch task_id with exact duration d (no sampling).  Used by the optimal solver."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"
        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d
        loc = self._advance_to_decision(loc)
        self._loc = tuple(loc)
        done = all(s == "done" for s in self._loc)
        actions = [] if done else self._compute_actions(self._loc)
        return self._loc, actions, done

    def save_state(self) -> tuple:
        """Snapshot current env state for later restore."""
        return (self._loc, dict(self._virtual_clocks), dict(self._active_d))

    def restore_state(self, snapshot: tuple) -> None:
        """Restore env to a previously saved snapshot."""
        self._loc, vc, ad = snapshot
        self._virtual_clocks = dict(vc)
        self._active_d = dict(ad)

    def invariant(self, loc: tuple) -> List[Tuple[str, int]]:
        """Upper-bound constraints for active tasks: [(clock_name, exact_d)].

        Because each active status encodes the exact duration, the invariant
        c_J <= d combined with completion guard c_J >= d gives c_J = d in the zone.
        """
        result = []
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active_d"):
                job_id = self._task_job[tid]
                exact_d = int(status[len("active_d"):])
                result.append((f"c_{job_id}", exact_d))
        return result

    def rate(self, loc: tuple) -> int:
        return 1

    def clock_names(self) -> List[str]:
        return [f"c_{j.job_id}" for j in self.instance.jobs] + ["delta"]

    def M(self) -> int:
        return max(t.distribution.max_duration() for t in self.instance.all_tasks())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _has_dispatch_pure(self, loc) -> bool:
        """Return True if any task can be dispatched (no path computation)."""
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None:
                if loc[self._task_to_idx[pred_id]] != "done":
                    continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                return True
        return False

    def _compute_path(self, loc: tuple, tid: str, d: int):
        """Compute the completion-guard path for dispatching tid with exact duration d.

        Uses COPIES of the current virtual clock state so the env is not mutated.

        Returns
        -------
        path : list of {"completion_guards": [(clock_name, exact_d)], "next_loc": tuple}
        decision_loc : tuple — final decision point location
        """
        job_id = self._task_job[tid]
        sim_loc = list(loc)
        sim_loc[self._task_to_idx[tid]] = f"active_d{d}"

        virtual_clocks = dict(self._virtual_clocks)
        virtual_clocks[job_id] = 0.0

        path: List[dict] = []

        while not self._has_dispatch_pure(sim_loc) and not all(
            s == "done" for s in sim_loc
        ):
            remaining = {}
            for t_id, status in zip(self._task_ids, sim_loc):
                if status.startswith("active_d"):
                    jid = self._task_job[t_id]
                    exact_d = int(status[len("active_d"):])
                    rem = exact_d - virtual_clocks.get(jid, 0.0)
                    remaining[t_id] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            for jid in virtual_clocks:
                virtual_clocks[jid] += dt

            completing_guards: List[Tuple[str, int]] = []
            for t_id, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    cjid = self._task_job[t_id]
                    exact_d = int(sim_loc[self._task_to_idx[t_id]][len("active_d"):])
                    completing_guards.append((f"c_{cjid}", exact_d))
                    sim_loc[self._task_to_idx[t_id]] = "done"

            path.append({
                "completion_guards": completing_guards,
                "next_loc":          tuple(sim_loc),
            })

        return path, tuple(sim_loc)

    def _compute_actions(self, loc) -> List[dict]:
        """Return enabled dispatch actions, each with a list of duration branches."""
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        actions = []
        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None:
                pred_idx = self._task_to_idx[pred_id]
                if loc[pred_idx] != "done":
                    continue
            task = self.instance.task_by_id(tid)
            if task.machine in active_machines:
                continue

            job_id = self._task_job[tid]
            dist = task.distribution

            branches = []
            for d, prob in zip(dist.durations, dist.probabilities):
                next_loc_imm = list(loc)
                next_loc_imm[self._task_to_idx[tid]] = f"active_d{d}"
                path, decision_loc = self._compute_path(loc, tid, d)
                branches.append({
                    "d":            d,
                    "prob":         prob,
                    "next_loc":     tuple(next_loc_imm),
                    "path":         path,
                    "decision_loc": decision_loc,
                })

            actions.append({
                "action_id": tid,
                "guards":    [],
                "resets":    [f"c_{job_id}"],
                "price":     0,
                "branches":  branches,
            })
        return actions

    def _advance_to_decision(self, loc: list) -> list:
        """Advance loc using exact sampled durations until a dispatch is available."""
        while True:
            if self._has_dispatch_pure(loc):
                break
            if all(s == "done" for s in loc):
                break

            remaining = {}
            for tid, status in zip(self._task_ids, loc):
                if status.startswith("active_d"):
                    job_id = self._task_job[tid]
                    exact_d = int(status[len("active_d"):])
                    rem = exact_d - self._virtual_clocks.get(job_id, 0.0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            for job_id in self._virtual_clocks:
                self._virtual_clocks[job_id] += dt

            for tid, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    idx = self._task_to_idx[tid]
                    loc[idx] = "done"
                    job_id = self._task_job[tid]
                    self._active_d.pop(job_id, None)

        return loc
