"""
graph_sizes.py
--------------
BFS-based enumeration of reachable states in the region graph and zone graph
for a ProbJobShopInstance.

Region graph
  State key: (simplified_loc, capped_clock_tuple)
  simplified_loc : "waiting" | "active" | "done" per task
  capped_clocks  : per-job clock capped at max_task_duration_for_job + 1

Zone graph
  State key: (full_loc_with_exact_durations, clock_tuple)
  full_loc    : "waiting" | "active_d{d}" | "done" per task
  clock_tuple : exact integer elapsed-clock per job, capped at M
                (M = sum of all max task durations — the normalization bound)

Since all task durations are integers, every clock value at a decision point
is an exact integer, so zones reduce to single points and the zone key is
equivalent to (loc_with_exact_durations, exact_integer_clocks).  This means
the zone graph has strictly more states than the region graph because its
locations distinguish "active_d4" from "active_d7" while the region graph
collapses both to "active".

For instances where the graph exceeds `max_states` the BFS is aborted early
and the function returns (count_so_far, truncated=True).
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance


def count_region_graph(
    instance: "ProbJobShopInstance",
    max_states: int = 200_000,
) -> Tuple[int, bool]:
    """Count reachable region-graph states via BFS.

    Returns (n_states, truncated).
    """
    from .region_env import RegionJobShopEnv

    env = RegionJobShopEnv(instance, seed=0)

    env.reset()
    initial_loc = env._loc
    initial_clocks = dict(env._clocks)
    s0 = env.state_key()

    visited: set = {s0}
    queue: deque = deque([(initial_loc, initial_clocks)])

    while queue and len(visited) < max_states:
        loc, clocks = queue.popleft()

        env._loc = loc
        env._clocks = dict(clocks)
        enabled = env.enabled_tasks()

        for tid in enabled:
            task = instance.task_by_id(tid)
            for d in task.distribution.durations:
                # Reset to parent state, then step with exact duration d
                env._loc = loc
                env._clocks = dict(clocks)

                idx = env._task_to_idx[tid]
                new_loc_list = list(loc)
                new_loc_list[idx] = f"active_d{d}"
                jid = env._task_job[tid]
                env._clocks[jid] = 0
                env._advance(new_loc_list)
                env._loc = tuple(new_loc_list)

                key = env.state_key()
                child_loc = env._loc
                child_clocks = dict(env._clocks)

                if key not in visited:
                    visited.add(key)
                    done = all(s == "done" for s in child_loc)
                    if not done:
                        queue.append((child_loc, child_clocks))

    return len(visited), len(visited) >= max_states


def count_zone_graph(
    instance: "ProbJobShopInstance",
    max_states: int = 200_000,
) -> Tuple[int, bool]:
    """Count reachable zone-graph states via BFS.

    Since all durations are integers, zones at decision points are single
    points (active-task clocks are pinned by guard + invariant to an exact
    integer; idle-task clocks accumulate as exact integers).  The zone key
    therefore reduces to (full_loc, capped_clock_tuple) where each job clock
    is capped at the per-job max-duration constant — the same threshold used
    by the region abstraction — so that idle-job clocks above this bound are
    treated as equivalent, exactly as k-normalisation would do.

    This gives a fair comparison: both region and zone use the same location
    encoding and the same per-job clock cap.  The only remaining structural
    difference would appear with non-integer clocks or in the full TA graph
    (intermediate states between dispatches), where a zone represents a
    convex range {0 ≤ c ≤ d} in one DBM while the region needs one state
    per integer value.

    Returns (n_states, truncated).
    """
    from .symbolic_env import SymbolicJobShopEnv

    env = SymbolicJobShopEnv(instance, seed=0)
    # Per-job tight cap: the largest value that appears in any guard on c_J
    # is max_duration_j, matching the region abstraction's cap.
    job_caps = {
        j.job_id: max(t.distribution.max_duration() for t in j.tasks)
        for j in instance.jobs
    }
    job_ids = [j.job_id for j in instance.jobs]

    def zone_key(loc: tuple, vc: dict) -> tuple:
        clk = tuple(min(int(round(vc[jid])), job_caps[jid]) for jid in job_ids)
        return (loc, clk)

    loc0, _ = env.reset()
    vc0 = dict(env._virtual_clocks)
    s0 = zone_key(loc0, vc0)

    visited: set = {s0}
    queue: deque = deque([(loc0, vc0)])

    # Precompute durations per task to avoid repeated attribute lookups
    task_durations = {
        t.task_id: t.distribution.durations
        for j in instance.jobs for t in j.tasks
    }
    task_ids = [t.task_id for j in instance.jobs for t in j.tasks]
    task_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    def enabled_tasks(loc):
        active_machines = {
            instance.task_by_id(tid).machine
            for tid, s in zip(task_ids, loc) if s.startswith("active")
        }
        result = []
        for tid, s in zip(task_ids, loc):
            if s != "waiting":
                continue
            pred = instance.predecessor_of(tid)
            if pred and loc[task_to_idx[pred]] != "done":
                continue
            if instance.task_by_id(tid).machine not in active_machines:
                result.append(tid)
        return result

    while queue and len(visited) < max_states:
        loc, vc = queue.popleft()

        for tid in enabled_tasks(loc):
            for d in task_durations[tid]:
                # Reset to parent state, then step with exact duration d
                env._loc = loc
                env._virtual_clocks = dict(vc)
                env._active_d = {}

                new_loc, _, done = env.step_deterministic(tid, d)
                new_vc = dict(env._virtual_clocks)

                key = zone_key(new_loc, new_vc)

                if key not in visited:
                    visited.add(key)
                    if not done:
                        queue.append((new_loc, new_vc))

    return len(visited), len(visited) >= max_states
