from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance
    from .pta import GlobalState

_PROB_TOL = 1e-9


def verify_instance(instance: "ProbJobShopInstance") -> List[str]:
    errors: List[str] = []
    seen_task_ids: set = set()

    for job in instance.jobs:
        for task in job.tasks:
            # Unique task IDs
            if task.task_id in seen_task_ids:
                errors.append(f"Duplicate task_id: {task.task_id!r}")
            seen_task_ids.add(task.task_id)

            # Machine validity
            if task.machine not in instance.machines:
                errors.append(
                    f"Task {task.task_id!r} references unknown machine {task.machine!r}"
                )

            dist = task.distribution
            # Duration validity
            for d in dist.durations:
                if not isinstance(d, int) or d <= 0:
                    errors.append(
                        f"Task {task.task_id!r}: duration {d!r} must be a positive integer"
                    )

            # Probability validity
            for p in dist.probabilities:
                if p < 0:
                    errors.append(
                        f"Task {task.task_id!r}: negative probability {p}"
                    )

            # Probability sum
            prob_sum = sum(dist.probabilities)
            if abs(prob_sum - 1.0) > _PROB_TOL:
                errors.append(
                    f"Task {task.task_id!r}: probabilities sum to {prob_sum} (expected 1.0)"
                )

    # Duplicate job IDs
    job_ids = [job.job_id for job in instance.jobs]
    if len(job_ids) != len(set(job_ids)):
        errors.append(f"Duplicate job_ids found: {job_ids}")

    # At least one job and one machine
    if not instance.jobs:
        errors.append("Instance has no jobs")
    if not instance.machines:
        errors.append("Instance has no machines")

    return errors


def verify_simulation(
    instance: "ProbJobShopInstance",
    state_trace: List["GlobalState"],
) -> List[str]:
    errors: List[str] = []

    if not state_trace:
        errors.append("State trace is empty")
        return errors

    # Monotone time
    for i in range(1, len(state_trace)):
        prev_t = state_trace[i - 1].current_time
        curr_t = state_trace[i].current_time
        if curr_t < prev_t - _PROB_TOL:
            errors.append(
                f"Time decreased at trace step {i}: {prev_t} -> {curr_t}"
            )

    for step_idx, state in enumerate(state_trace):
        # Mutual exclusion
        if state.is_conflicting(instance):
            errors.append(f"Mutual exclusion violated at step {step_idx}")

        # Precedence constraint
        for job in instance.jobs:
            for task in job.tasks:
                ts = state.task_states[task.task_id]
                if ts.status == "active":
                    pred_id = instance.predecessor_of(task.task_id)
                    if pred_id is not None:
                        pred_status = state.task_states[pred_id].status
                        if pred_status != "done":
                            errors.append(
                                f"Precedence violated at step {step_idx}: "
                                f"task {task.task_id!r} is active but predecessor "
                                f"{pred_id!r} is {pred_status!r}"
                            )

    # Final state must be terminal
    if not state_trace[-1].is_terminal():
        errors.append("Final state in trace is not terminal (not all tasks done)")

    return errors
