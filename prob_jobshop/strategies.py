from __future__ import annotations
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance
    from .pta import GlobalState


StrategyFn = Callable[["GlobalState"], Optional[str]]


def random_strategy(instance: "ProbJobShopInstance", rng=None) -> StrategyFn:
    _rng = rng if rng is not None else np.random.default_rng()

    def _strategy(state: "GlobalState") -> Optional[str]:
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        idx = int(_rng.integers(len(enabled)))
        return enabled[idx]

    return _strategy


def sept_strategy(instance: "ProbJobShopInstance") -> StrategyFn:
    """Smallest Expected Processing Time first."""

    def _strategy(state: "GlobalState") -> Optional[str]:
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        return min(
            enabled,
            key=lambda tid: instance.task_by_id(tid).distribution.expected_duration(),
        )

    return _strategy


def lept_strategy(instance: "ProbJobShopInstance") -> StrategyFn:
    """Largest Expected Processing Time first."""

    def _strategy(state: "GlobalState") -> Optional[str]:
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        return max(
            enabled,
            key=lambda tid: instance.task_by_id(tid).distribution.expected_duration(),
        )

    return _strategy


def mwr_strategy(instance: "ProbJobShopInstance") -> StrategyFn:
    """Most Work Remaining — start the task whose job has the most remaining expected work."""

    def _remaining_work(state: "GlobalState", job_id: str) -> float:
        job = next(j for j in instance.jobs if j.job_id == job_id)
        return sum(
            task.distribution.expected_duration()
            for task in job.tasks
            if state.task_states[task.task_id].status != "done"
        )

    def _strategy(state: "GlobalState") -> Optional[str]:
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        return max(
            enabled,
            key=lambda tid: _remaining_work(state, instance.job_of_task(tid).job_id),
        )

    return _strategy


def fifo_strategy(instance: "ProbJobShopInstance") -> StrategyFn:
    """FIFO — fixed priority by job index (earlier jobs preferred)."""
    job_priority: Dict[str, int] = {
        job.job_id: idx for idx, job in enumerate(instance.jobs)
    }

    def _strategy(state: "GlobalState") -> Optional[str]:
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        return min(
            enabled,
            key=lambda tid: job_priority[instance.job_of_task(tid).job_id],
        )

    return _strategy


STRATEGY_REGISTRY = {
    "random": random_strategy,
    "sept": sept_strategy,
    "lept": lept_strategy,
    "mwr": mwr_strategy,
    "fifo": fifo_strategy,
}
