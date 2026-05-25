from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .instance import ProbJobShopInstance
from .pta import GlobalState, TaskState

_EPS = 1e-9


class PTASimulator:
    def __init__(self, instance: ProbJobShopInstance, seed: Optional[int] = None):
        self.instance = instance
        self.rng = np.random.default_rng(seed)

    def initial_state(self) -> GlobalState:
        task_states = {
            task.task_id: TaskState(status="waiting")
            for task in self.instance.all_tasks()
        }
        clock_values = {job.job_id: 0.0 for job in self.instance.jobs}
        return GlobalState(
            task_states=task_states,
            clock_values=clock_values,
            current_time=0.0,
        )

    def start_task(self, state: GlobalState, task_id: str) -> GlobalState:
        task = self.instance.task_by_id(task_id)
        duration = int(
            self.rng.choice(
                task.distribution.durations,
                p=task.distribution.probabilities,
            )
        )
        new_task_states = dict(state.task_states)
        new_task_states[task_id] = TaskState(status="active", duration_if_active=duration)

        job = self.instance.job_of_task(task_id)
        new_clocks = dict(state.clock_values)
        new_clocks[job.job_id] = 0.0

        return GlobalState(
            task_states=new_task_states,
            clock_values=new_clocks,
            current_time=state.current_time,
        )

    def advance_time(self, state: GlobalState) -> GlobalState:
        remaining: Dict[str, float] = {}
        for tid, ts in state.task_states.items():
            if ts.status == "active":
                job_id = self.instance.job_of_task(tid).job_id
                remaining[tid] = ts.duration_if_active - state.clock_values[job_id]

        if not remaining:
            raise RuntimeError(
                "advance_time called with no active tasks — possible deadlock"
            )

        dt = min(remaining.values())

        new_clocks = {jid: v + dt for jid, v in state.clock_values.items()}
        new_task_states = dict(state.task_states)
        for tid, rem in remaining.items():
            if abs(rem - dt) < _EPS:
                new_task_states[tid] = TaskState(status="done")

        return GlobalState(
            task_states=new_task_states,
            clock_values=new_clocks,
            current_time=state.current_time + dt,
        )

    def run_episode(
        self,
        strategy: Callable[[GlobalState], Optional[str]],
        record_trace: bool = False,
    ) -> Union[float, Tuple[float, List[GlobalState]]]:
        state = self.initial_state()
        trace = [state] if record_trace else None

        while not state.is_terminal():
            # Phase 1: start any tasks the strategy selects
            task_to_start = strategy(state)
            while task_to_start is not None:
                state = self.start_task(state, task_to_start)
                if record_trace:
                    trace.append(state)
                task_to_start = strategy(state)

            # Phase 2: advance time to next task completion
            if not state.is_terminal():
                state = self.advance_time(state)
                if record_trace:
                    trace.append(state)

        if record_trace:
            return state.current_time, trace
        return state.current_time

    def evaluate_strategy(
        self,
        strategy: Callable[[GlobalState], Optional[str]],
        n_episodes: int = 1000,
        desc: str = "",
    ) -> Dict:
        makespans: List[float] = []
        label = desc or self.instance.name
        for _ in tqdm(range(n_episodes), desc=label, leave=False):
            makespans.append(self.run_episode(strategy))

        arr = np.array(makespans)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "raw": makespans,
        }
