from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance, Task


@dataclass
class TaskState:
    status: str                       # "waiting" | "active" | "done"
    duration_if_active: Optional[int] = None

    def __post_init__(self):
        if self.status == "active" and self.duration_if_active is None:
            raise ValueError("duration_if_active must be set when status is 'active'")
        if self.status not in ("waiting", "active", "done"):
            raise ValueError(f"Invalid status: {self.status!r}")


@dataclass
class GlobalState:
    task_states: Dict[str, TaskState]  # task_id → TaskState
    clock_values: Dict[str, float]     # job_id → elapsed time since last task start
    current_time: float

    def is_terminal(self) -> bool:
        return all(ts.status == "done" for ts in self.task_states.values())

    def active_tasks(self) -> List[str]:
        return [tid for tid, ts in self.task_states.items() if ts.status == "active"]

    def enabled_tasks(self, instance: "ProbJobShopInstance") -> List[str]:
        active_machines = {
            instance.task_by_id(tid).machine for tid in self.active_tasks()
        }
        enabled = []
        for job in instance.jobs:
            for task in job.tasks:
                ts = self.task_states[task.task_id]
                if ts.status != "waiting":
                    continue
                pred_id = instance.predecessor_of(task.task_id)
                if pred_id is not None and self.task_states[pred_id].status != "done":
                    continue
                if task.machine in active_machines:
                    continue
                enabled.append(task.task_id)
        return enabled

    def is_conflicting(self, instance: "ProbJobShopInstance") -> bool:
        machine_active: Dict[str, str] = {}
        for tid in self.active_tasks():
            machine = instance.task_by_id(tid).machine
            if machine in machine_active:
                return True
            machine_active[machine] = tid
        return False


def build_task_automaton(task: "Task", job_id: str) -> Dict:
    clock = f"c_{job_id}"
    waiting = f"{task.task_id}_waiting"
    done = f"{task.task_id}_done"
    active_states = [
        f"{task.task_id}_active_d{d}" for d in task.distribution.durations
    ]
    states = [waiting] + active_states + [done]

    invariants = {
        f"{task.task_id}_active_d{d}": f"{clock} <= {d}"
        for d in task.distribution.durations
    }

    start_transitions = [
        {
            "from": waiting,
            "to": f"{task.task_id}_active_d{d}",
            "prob": p,
            "reset": clock,
            "guard": "all predecessors in done state",
        }
        for d, p in zip(task.distribution.durations, task.distribution.probabilities)
    ]

    end_transitions = [
        {
            "from": f"{task.task_id}_active_d{d}",
            "to": done,
            "guard": f"{clock} = {d}",
        }
        for d in task.distribution.durations
    ]

    return {
        "task_id": task.task_id,
        "clock": clock,
        "states": states,
        "initial_state": waiting,
        "final_state": done,
        "invariants": invariants,
        "start_transitions": start_transitions,
        "end_transitions": end_transitions,
    }
