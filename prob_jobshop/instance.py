from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskDistribution:
    durations: List[int]
    probabilities: List[float]

    def __post_init__(self):
        if len(self.durations) != len(self.probabilities):
            raise ValueError(
                f"durations and probabilities must have the same length, "
                f"got {len(self.durations)} and {len(self.probabilities)}"
            )

    def expected_duration(self) -> float:
        return sum(d * p for d, p in zip(self.durations, self.probabilities))

    def max_duration(self) -> int:
        return max(self.durations)

    def min_duration(self) -> int:
        return min(self.durations)


@dataclass
class Task:
    task_id: str    # format: "J1_T1"
    machine: str    # format: "m1"
    distribution: TaskDistribution


@dataclass
class Job:
    job_id: str     # format: "J1"
    tasks: List[Task]


@dataclass
class ProbJobShopInstance:
    name: str
    description: str
    machines: List[str]
    jobs: List[Job]

    # Lookup caches — built in __post_init__, not declared as dataclass fields
    def __post_init__(self):
        self._task_index: Dict[str, Task] = {}
        self._task_to_job: Dict[str, Job] = {}
        self._predecessor: Dict[str, Optional[str]] = {}
        for job in self.jobs:
            for i, task in enumerate(job.tasks):
                self._task_index[task.task_id] = task
                self._task_to_job[task.task_id] = job
                self._predecessor[task.task_id] = job.tasks[i - 1].task_id if i > 0 else None

    def all_tasks(self) -> List[Task]:
        return [task for job in self.jobs for task in job.tasks]

    def task_by_id(self, task_id: str) -> Task:
        return self._task_index[task_id]

    def job_of_task(self, task_id: str) -> Job:
        return self._task_to_job[task_id]

    def predecessor_of(self, task_id: str) -> Optional[str]:
        return self._predecessor[task_id]

    def expected_makespan_lower_bound(self) -> float:
        machine_load: Dict[str, float] = {m: 0.0 for m in self.machines}
        for task in self.all_tasks():
            machine_load[task.machine] += task.distribution.expected_duration()
        return max(machine_load.values())

    def worst_case_makespan_upper_bound(self) -> int:
        return sum(task.distribution.max_duration() for task in self.all_tasks())

    def num_tasks(self) -> int:
        return sum(len(job.tasks) for job in self.jobs)

    def mean_expected_duration(self) -> float:
        tasks = self.all_tasks()
        return sum(t.distribution.expected_duration() for t in tasks) / len(tasks)
