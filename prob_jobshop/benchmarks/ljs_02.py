"""LJS_02: Large benchmark — 7 jobs, 5 machines (35 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5"]
    raw = [
        # J1
        [("m1", 13), ("m2",  9), ("m3", 15), ("m4", 11), ("m5", 12)],
        # J2
        [("m2", 11), ("m4", 14), ("m1", 10), ("m5", 13), ("m3", 16)],
        # J3
        [("m3", 15), ("m5", 10), ("m2", 12), ("m1", 16), ("m4",  9)],
        # J4
        [("m4",  9), ("m1", 13), ("m5", 11), ("m3", 10), ("m2", 14)],
        # J5
        [("m5", 14), ("m3", 12), ("m4",  9), ("m2", 15), ("m1", 11)],
        # J6
        [("m1", 16), ("m4", 11), ("m2", 13), ("m5",  9), ("m3", 15)],
        # J7
        [("m2", 10), ("m5", 15), ("m3", 14), ("m4", 12), ("m1", 13)],
    ]
    jobs = []
    for j_idx, job_raw in enumerate(raw, 1):
        job_id = f"J{j_idx}"
        tasks = [
            Task(f"{job_id}_T{t_idx}", machine, apply_formula_070_050_140(d))
            for t_idx, (machine, d) in enumerate(job_raw, 1)
        ]
        jobs.append(Job(job_id=job_id, tasks=tasks))
    return ProbJobShopInstance(
        name="LJS_02",
        description="Large-02: 7 jobs, 5 machines, 35 tasks. "
                    "Nominal durations 9-16, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
