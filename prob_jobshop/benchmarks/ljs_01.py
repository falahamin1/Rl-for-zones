"""LJS_01: Large benchmark — 6 jobs, 4 machines (24 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4"]
    raw = [
        # J1
        [("m1", 14), ("m2", 10), ("m3", 12), ("m4",  9)],
        # J2
        [("m2", 11), ("m4", 16), ("m1",  9), ("m3", 13)],
        # J3
        [("m3", 13), ("m1", 11), ("m4", 15), ("m2",  8)],
        # J4
        [("m4", 10), ("m3",  8), ("m2", 13), ("m1", 17)],
        # J5
        [("m1", 12), ("m3", 16), ("m4", 10), ("m2", 14)],
        # J6
        [("m2", 18), ("m1",  9), ("m3", 11), ("m4", 13)],
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
        name="LJS_01",
        description="Large-01: 6 jobs, 4 machines, 24 tasks. "
                    "Nominal durations 8-18, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
