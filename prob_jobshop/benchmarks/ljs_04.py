"""LJS_04: Large benchmark — 8 jobs, 6 machines (48 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6"]
    raw = [
        # J1
        [("m1", 12), ("m4",  9), ("m2", 15), ("m6", 11), ("m3", 13), ("m5", 10)],
        # J2
        [("m2", 14), ("m6", 11), ("m4", 10), ("m1", 16), ("m5",  9), ("m3", 13)],
        # J3
        [("m3", 10), ("m1", 13), ("m6", 12), ("m4", 14), ("m2", 16), ("m5",  9)],
        # J4
        [("m4", 16), ("m2", 10), ("m5", 13), ("m3",  9), ("m6", 11), ("m1", 14)],
        # J5
        [("m5",  9), ("m3", 14), ("m1", 11), ("m2", 13), ("m4", 15), ("m6", 12)],
        # J6
        [("m6", 13), ("m5", 12), ("m3", 16), ("m1", 10), ("m2", 14), ("m4", 11)],
        # J7
        [("m1", 11), ("m3", 15), ("m4",  9), ("m5", 12), ("m6", 10), ("m2", 16)],
        # J8
        [("m2", 15), ("m4", 12), ("m1", 11), ("m3", 14), ("m6",  9), ("m5", 13)],
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
        name="LJS_04",
        description="Large-04: 8 jobs, 6 machines, 48 tasks. "
                    "Nominal durations 9-16, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
