"""LJS_05: Large benchmark — 9 jobs, 6 machines (54 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6"]
    raw = [
        # J1
        [("m1", 13), ("m2", 10), ("m3", 15), ("m4", 12), ("m5",  9), ("m6", 14)],
        # J2
        [("m2", 11), ("m4", 14), ("m6",  9), ("m3", 16), ("m1", 13), ("m5", 10)],
        # J3
        [("m3", 15), ("m6",  9), ("m5", 13), ("m1", 11), ("m4", 14), ("m2", 12)],
        # J4
        [("m4",  9), ("m1", 13), ("m2", 11), ("m6", 10), ("m3", 16), ("m5", 14)],  # noqa: E501
        # J5
        [("m5", 14), ("m3", 11), ("m4", 10), ("m2", 13), ("m6", 12), ("m1", 15)],
        # J6
        [("m6", 12), ("m5", 15), ("m1", 14), ("m4",  9), ("m2", 11), ("m3", 13)],
        # J7
        [("m1", 10), ("m3", 16), ("m5", 12), ("m6", 13), ("m4", 15), ("m2", 11)],
        # J8
        [("m2", 16), ("m6", 11), ("m3", 14), ("m5", 10), ("m1", 13), ("m4",  9)],
        # J9
        [("m3", 11), ("m4", 13), ("m6", 10), ("m1", 15), ("m5",  9), ("m2", 16)],
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
        name="LJS_05",
        description="Large-05: 9 jobs, 6 machines, 54 tasks. "
                    "Nominal durations 9-16, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
