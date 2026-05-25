"""LJS_08: Large benchmark — 11 jobs, 7 machines (77 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
    raw = [
        # J1
        [("m1", 12), ("m3",  9), ("m5", 14), ("m7", 11), ("m2", 16), ("m4", 10), ("m6", 13)],
        # J2
        [("m2", 14), ("m5", 11), ("m7", 10), ("m1", 13), ("m4",  9), ("m6", 15), ("m3", 12)],
        # J3
        [("m3", 11), ("m7", 14), ("m2", 13), ("m5",  9), ("m1", 15), ("m4", 12), ("m6", 10)],  # noqa: E501
        # J4
        [("m4", 16), ("m1", 12), ("m6", 11), ("m3", 14), ("m7", 10), ("m2", 13), ("m5",  9)],
        # J5
        [("m5",  9), ("m2", 15), ("m4", 12), ("m6", 13), ("m3", 11), ("m1", 16), ("m7", 14)],
        # J6
        [("m6", 13), ("m4", 10), ("m1", 15), ("m2", 16), ("m5", 12), ("m3",  9), ("m7", 11)],
        # J7
        [("m7", 10), ("m6", 13), ("m3", 16), ("m4", 12), ("m2", 14), ("m5", 11), ("m1", 15)],
        # J8
        [("m1", 15), ("m4", 11), ("m7", 13), ("m5", 14), ("m6",  9), ("m3", 12), ("m2", 16)],
        # J9
        [("m2", 11), ("m6", 16), ("m4", 10), ("m7",  9), ("m3", 13), ("m5", 14), ("m1", 12)],
        # J10
        [("m3", 13), ("m5", 12), ("m6",  9), ("m1", 10), ("m4", 15), ("m7", 11), ("m2", 14)],
        # J11
        [("m4",  9), ("m7", 14), ("m2", 15), ("m6", 11), ("m1", 12), ("m3", 16), ("m5", 10)],
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
        name="LJS_08",
        description="Large-08: 11 jobs, 7 machines, 77 tasks. "
                    "Nominal durations 9-16, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
