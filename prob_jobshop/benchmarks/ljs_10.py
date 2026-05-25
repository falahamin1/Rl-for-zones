"""LJS_10: Large benchmark — 14 jobs, 8 machines (112 tasks)."""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    raw = [
        # J1
        [("m1", 12), ("m3",  9), ("m5", 14), ("m7", 11), ("m2", 16), ("m4", 10), ("m6", 13), ("m8", 15)],  # noqa: E501
        # J2
        [("m2", 14), ("m5", 11), ("m8", 10), ("m1", 13), ("m4",  9), ("m7", 15), ("m3", 12), ("m6", 16)],  # noqa: E501
        # J3
        [("m3", 11), ("m7", 14), ("m2", 13), ("m6",  9), ("m1", 15), ("m8", 12), ("m4", 10), ("m5", 13)],  # noqa: E501
        # J4
        [("m4", 16), ("m1", 12), ("m6", 11), ("m3", 14), ("m8", 10), ("m2", 13), ("m7",  9), ("m5", 12)],  # noqa: E501
        # J5
        [("m5",  9), ("m2", 15), ("m4", 12), ("m8", 13), ("m3", 11), ("m1", 16), ("m7", 14), ("m6", 10)],  # noqa: E501
        # J6
        [("m6", 13), ("m4", 10), ("m1", 15), ("m2", 16), ("m7", 12), ("m3",  9), ("m8", 11), ("m5", 14)],  # noqa: E501
        # J7
        [("m7", 10), ("m6", 13), ("m3", 16), ("m4", 12), ("m2", 14), ("m5", 11), ("m1", 15), ("m8",  9)],  # noqa: E501
        # J8
        [("m8", 15), ("m1", 11), ("m4", 13), ("m7", 14), ("m5",  9), ("m6", 12), ("m3", 16), ("m2", 10)],  # noqa: E501
        # J9
        [("m2", 11), ("m8", 16), ("m6", 10), ("m1",  9), ("m3", 13), ("m5", 14), ("m4", 12), ("m7", 15)],  # noqa: E501
        # J10
        [("m3", 13), ("m5", 12), ("m7",  9), ("m8", 10), ("m4", 15), ("m6", 11), ("m2", 14), ("m1", 16)],  # noqa: E501
        # J11
        [("m4",  9), ("m7", 14), ("m2", 15), ("m5", 11), ("m8", 12), ("m1", 16), ("m3", 10), ("m6", 13)],  # noqa: E501
        # J12
        [("m5", 16), ("m3", 10), ("m8", 11), ("m6", 13), ("m1", 14), ("m4",  9), ("m2", 15), ("m7", 12)],  # noqa: E501
        # J13
        [("m6", 12), ("m2", 13), ("m5", 16), ("m3", 15), ("m7",  9), ("m8", 11), ("m1", 14), ("m4", 10)],  # noqa: E501
        # J14
        [("m7", 10), ("m4", 15), ("m1", 13), ("m8", 12), ("m6", 11), ("m3", 14), ("m5",  9), ("m2", 16)],  # noqa: E501
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
        name="LJS_10",
        description="Large-10: 14 jobs, 8 machines, 112 tasks. "
                    "Nominal durations 9-16, dist {0.7d:0.25, d:0.50, 1.4d:0.25}.",
        machines=machines,
        jobs=jobs,
    )
