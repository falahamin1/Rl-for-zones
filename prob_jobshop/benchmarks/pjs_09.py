"""
PJS_09: Large, high variance — 8 jobs, 6 machines.

Own routing table with cyclic-offset structure.
Distribution formula: {(max(1, d-3), 0.35), (d, 0.30), (d+4, 0.35)}
High uncertainty: d_high/d_low ratio ≈ 4 for typical durations.
"""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_spread4


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6"]

    # (machine, nominal_duration) in job-processing order
    raw = [
        [("m1", 3), ("m2", 5), ("m3", 4), ("m4", 7), ("m5", 3), ("m6", 6)],  # J1
        [("m2", 6), ("m3", 4), ("m4", 5), ("m5", 8), ("m6", 3), ("m1", 7)],  # J2
        [("m3", 5), ("m4", 6), ("m5", 3), ("m6", 4), ("m1", 8), ("m2", 5)],  # J3
        [("m4", 4), ("m5", 5), ("m6", 7), ("m1", 3), ("m2", 6), ("m3", 4)],  # J4
        [("m5", 7), ("m6", 3), ("m1", 5), ("m2", 4), ("m3", 6), ("m4", 5)],  # J5
        [("m6", 3), ("m1", 6), ("m2", 4), ("m5", 5), ("m4", 3), ("m3", 7)],  # J6
        [("m1", 5), ("m3", 4), ("m5", 6), ("m2", 3), ("m6", 7), ("m4", 4)],  # J7
        [("m2", 4), ("m4", 3), ("m6", 5), ("m1", 6), ("m3", 4), ("m5", 7)],  # J8
    ]

    jobs = []
    for j_idx, job_raw in enumerate(raw, 1):
        job_id = f"J{j_idx}"
        tasks = [
            Task(f"{job_id}_T{t_idx}", machine, apply_formula_spread4(d))
            for t_idx, (machine, d) in enumerate(job_raw, 1)
        ]
        jobs.append(Job(job_id=job_id, tasks=tasks))

    return ProbJobShopInstance(
        name="PJS_09",
        description=(
            "Large, high variance stress test: 8 jobs, 6 machines. "
            "Distribution: {(max(1,d-3), 0.35), (d, 0.30), (d+4, 0.35)} — "
            "d_high/d_low ratio ≈ 4."
        ),
        machines=machines,
        jobs=jobs,
    )
