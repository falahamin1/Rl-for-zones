"""
PJS_08: Probabilistic FT06 equivalent — 6 jobs, 6 machines.

Nominal durations from the classical FT06 (Fisher-Thompson 6x6) instance.
Distribution formula applied to each nominal d:
    {(floor(0.7d), 0.25), (d, 0.50), (ceil(1.4d), 0.25)}
"""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5", "m6"]

    # FT06 nominal durations — (machine, nominal_duration) in job-processing order
    raw = [
        [("m3", 1), ("m1", 3), ("m2", 6), ("m4", 7), ("m6", 3), ("m5", 6)],   # J1
        [("m2", 8), ("m3", 5), ("m5", 10), ("m6", 10), ("m1", 10), ("m4", 4)], # J2
        [("m3", 5), ("m4", 4), ("m6", 8), ("m1", 9), ("m2", 1), ("m5", 7)],   # J3
        [("m2", 5), ("m1", 5), ("m3", 5), ("m4", 3), ("m5", 8), ("m6", 9)],   # J4
        [("m3", 9), ("m2", 3), ("m5", 5), ("m6", 4), ("m1", 3), ("m4", 1)],   # J5
        [("m2", 3), ("m4", 3), ("m6", 9), ("m1", 10), ("m5", 4), ("m3", 1)],  # J6
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
        name="PJS_08",
        description=(
            "Large, probabilistic FT06 equivalent: 6 jobs, 6 machines. "
            "Duration distribution: {(floor(0.7d), 0.25), (d, 0.50), (ceil(1.4d), 0.25)}."
        ),
        machines=machines,
        jobs=jobs,
    )
