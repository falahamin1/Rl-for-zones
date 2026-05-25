"""
PJS_10: Stress test — probabilistic FT10 equivalent: 10 jobs, 10 machines.

Nominal durations from the classical FT10 (Muth-Thompson 10x10) instance.
Known optimal deterministic makespan: 930.

Distribution formula applied to each nominal d:
    {(max(1, floor(0.7d)), 0.25), (d, 0.50), (ceil(1.4d), 0.25)}
"""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import apply_formula_070_050_140


def get_instance() -> ProbJobShopInstance:
    machines = [f"m{i}" for i in range(1, 11)]

    # FT10 nominal durations — (machine, nominal_duration) in job-processing order
    raw = [
        # J1
        [("m1",29),("m2",78),("m3",9),("m4",36),("m5",49),
         ("m6",11),("m7",62),("m8",56),("m9",44),("m10",21)],
        # J2
        [("m1",43),("m3",90),("m2",75),("m5",11),("m4",69),
         ("m7",28),("m8",46),("m6",46),("m10",72),("m9",30)],
        # J3
        [("m2",91),("m1",85),("m4",39),("m6",74),("m3",90),
         ("m8",10),("m7",12),("m9",89),("m5",45),("m10",33)],
        # J4
        [("m2",81),("m3",95),("m1",71),("m5",99),("m7",9),
         ("m9",52),("m8",85),("m6",98),("m4",22),("m10",43)],
        # J5
        [("m3",14),("m1",6),("m2",22),("m4",61),("m5",26),
         ("m8",69),("m10",21),("m9",49),("m7",72),("m6",53)],
        # J6
        [("m3",84),("m2",2),("m1",52),("m4",95),("m8",48),
         ("m7",72),("m9",47),("m6",65),("m10",6),("m5",25)],
        # J7
        [("m2",46),("m1",37),("m4",61),("m3",13),("m7",32),
         ("m6",21),("m8",32),("m9",89),("m10",30),("m5",55)],
        # J8
        [("m3",31),("m1",86),("m2",46),("m5",74),("m4",32),
         ("m7",88),("m6",19),("m9",48),("m10",36),("m8",79)],
        # J9
        [("m1",76),("m2",69),("m3",76),("m4",51),("m7",85),
         ("m5",11),("m10",40),("m8",89),("m9",26),("m6",74)],
        # J10
        [("m2",85),("m1",13),("m3",61),("m7",7),("m5",64),
         ("m4",76),("m9",47),("m10",52),("m8",90),("m6",45)],
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
        name="PJS_10",
        description=(
            "Scalability stress test — probabilistic FT10 equivalent: 10 jobs, 10 machines. "
            "Deterministic optimal makespan: 930. "
            "Distribution: {(max(1,floor(0.7d)), 0.25), (d, 0.50), (ceil(1.4d), 0.25)}."
        ),
        machines=machines,
        jobs=jobs,
    )
