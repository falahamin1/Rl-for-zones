"""
PJS_07: Same machine routing as PJS_06, but with asymmetric uncertainty per machine group.

Nominal duration d for each task = modal duration (index 1, prob 0.4) from PJS_06.

Distribution rules:
  m1, m2  → near-deterministic: {(round(0.9d), 0.1), (d, 0.8), (round(1.1d), 0.1)}
  m3, m4  → high uncertainty:   {(round(0.5d), 0.4), (d, 0.2), (round(1.5d), 0.4)}
  m5      → bimodal:            {(round(0.6d), 0.5), (d, 0.1), (round(1.8d), 0.4)}

Duplicate durations (from rounding) are merged by combining their probabilities.
"""
from ..instance import Job, ProbJobShopInstance, Task
from ._utils import merge_distribution


def _derive(d: int, machine: str):
    if machine in ("m1", "m2"):
        raw = [(round(d * 0.9), 0.1), (d, 0.8), (round(d * 1.1), 0.1)]
    elif machine in ("m3", "m4"):
        raw = [(round(d * 0.5), 0.4), (d, 0.2), (round(d * 1.5), 0.4)]
    else:  # m5
        raw = [(round(d * 0.6), 0.5), (d, 0.1), (round(d * 1.8), 0.4)]
    return merge_distribution(raw)


# Nominal durations = modal durations (index 1) from PJS_06 distributions
_J1 = {"m2": 5, "m3": 6, "m1": 4, "m4": 6, "m5": 5}
_J2 = {"m3": 6, "m1": 4, "m5": 5, "m2": 5, "m4": 5}
_J3 = {"m5": 5, "m4": 5, "m2": 4, "m1": 5, "m3": 6}
_J4 = {"m4": 6, "m2": 4, "m3": 6, "m5": 4, "m1": 5}
_J5 = {"m1": 5, "m5": 5, "m4": 6, "m3": 5, "m2": 4}


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m2", _derive(_J1["m2"], "m2")),
            Task("J1_T2", "m3", _derive(_J1["m3"], "m3")),
            Task("J1_T3", "m1", _derive(_J1["m1"], "m1")),
            Task("J1_T4", "m4", _derive(_J1["m4"], "m4")),
            Task("J1_T5", "m5", _derive(_J1["m5"], "m5")),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m3", _derive(_J2["m3"], "m3")),
            Task("J2_T2", "m1", _derive(_J2["m1"], "m1")),
            Task("J2_T3", "m5", _derive(_J2["m5"], "m5")),
            Task("J2_T4", "m2", _derive(_J2["m2"], "m2")),
            Task("J2_T5", "m4", _derive(_J2["m4"], "m4")),
        ]),
        Job(job_id="J3", tasks=[
            Task("J3_T1", "m5", _derive(_J3["m5"], "m5")),
            Task("J3_T2", "m4", _derive(_J3["m4"], "m4")),
            Task("J3_T3", "m2", _derive(_J3["m2"], "m2")),
            Task("J3_T4", "m1", _derive(_J3["m1"], "m1")),
            Task("J3_T5", "m3", _derive(_J3["m3"], "m3")),
        ]),
        Job(job_id="J4", tasks=[
            Task("J4_T1", "m4", _derive(_J4["m4"], "m4")),
            Task("J4_T2", "m2", _derive(_J4["m2"], "m2")),
            Task("J4_T3", "m3", _derive(_J4["m3"], "m3")),
            Task("J4_T4", "m5", _derive(_J4["m5"], "m5")),
            Task("J4_T5", "m1", _derive(_J4["m1"], "m1")),
        ]),
        Job(job_id="J5", tasks=[
            Task("J5_T1", "m1", _derive(_J5["m1"], "m1")),
            Task("J5_T2", "m5", _derive(_J5["m5"], "m5")),
            Task("J5_T3", "m4", _derive(_J5["m4"], "m4")),
            Task("J5_T4", "m3", _derive(_J5["m3"], "m3")),
            Task("J5_T5", "m2", _derive(_J5["m2"], "m2")),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_07",
        description=(
            "Medium-large, asymmetric uncertainty: 5 jobs, 5 machines. "
            "Same routing as PJS_06 but m1/m2 near-deterministic, "
            "m3/m4 high uncertainty, m5 bimodal."
        ),
        machines=machines,
        jobs=jobs,
    )
