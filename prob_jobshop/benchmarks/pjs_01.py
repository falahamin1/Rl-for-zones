from ..instance import Job, ProbJobShopInstance, Task, TaskDistribution


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m1", TaskDistribution([3, 4, 5], [0.1, 0.8, 0.1])),
            Task("J1_T2", "m2", TaskDistribution([2, 3, 4], [0.1, 0.8, 0.1])),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m2", TaskDistribution([2, 3, 4], [0.2, 0.6, 0.2])),
            Task("J2_T2", "m1", TaskDistribution([1, 2, 3], [0.1, 0.8, 0.1])),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_01",
        description="Tiny, low uncertainty: 2 jobs, 2 machines, near-deterministic durations.",
        machines=machines,
        jobs=jobs,
    )
