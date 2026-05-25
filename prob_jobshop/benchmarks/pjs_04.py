from ..instance import Job, ProbJobShopInstance, Task, TaskDistribution


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m1", TaskDistribution([1, 2, 3], [0.3, 0.4, 0.3])),
            Task("J1_T2", "m2", TaskDistribution([3, 4, 5], [0.3, 0.4, 0.3])),
            Task("J1_T3", "m3", TaskDistribution([2, 3, 4], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m2", TaskDistribution([2, 3, 4], [0.3, 0.4, 0.3])),
            Task("J2_T2", "m3", TaskDistribution([1, 2, 3], [0.3, 0.4, 0.3])),
            Task("J2_T3", "m1", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J3", tasks=[
            Task("J3_T1", "m3", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
            Task("J3_T2", "m1", TaskDistribution([2, 3, 4], [0.3, 0.4, 0.3])),
            Task("J3_T3", "m2", TaskDistribution([1, 3, 5], [0.3, 0.4, 0.3])),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_04",
        description="Probabilistic FT03 equivalent: 3 jobs, 3 machines.",
        machines=machines,
        jobs=jobs,
    )
