from ..instance import Job, ProbJobShopInstance, Task, TaskDistribution


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m1", TaskDistribution([1, 4, 8], [0.4, 0.3, 0.3])),
            Task("J1_T2", "m2", TaskDistribution([2, 5, 8], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m2", TaskDistribution([1, 3, 6], [0.3, 0.4, 0.3])),
            Task("J2_T2", "m1", TaskDistribution([2, 5, 8], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J3", tasks=[
            Task("J3_T1", "m1", TaskDistribution([2, 4, 7], [0.4, 0.3, 0.3])),
            Task("J3_T2", "m2", TaskDistribution([1, 4, 7], [0.3, 0.4, 0.3])),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_03",
        description="Small, high uncertainty: 3 jobs, 2 machines, high variance.",
        machines=machines,
        jobs=jobs,
    )
