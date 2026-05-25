from ..instance import Job, ProbJobShopInstance, Task, TaskDistribution


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m1", TaskDistribution([2, 3, 7], [0.5, 0.1, 0.4])),
            Task("J1_T2", "m2", TaskDistribution([1, 2, 6], [0.5, 0.1, 0.4])),
            Task("J1_T3", "m3", TaskDistribution([3, 4, 8], [0.5, 0.1, 0.4])),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m2", TaskDistribution([2, 3, 7], [0.4, 0.2, 0.4])),
            Task("J2_T2", "m1", TaskDistribution([1, 2, 6], [0.4, 0.2, 0.4])),
            Task("J2_T3", "m3", TaskDistribution([2, 3, 7], [0.4, 0.2, 0.4])),
        ]),
        Job(job_id="J3", tasks=[
            Task("J3_T1", "m3", TaskDistribution([1, 2, 5], [0.5, 0.1, 0.4])),
            Task("J3_T2", "m2", TaskDistribution([2, 3, 7], [0.5, 0.1, 0.4])),
            Task("J3_T3", "m1", TaskDistribution([3, 4, 8], [0.5, 0.1, 0.4])),
        ]),
        Job(job_id="J4", tasks=[
            Task("J4_T1", "m1", TaskDistribution([2, 4, 7], [0.4, 0.2, 0.4])),
            Task("J4_T2", "m3", TaskDistribution([1, 3, 6], [0.4, 0.2, 0.4])),
            Task("J4_T3", "m2", TaskDistribution([2, 3, 7], [0.4, 0.2, 0.4])),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_05",
        description="Medium, bimodal uncertainty: 4 jobs, 3 machines, either fast or slow.",
        machines=machines,
        jobs=jobs,
    )
