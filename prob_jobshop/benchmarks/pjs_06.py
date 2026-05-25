from ..instance import Job, ProbJobShopInstance, Task, TaskDistribution


def get_instance() -> ProbJobShopInstance:
    machines = ["m1", "m2", "m3", "m4", "m5"]
    jobs = [
        Job(job_id="J1", tasks=[
            Task("J1_T1", "m2", TaskDistribution([3, 5, 8], [0.3, 0.4, 0.3])),
            Task("J1_T2", "m3", TaskDistribution([5, 6, 8], [0.3, 0.4, 0.3])),
            Task("J1_T3", "m1", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
            Task("J1_T4", "m4", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
            Task("J1_T5", "m5", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J2", tasks=[
            Task("J2_T1", "m3", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
            Task("J2_T2", "m1", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
            Task("J2_T3", "m5", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J2_T4", "m2", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J2_T5", "m4", TaskDistribution([4, 5, 7], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J3", tasks=[
            Task("J3_T1", "m5", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J3_T2", "m4", TaskDistribution([3, 5, 8], [0.3, 0.4, 0.3])),
            Task("J3_T3", "m2", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
            Task("J3_T4", "m1", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J3_T5", "m3", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J4", tasks=[
            Task("J4_T1", "m4", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
            Task("J4_T2", "m2", TaskDistribution([3, 4, 6], [0.3, 0.4, 0.3])),
            Task("J4_T3", "m3", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
            Task("J4_T4", "m5", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
            Task("J4_T5", "m1", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
        ]),
        Job(job_id="J5", tasks=[
            Task("J5_T1", "m1", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J5_T2", "m5", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J5_T3", "m4", TaskDistribution([4, 6, 8], [0.3, 0.4, 0.3])),
            Task("J5_T4", "m3", TaskDistribution([3, 5, 7], [0.3, 0.4, 0.3])),
            Task("J5_T5", "m2", TaskDistribution([2, 4, 6], [0.3, 0.4, 0.3])),
        ]),
    ]
    return ProbJobShopInstance(
        name="PJS_06",
        description="Probabilistic LA01 equivalent: 5 jobs, 5 machines, moderate uncertainty.",
        machines=machines,
        jobs=jobs,
    )
