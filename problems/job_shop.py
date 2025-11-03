"""Job shop problem factory."""
from __future__ import annotations

import pandas as pd

from core.problem import ManufacturingProblem


def create_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    objectives = ["makespan", "energy", "total_tardiness"]
    constraints = {"machine_capacity": 1.0}
    if data.empty:
        jobs = pd.DataFrame(columns=[
            "Job_ID",
            "Machine_ID",
            "Scheduled_Start",
            "Scheduled_End",
            "Processing_Time",
            "Energy_Consumption",
            "Due_Date",
        ])
    else:
        jobs = data.reset_index(drop=True)
        if "Job_ID" not in jobs:
            jobs["Job_ID"] = [f"JOB_{i:05d}" for i in range(len(jobs))]
    return ManufacturingProblem(jobs=jobs, objectives=objectives, constraints=constraints)
