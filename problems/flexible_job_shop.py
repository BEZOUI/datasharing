"""Flexible job shop problem factory."""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


def _normalise_eligible_machines(values: Sequence[str]) -> List[str]:
    machines: List[str] = []
    for value in values:
        if not value:
            continue
        for token in str(value).replace("|", ",").split(","):
            candidate = token.strip()
            if candidate and candidate not in machines:
                machines.append(candidate)
    return machines if machines else ["M0"]


def create_flexible_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Construct a flexible job shop instance where jobs have machine choices."""

    frame = data.copy()
    if "Eligible_Machines" in frame.columns:
        frame["Eligible_Machines"] = frame["Eligible_Machines"].fillna("")
    else:
        frame["Eligible_Machines"] = frame.get("Machine_ID", "M0").astype(str)

    eligibility: Dict[str, List[str]] = {}
    for _, row in frame.iterrows():
        job = str(row.get("Job_ID", "JOB_UNKNOWN"))
        eligible = _normalise_eligible_machines([row.get("Eligible_Machines", "")])
        eligibility[job] = eligible
    frame["Eligible_Machine_Count"] = [len(eligibility[str(row.get("Job_ID", "JOB_UNKNOWN"))]) for _, row in frame.iterrows()]
    objectives = ["makespan", "total_tardiness", "energy"]
    constraints = make_constraint_bundle(frame, {"flexible_choices": float(sum(frame["Eligible_Machine_Count"]))})
    metadata = {
        "problem_type": "flexible_job_shop",
        "eligibility_encoded": True,
    }
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = ["create_flexible_job_shop_problem"]
