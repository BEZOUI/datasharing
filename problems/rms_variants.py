"""Specialised RMS problem variants."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


def _annotate_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["Scenario"] = [variant] * len(annotated)
    return annotated


def create_dynamic_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Dynamic job shop with online arrivals and breakdown markers."""

    frame = _annotate_variant(data, "dynamic")
    if "Arrival_Time" not in frame.columns:
        raw_start = frame.get("Scheduled_Start")
        if raw_start is None or getattr(raw_start, "empty", False):
            frame["Arrival_Time"] = [pd.Timestamp.now()] * len(frame)
        else:
            frame["Arrival_Time"] = pd.to_datetime(raw_start)
    if "Breakdown_Risk" not in frame.columns:
        frame["Breakdown_Risk"] = [0.05] * len(frame)
    objectives = ["makespan", "total_tardiness", "num_tardy_jobs"]
    constraints = make_constraint_bundle(frame, {"dynamic_arrivals": float(len(frame))})
    metadata = {"problem_type": "dynamic_job_shop", "supports_online": "true"}
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


def create_distributed_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Distributed manufacturing with plant identifiers and logistics."""

    frame = _annotate_variant(data, "distributed")
    if "Plant" not in frame.columns:
        frame["Plant"] = ["Plant_A"] * len(frame)
    if "Transfer_Time" not in frame.columns:
        frame["Transfer_Time"] = [0.0] * len(frame)
    plant_series = frame["Plant"]
    plant_values = plant_series.to_list() if hasattr(plant_series, "to_list") else list(plant_series)
    unique_plants = len(dict.fromkeys(str(value) for value in plant_values))
    objectives = ["makespan", "total_completion_time", "energy"]
    constraints = make_constraint_bundle(frame, {"plants": float(unique_plants)})
    metadata = {"problem_type": "distributed_job_shop", "plants": str(unique_plants)}
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


def create_hybrid_manufacturing_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Hybrid additive/subtractive manufacturing scenario."""

    frame = _annotate_variant(data, "hybrid")
    if "Process_Type" not in frame.columns:
        frame["Process_Type"] = ["subtractive"] * len(frame)
    if "Additive_Layer_Time" not in frame.columns:
        frame["Additive_Layer_Time"] = [0.0] * len(frame)
    objectives = ["makespan", "energy", "total_tardiness"]
    constraints = make_constraint_bundle(frame, {"hybrid_steps": float((frame["Process_Type"] == "additive").sum())})
    metadata: Dict[str, str] = {
        "problem_type": "hybrid_manufacturing",
        "hybrid_operations": str((frame["Process_Type"] == "additive").sum()),
    }
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = [
    "create_dynamic_job_shop_problem",
    "create_distributed_job_shop_problem",
    "create_hybrid_manufacturing_problem",
]
