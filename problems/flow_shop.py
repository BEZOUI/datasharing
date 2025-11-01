"""Flow shop problem factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


@dataclass
class FlowShopSchema:
    """Describe the machine sequence in a flow shop scenario."""

    machines: Sequence[str]

    @staticmethod
    def from_frame(data: pd.DataFrame) -> "FlowShopSchema":
        if "Stage" in data.columns and "Machine_ID" in data.columns:
            ordered = (
                data.sort_values("Stage")["Machine_ID"].astype(str).unique().tolist()
            )
            return FlowShopSchema(tuple(ordered))
        machine_columns: List[str] = [
            column
            for column in data.columns
            if column.lower().startswith("machine_")
        ]
        if machine_columns:
            ordered = [data[column].iloc[0] for column in machine_columns]
            return FlowShopSchema(tuple(str(machine) for machine in ordered))
        machines = data.get("Machine_ID")
        if machines is not None:
            return FlowShopSchema(tuple(str(machine) for machine in machines.astype(str).unique()))
        return FlowShopSchema(("M0",))


def _expand_flow_shop(data: pd.DataFrame, schema: FlowShopSchema) -> pd.DataFrame:
    """Expand wide-form records into operation-level rows."""

    if {"Stage", "Machine_ID"}.issubset(data.columns):
        return data.copy().reset_index(drop=True)

    records: List[dict] = []
    processing_columns = [
        column
        for column in data.columns
        if column.lower().startswith("processing_time_")
    ]
    for _, row in data.iterrows():
        job_id = row.get("Job_ID", "JOB_UNKNOWN")
        due = row.get("Due_Date")
        energy = row.get("Energy_Consumption", 0.0)
        for stage_index, machine in enumerate(schema.machines):
            processing_column_candidates: Iterable[str] = [
                f"Processing_Time_{stage_index + 1}",
                f"Processing_Time_{machine}",
                f"processing_time_{stage_index + 1}",
            ] + processing_columns
            processing_time = None
            for column in processing_column_candidates:
                if column in row and pd.notna(row[column]):
                    processing_time = float(row[column])
                    break
            if processing_time is None:
                processing_time = float(row.get("Processing_Time", 0.0))
            records.append(
                {
                    "Job_ID": job_id,
                    "Machine_ID": machine,
                    "Stage": stage_index + 1,
                    "Processing_Time": processing_time,
                    "Energy_Consumption": energy,
                    "Due_Date": due,
                }
            )
    return pd.DataFrame(records)


def create_flow_shop_problem(data: pd.DataFrame, machine_sequence: Sequence[str] | None = None) -> ManufacturingProblem:
    """Build a :class:`ManufacturingProblem` for deterministic flow shops."""

    schema = FlowShopSchema(tuple(machine_sequence)) if machine_sequence else FlowShopSchema.from_frame(data)
    expanded = _expand_flow_shop(data, schema)
    objectives = ["makespan", "total_completion_time", "energy"]
    constraints = make_constraint_bundle(expanded, {"flow_order": len(schema.machines)})
    metadata = {
        "problem_type": "flow_shop",
        "machine_sequence": ",".join(schema.machines),
    }
    return ManufacturingProblem(jobs=expanded, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = ["create_flow_shop_problem", "FlowShopSchema"]
