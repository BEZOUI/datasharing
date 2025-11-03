"""Constraint inference helpers for manufacturing problems."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd


def infer_machine_capacities(data: pd.DataFrame) -> Dict[str, float]:
    """Infer per-machine capacity based on dataset utilisation.

    The heuristic assumes that a machine appearing ``n`` times in the dataset
    can process one job at a time.  The resulting capacity value corresponds to
    the share of the planning horizon that can be allocated concurrently to a
    single job.  This provides a pragmatic constraint bundle that keeps the
    optimisation models consistent with the supplied data.
    """

    if data.empty or "Machine_ID" not in data.columns:
        return {"global": 1.0}

    machine_series = data["Machine_ID"]
    machine_values = machine_series.to_list() if hasattr(machine_series, "to_list") else list(machine_series)
    capacities: Dict[str, float] = {}
    for machine in machine_values:
        key = str(machine)
        capacities[key] = capacities.get(key, 0.0) + 1.0
    for machine, count in list(capacities.items()):
        capacities[machine] = 1.0 / max(float(count), 1.0)
    return capacities


def compute_buffer_limits(data: pd.DataFrame, buffer_columns: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Infer buffer capacities from optional buffer-related columns."""

    if buffer_columns is None:
        buffer_columns = ["Buffer_Capacity", "WIP_Limit"]
    limits: Dict[str, float] = {}
    for column in buffer_columns:
        if column in data.columns:
            series = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
            limits[column.lower()] = float(series.max())
    return limits


def make_constraint_bundle(data: pd.DataFrame, extra_constraints: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Combine machine capacities, buffer limits, and user overrides."""

    constraints = {"machine_capacity": 1.0}
    constraints.update(infer_machine_capacities(data))
    constraints.update(compute_buffer_limits(data))
    if extra_constraints:
        constraints.update(extra_constraints)
    return constraints


__all__ = [
    "infer_machine_capacities",
    "compute_buffer_limits",
    "make_constraint_bundle",
]
