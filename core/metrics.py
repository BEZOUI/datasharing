"""Core metrics for manufacturing optimisation."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def compute_makespan(schedule: pd.DataFrame) -> float:
    if schedule.empty:
        return 0.0
    end_times = _ensure_datetime(schedule["Scheduled_End"])
    start_times = _ensure_datetime(schedule["Scheduled_Start"])
    if end_times.isna().all() or start_times.isna().all():
        return 0.0
    return float((end_times.max() - start_times.min()).total_seconds() / 60.0)


def compute_total_completion_time(schedule: pd.DataFrame) -> float:
    completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
    if completion.isna().all():
        return 0.0
    start = _ensure_datetime(schedule.get("Release_Date", schedule.get("Scheduled_Start", pd.NaT)))
    start = start.fillna(start.min())
    flow_times = (completion - start).dt.total_seconds() / 60.0
    return float(np.nansum(flow_times))


def compute_total_tardiness(schedule: pd.DataFrame) -> float:
    if "Due_Date" not in schedule.columns:
        return 0.0
    due = _ensure_datetime(schedule["Due_Date"])
    completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
    tardiness = (completion - due).dt.total_seconds() / 60.0
    tardiness = tardiness.clip(lower=0)
    return float(np.nansum(tardiness))


def compute_energy(schedule: pd.DataFrame) -> float:
    if "Energy_Consumption" not in schedule:
        return 0.0
    return float(pd.to_numeric(schedule["Energy_Consumption"], errors="coerce").fillna(0.0).sum())


def evaluate_schedule(schedule: pd.DataFrame) -> Dict[str, float]:
    makespan = compute_makespan(schedule)
    total_completion = compute_total_completion_time(schedule)
    energy = compute_energy(schedule)
    total_tardiness = compute_total_tardiness(schedule)
    num_tardy = 0
    if "Due_Date" in schedule.columns:
        due = _ensure_datetime(schedule["Due_Date"])
        completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
        tardy_mask = completion > due
        num_tardy = int(tardy_mask.sum())
    mean_flow_time = float(total_completion / max(len(schedule), 1)) if schedule is not None else 0.0
    return {
        "makespan": makespan,
        "total_completion_time": total_completion,
        "mean_flow_time": mean_flow_time,
        "total_tardiness": total_tardiness,
        "num_tardy_jobs": num_tardy,
        "energy": energy,
    }
