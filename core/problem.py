"""Problem representations and helpers for RMS optimisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetime while preserving NaNs."""

    if series.empty:
        return pd.Series(dtype="datetime64[ns]")
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce")


def _infer_processing_time(row: pd.Series) -> float:
    """Infer the processing time for a job in minutes."""

    processing = row.get("Processing_Time")
    if pd.notna(processing):
        return float(processing)
    start = row.get("Scheduled_Start")
    end = row.get("Scheduled_End")
    if pd.notna(start) and pd.notna(end):
        return float((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60.0)
    return 0.0


@dataclass
class ManufacturingProblem:
    """Encapsulate the data describing a scheduling instance."""

    jobs: pd.DataFrame
    objectives: List[str]
    constraints: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, pd.DataFrame):
            raise TypeError("jobs must be provided as a pandas DataFrame")
        if not self.objectives:
            raise ValueError("At least one objective must be specified")
        if self.jobs.index.has_duplicates:
            # Ensure every job can be uniquely addressed when building sequences.
            self.jobs = self.jobs.reset_index(drop=True)

    def build_schedule(self, order: Sequence[int | str] | None = None) -> pd.DataFrame:
        """Construct a feasible schedule following a given job order.

        The implementation assumes a job-shop scenario with potentially
        multiple machines.  Jobs are executed on their designated machine
        and start as soon as both the machine becomes available and the job
        release time has elapsed.  Processing times are handled in minutes.

        Parameters
        ----------
        order:
            Sequence of row indices describing the desired execution order.
            When *None*, the current dataframe order is used.
        """

        if self.jobs.empty:
            return self.jobs.copy()

        if order is None:
            frame = self.jobs.copy()
        else:
            try:
                frame = self.jobs.loc[list(order)].copy()
            except (KeyError, TypeError):
                frame = self.jobs.iloc[list(order)].copy()

        frame = frame.reset_index(drop=True)
        machine_col = "Machine_ID" if "Machine_ID" in frame.columns else None

        release = _ensure_datetime(frame.get("Release_Date", frame.get("Scheduled_Start", pd.NaT)))
        default_release = pd.Timestamp("1970-01-01")
        if release.isna().all():
            release = pd.Series([default_release] * len(frame), index=frame.index, dtype="datetime64[ns]")
        else:
            release = release.fillna(release.min())
        processing_times = frame.apply(_infer_processing_time, axis=1).astype(float).to_numpy()

        machine_available: Dict[str, pd.Timestamp] = {}
        global_clock = min(release.min(), default_release)

        starts: List[pd.Timestamp] = []
        ends: List[pd.Timestamp] = []

        for idx, row in frame.iterrows():
            machine = str(row[machine_col]) if machine_col else "M0"
            release_time = release.iloc[idx]
            if pd.isna(release_time):
                release_time = global_clock
            start_time = max(machine_available.get(machine, global_clock), release_time)
            processing_minutes = processing_times[idx]
            end_time = start_time + pd.to_timedelta(processing_minutes, unit="m")
            machine_available[machine] = end_time
            global_clock = max(global_clock, end_time)
            starts.append(start_time)
            ends.append(end_time)

        frame["Scheduled_Start"] = starts
        frame["Scheduled_End"] = ends
        frame["Processing_Time"] = processing_times
        frame["Completion_Time"] = frame["Scheduled_End"]
        frame["Start_Time"] = frame["Scheduled_Start"]
        return frame

    def job_indices(self) -> Iterable[int]:
        """Return the job indices in execution order."""

        return list(range(len(self.jobs)))
