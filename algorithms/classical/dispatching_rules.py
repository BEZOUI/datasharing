"""Implementation of classical dispatching rules."""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from core.base_optimizer import BaseOptimizer
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _ensure_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _ensure_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce")


def _fill_reference(series: pd.Series, default: pd.Timestamp) -> pd.Series:
    if series.isna().all():
        return pd.Series([default] * len(series), index=series.index, dtype="datetime64[ns]")
    return series.fillna(series.min())


class DispatchingRule(BaseOptimizer):
    """Base class encapsulating a dispatching rule."""

    rule_name: str = "dispatching_rule"
    ascending: bool = True

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = problem.jobs.copy()
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        priority = self._priority(jobs)
        priority = priority.reindex(jobs.index)
        jobs = jobs.assign(_priority=priority)
        ordered = jobs.sort_values("_priority", ascending=self.ascending, kind="mergesort")
        schedule = problem.build_schedule(ordered.index)
        schedule = schedule.reset_index(drop=True)
        return ScheduleSolution(schedule=schedule, metadata={"rule": self.rule_name})


class FCFSRule(DispatchingRule):
    """First-Come-First-Served based on release time."""

    rule_name = "fcfs"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_datetime(jobs, "Scheduled_Start").rank(method="first")


class SPTRule(DispatchingRule):
    """Shortest processing time first."""

    rule_name = "spt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_series(jobs, "Processing_Time")


class LPTRule(DispatchingRule):
    """Longest processing time first."""

    rule_name = "lpt"
    ascending = False

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_series(jobs, "Processing_Time")


class EDDRule(DispatchingRule):
    """Earliest due date rule."""

    rule_name = "edd"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_datetime(jobs, "Due_Date").rank(method="first")


class SLACKRule(DispatchingRule):
    """Schedule jobs with minimum slack."""

    rule_name = "slack"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        return pd.Series(slack, index=jobs.index)


class CriticalRatioRule(DispatchingRule):
    """Critical ratio rule (time remaining / processing)."""

    rule_name = "critical_ratio"
    ascending = False

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        time_remaining = (due - start).dt.total_seconds() / 60.0
        ratio = time_remaining / processing.replace(0, math.nan)
        return ratio.fillna(0.0)


class WSPTRule(DispatchingRule):
    """Weighted shortest processing time rule."""

    rule_name = "wspt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        weights = _ensure_series(jobs, "Priority", default=1.0)
        return processing / weights.replace(0, math.nan)


class ATRule(DispatchingRule):
    """Apparent tardiness cost (ATC) rule."""

    rule_name = "atc"

    def __init__(self, k: float = 2.0, **kwargs):
        super().__init__(k=k, **kwargs)
        self.k = k

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        release = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        avg_proc = processing.mean() if not processing.empty else 1.0
        urgency = (due - release).dt.total_seconds() / 60.0 - processing
        exponent = urgency.clip(lower=0.0) / (self.k * avg_proc)
        exponent = exponent.fillna(0.0)
        priority = exponent.apply(lambda value: math.exp(-value)) / processing.replace(0, math.nan)
        priority = priority.apply(
            lambda value: 0.0 if value in (math.inf, -math.inf) or pd.isna(value) else value
        )
        return priority


class MSERule(DispatchingRule):
    """Minimum slack per operation."""

    rule_name = "mse"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        operations = _ensure_series(jobs, "Remaining_Operations", default=1.0)
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        return slack / operations.replace(0, math.nan)


class SRPTRule(DispatchingRule):
    """Shortest remaining processing time."""

    rule_name = "srpt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        remaining = _ensure_series(jobs, "Remaining_Processing_Time")
        if (remaining == 0).all():
            remaining = _ensure_series(jobs, "Processing_Time")
        return remaining


class CoversionRule(DispatchingRule):
    """CoVERT rule emphasising tardiness avoidance."""

    rule_name = "covert"
    ascending = False

    def __init__(self, k: float = 3.0, **kwargs):
        super().__init__(k=k, **kwargs)
        self.k = k

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        avg_proc = processing.mean() if not processing.empty else 1.0
        exponent = slack.clip(lower=0.0) / (self.k * avg_proc)
        return exponent.apply(lambda value: math.exp(-value))


DISPATCHING_RULES: Dict[str, type[DispatchingRule]] = {
    "fcfs": FCFSRule,
    "spt": SPTRule,
    "lpt": LPTRule,
    "edd": EDDRule,
    "slack": SLACKRule,
    "critical_ratio": CriticalRatioRule,
    "wspt": WSPTRule,
    "atc": ATRule,
    "mse": MSERule,
    "srpt": SRPTRule,
    "covert": CoversionRule,
}


def list_dispatching_rules() -> List[str]:
    """Return the available dispatching rule identifiers."""

    return sorted(DISPATCHING_RULES.keys())
