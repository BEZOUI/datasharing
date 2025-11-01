"""Empirical validation utilities."""
from __future__ import annotations

from typing import Dict

import pandas as pd

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - fallback for constrained environments
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from scipy import stats
except ImportError:  # pragma: no cover - fallback for constrained environments
    stats = None  # type: ignore


def friedman_test(results: pd.DataFrame) -> Dict[str, float]:
    if stats is None:
        raise RuntimeError("scipy is required to run the Friedman test")
    pivot = results.pivot(index="replication", columns="algorithm", values="makespan")
    statistic, pvalue = stats.friedmanchisquare(*pivot.T.values)
    return {"statistic": float(statistic), "p_value": float(pvalue)}


def confidence_interval(values: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    if stats is None or np is None:
        raise RuntimeError("scipy and numpy are required to compute confidence intervals")
    mean = float(np.mean(values))
    sem = stats.sem(values)
    interval = stats.t.interval(level, len(values) - 1, loc=mean, scale=sem)
    return {"mean": mean, "lower": float(interval[0]), "upper": float(interval[1])}
