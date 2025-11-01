"""Shared helpers for metaheuristic scheduling algorithms."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem

DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "makespan": 1.0,
    "total_completion_time": 0.05,
    "total_tardiness": 0.1,
    "energy": 0.01,
}


def merge_objective_weights(overrides: Dict[str, float] | None) -> Dict[str, float]:
    """Combine user provided weights with sensible defaults."""

    weights = DEFAULT_OBJECTIVE_WEIGHTS.copy()
    if overrides:
        weights.update(overrides)
    return weights


def sequence_objective(
    problem: ManufacturingProblem, sequence: Sequence[int], weights: Dict[str, float]
) -> tuple[float, Dict[str, float]]:
    """Evaluate a permutation of jobs returning weighted objective and metrics."""

    schedule = problem.build_schedule(sequence)
    metrics = evaluate_schedule(schedule)
    objective = 0.0
    for key, weight in weights.items():
        objective += weight * metrics.get(key, 0.0)
    return objective, metrics


def random_sequence(problem: ManufacturingProblem, rng: random.Random) -> List[int]:
    """Generate a random permutation of job indices for the problem."""

    indices = list(problem.jobs.index)
    rng.shuffle(indices)
    return indices


def processing_times(problem: ManufacturingProblem) -> Dict[int, float]:
    """Return the processing time per job index for quick lookup."""

    durations: Dict[int, float] = {}
    for idx, row in problem.jobs.iterrows():
        value = row.get("Processing_Time")
        if value is None:
            value = row.get("Duration", 0.0)
        durations[idx] = float(value if value is not None else 0.0)
    return durations


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "merge_objective_weights",
    "sequence_objective",
    "random_sequence",
    "processing_times",
]
