"""Experiment orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from algorithms import get_algorithm
from config.base_config import ExperimentalConfig
from core.problem import ManufacturingProblem


@dataclass
class ExperimentResult:
    algorithm: str
    metrics: Dict[str, float]


class ExperimentManager:
    """Coordinate data loading, algorithm execution, and metric logging."""

    def __init__(self, config: ExperimentalConfig) -> None:
        self.config = config

    def _algorithm_names(self) -> Iterable[str]:
        requested = (
            self.config.algorithm.hyperparameters.get("candidates")
            if self.config.algorithm.hyperparameters
            else None
        )
        if requested:
            return [name.lower() for name in requested]
        name = self.config.algorithm.name.lower()
        if name == "all_dispatching":
            from algorithms.classical.dispatching_rules import list_dispatching_rules

            return list_dispatching_rules()
        return [name]

    def run(self, problem: ManufacturingProblem) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for name in self._algorithm_names():
            optimizer = get_algorithm(name)
            solution = optimizer.solve(problem)
            results.append(ExperimentResult(algorithm=name, metrics=solution.metrics))
        return results

    def summarise(self, results: List[ExperimentResult]) -> pd.DataFrame:
        return pd.DataFrame([{"algorithm": r.algorithm, **r.metrics} for r in results])


def export_results(results: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
