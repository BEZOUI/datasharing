"""Adaptive hybrid optimiser that combines multiple strategies."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from algorithms.classical.dispatching_rules import DISPATCHING_RULES
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from core.base_optimizer import BaseOptimizer
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class AdaptiveHybridOptimizer(BaseOptimizer):
    """Select the best schedule among a portfolio of base optimisers."""

    def __init__(self, candidates: Iterable[str] | None = None, **kwargs) -> None:
        if candidates is None:
            candidates = ["fcfs", "spt", "edd", "simulated_annealing"]
        normalised = [name.lower() for name in candidates]
        super().__init__(candidates=normalised, **kwargs)
        self.candidates = normalised

    def _instantiate(self, name: str) -> BaseOptimizer:
        if name in DISPATCHING_RULES:
            return DISPATCHING_RULES[name]()
        if name == "simulated_annealing":
            return SimulatedAnnealing()
        raise ValueError(f"Unknown optimiser '{name}'")

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        results: List[Tuple[str, ScheduleSolution]] = []
        for name in self.candidates:
            optimizer = self._instantiate(name)
            solution = optimizer.solve(problem)
            results.append((name, solution))

        weights = {"makespan": 1.0, "total_tardiness": 0.5, "energy": 0.05}
        def score(metrics: Dict[str, float]) -> float:
            return sum(metrics.get(k, 0.0) * w for k, w in weights.items())

        best_name, best_solution = min(results, key=lambda item: score(item[1].metrics))
        metadata = {
            "selected": best_name,
            "portfolio": {name: sol.metrics for name, sol in results},
        }
        return ScheduleSolution(schedule=best_solution.schedule.copy(), metrics=best_solution.metrics, metadata=metadata)
