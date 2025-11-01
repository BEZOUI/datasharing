"""Iterated local search for manufacturing scheduling."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class IteratedLocalSearch(BaseOptimizer):
    """Repeated perturbation and descent to escape local optima."""

    def __init__(
        self,
        iterations: int = 80,
        perturbation_strength: int = 3,
        seed: int = 13,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            perturbation_strength=perturbation_strength,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.perturbation_strength = perturbation_strength
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _local_descent(self, problem: ManufacturingProblem, sequence: List[int]) -> tuple[List[int], float]:
        current_sequence = sequence[:]
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        improved = True
        rng = random.Random(self.seed + 1)
        while improved:
            improved = False
            for _ in range(len(sequence)):
                i, j = rng.sample(range(len(sequence)), 2)
                candidate = current_sequence[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value, _ = sequence_objective(problem, candidate, self.objective_weights)
                if value < current_value:
                    current_sequence = candidate
                    current_value = value
                    improved = True
                    break
        return current_sequence, current_value

    def _perturb(self, sequence: List[int], rng: random.Random) -> List[int]:
        perturbed = sequence[:]
        for _ in range(self.perturbation_strength):
            i, j = rng.sample(range(len(sequence)), 2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        return perturbed

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = random_sequence(problem, rng)
        current_sequence, current_value = self._local_descent(problem, current_sequence)
        best_sequence = current_sequence
        best_value = current_value

        for _ in range(self.iterations):
            candidate_sequence = self._perturb(current_sequence, rng)
            candidate_sequence, candidate_value = self._local_descent(problem, candidate_sequence)
            if candidate_value < best_value:
                best_sequence = candidate_sequence
                best_value = candidate_value
                current_sequence = candidate_sequence
                current_value = candidate_value
            else:
                current_sequence = candidate_sequence

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
