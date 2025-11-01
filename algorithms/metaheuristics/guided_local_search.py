"""Guided local search metaheuristic focusing on tardiness penalties."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class GuidedLocalSearch(BaseOptimizer):
    """Implements a simple GLS with feature penalties on tardy jobs."""

    def __init__(
        self,
        iterations: int = 120,
        lambda_penalty: float = 0.1,
        seed: int = 17,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            lambda_penalty=lambda_penalty,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.lambda_penalty = lambda_penalty
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        penalties: Dict[int, float] = {idx: 0.0 for idx in jobs}
        current_sequence = random_sequence(problem, rng)
        current_value, current_metrics = sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence[:]
        best_value = current_value
        best_metrics = current_metrics

        for _ in range(self.iterations):
            neighbourhood = []
            for _ in range(len(current_sequence)):
                i, j = rng.sample(range(len(current_sequence)), 2)
                neighbour = current_sequence[:]
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                neighbourhood.append(neighbour)

            candidate_sequence = current_sequence
            candidate_augmented = float("inf")
            candidate_value = current_value
            candidate_metrics = current_metrics
            for neighbour in neighbourhood:
                value, metrics = sequence_objective(problem, neighbour, self.objective_weights)
                augmented = value + self.lambda_penalty * sum(penalties[idx] for idx in neighbour)
                if augmented < candidate_augmented:
                    candidate_sequence = neighbour
                    candidate_value = value
                    candidate_augmented = augmented
                    candidate_metrics = metrics

            current_sequence = candidate_sequence
            current_value = candidate_value
            current_metrics = candidate_metrics

            if current_value < best_value:
                best_sequence = current_sequence[:]
                best_value = current_value
                best_metrics = current_metrics

            tardiness = current_metrics.get("total_tardiness", 0.0)
            if tardiness > 0:
                for job in current_sequence:
                    penalties[job] += tardiness / len(current_sequence)

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
