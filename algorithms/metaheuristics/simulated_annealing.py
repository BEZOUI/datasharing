"""Simulated annealing metaheuristic for job sequencing."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _sequence_objective(problem: ManufacturingProblem, sequence: Sequence[int], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    schedule = problem.build_schedule(sequence)
    metrics = evaluate_schedule(schedule)
    objective = 0.0
    for key, weight in weights.items():
        if key in metrics:
            objective += weight * metrics[key]
    return objective, metrics


class SimulatedAnnealing(BaseOptimizer):
    """Simple simulated annealing optimiser for job sequencing."""

    def __init__(
        self,
        initial_temperature: float = 250.0,
        cooling_rate: float = 0.95,
        steps_per_temperature: int = 20,
        max_iterations: int = 120,
        seed: int = 7,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            steps_per_temperature=steps_per_temperature,
            max_iterations=max_iterations,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.steps_per_temperature = steps_per_temperature
        self.max_iterations = max_iterations
        self.seed = seed
        self.objective_weights = objective_weights or {"makespan": 1.0, "energy": 0.01}

    def _neighbour(self, sequence: List[int], rng: random.Random) -> List[int]:
        if len(sequence) < 2:
            return sequence.copy()
        i, j = rng.sample(range(len(sequence)), 2)
        neighbour = sequence.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        return neighbour

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = jobs.copy()
        rng.shuffle(current_sequence)
        current_value, current_metrics = _sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence
        best_value = current_value
        best_metrics = current_metrics

        temperature = self.initial_temperature
        iteration = 0

        while temperature > 1e-3 and iteration < self.max_iterations:
            for _ in range(self.steps_per_temperature):
                candidate_sequence = self._neighbour(current_sequence, rng)
                candidate_value, candidate_metrics = _sequence_objective(
                    problem, candidate_sequence, self.objective_weights
                )

                delta = candidate_value - current_value
                if delta < 0 or math.exp(-delta / temperature) > rng.random():
                    current_sequence = candidate_sequence
                    current_value = candidate_value
                    current_metrics = candidate_metrics

                if current_value < best_value:
                    best_sequence = current_sequence.copy()
                    best_value = current_value
                    best_metrics = current_metrics

                iteration += 1
                if iteration >= self.max_iterations:
                    break

            temperature *= self.cooling_rate

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
