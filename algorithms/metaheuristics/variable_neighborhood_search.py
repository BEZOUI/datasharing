"""Variable neighbourhood search for adaptive job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class VariableNeighborhoodSearch(BaseOptimizer):
    """Implements a shaking and local improvement loop with three neighbourhoods."""

    def __init__(
        self,
        max_iterations: int = 120,
        seed: int = 11,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(max_iterations=max_iterations, seed=seed, objective_weights=objective_weights)
        self.max_iterations = max_iterations
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _swap(self, sequence: List[int], rng: random.Random) -> List[int]:
        i, j = rng.sample(range(len(sequence)), 2)
        seq = sequence[:]
        seq[i], seq[j] = seq[j], seq[i]
        return seq

    def _insert(self, sequence: List[int], rng: random.Random) -> List[int]:
        seq = sequence[:]
        i, j = rng.sample(range(len(sequence)), 2)
        value = seq.pop(i)
        seq.insert(j, value)
        return seq

    def _reverse(self, sequence: List[int], rng: random.Random) -> List[int]:
        seq = sequence[:]
        i, j = sorted(rng.sample(range(len(sequence)), 2))
        seq[i:j] = reversed(seq[i:j])
        return seq

    def _local_search(self, problem: ManufacturingProblem, sequence: List[int], rng: random.Random) -> List[int]:
        improved = True
        current_sequence = sequence[:]
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        while improved:
            improved = False
            for neighbour_generator in (self._swap, self._insert, self._reverse):
                neighbour = neighbour_generator(current_sequence, rng)
                value, _ = sequence_objective(problem, neighbour, self.objective_weights)
                if value < current_value:
                    current_sequence = neighbour
                    current_value = value
                    improved = True
                    break
        return current_sequence

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        best_sequence = random_sequence(problem, rng)
        best_value, _ = sequence_objective(problem, best_sequence, self.objective_weights)

        for _ in range(self.max_iterations):
            current_sequence = best_sequence[:]
            for neighbourhood in (self._swap, self._insert, self._reverse):
                shaken = neighbourhood(current_sequence, rng)
                improved = self._local_search(problem, shaken, rng)
                value, _ = sequence_objective(problem, improved, self.objective_weights)
                if value < best_value:
                    best_sequence = improved
                    best_value = value
                    break

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
