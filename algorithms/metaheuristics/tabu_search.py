"""Tabu search implementation for RMS job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class TabuSearch(BaseOptimizer):
    """Swap-based tabu search with aspiration criteria."""

    def __init__(
        self,
        iterations: int = 150,
        tabu_tenure: int = 8,
        neighbourhood_size: int = 25,
        seed: int = 5,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            tabu_tenure=tabu_tenure,
            neighbourhood_size=neighbourhood_size,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.tabu_tenure = tabu_tenure
        self.neighbourhood_size = neighbourhood_size
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _generate_neighbours(self, sequence: List[int], rng: random.Random) -> List[Tuple[List[int], Tuple[int, int]]]:
        neighbours: List[Tuple[List[int], Tuple[int, int]]] = []
        n = len(sequence)
        for _ in range(self.neighbourhood_size):
            i, j = sorted(rng.sample(range(n), 2))
            neighbour = sequence[:]
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
            neighbours.append((neighbour, (i, j)))
        return neighbours

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = random_sequence(problem, rng)
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence[:]
        best_value = current_value

        tabu_list: Dict[Tuple[int, int], int] = {}

        for iteration in range(self.iterations):
            neighbours = self._generate_neighbours(current_sequence, rng)
            candidate_sequence = None
            candidate_value = float("inf")
            candidate_move = (0, 0)
            for neighbour_sequence, move in neighbours:
                value, _ = sequence_objective(problem, neighbour_sequence, self.objective_weights)
                if value < candidate_value and (
                    move not in tabu_list or iteration >= tabu_list[move] or value < best_value
                ):
                    candidate_sequence = neighbour_sequence
                    candidate_value = value
                    candidate_move = move
            if candidate_sequence is None:
                continue
            current_sequence = candidate_sequence
            current_value = candidate_value
            tabu_list[candidate_move] = iteration + self.tabu_tenure
            if current_value < best_value:
                best_value = current_value
                best_sequence = current_sequence[:]

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
