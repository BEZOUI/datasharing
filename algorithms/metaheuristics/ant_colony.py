"""Ant colony optimisation tailored for job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, processing_times, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class AntColonyOptimization(BaseOptimizer):
    """Constructive ACO with pheromone evaporation and heuristic visibility."""

    def __init__(
        self,
        ants: int = 25,
        iterations: int = 60,
        evaporation: float = 0.4,
        alpha: float = 1.0,
        beta: float = 2.0,
        seed: int = 21,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            ants=ants,
            iterations=iterations,
            evaporation=evaporation,
            alpha=alpha,
            beta=beta,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.ants = ants
        self.iterations = iterations
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _select_next(
        self,
        available: List[int],
        pheromones: Dict[int, float],
        durations: Dict[int, float],
        rng: random.Random,
    ) -> int:
        weights: List[float] = []
        for job in available:
            tau = pheromones.get(job, 1.0) ** self.alpha
            eta = (1.0 / (1.0 + durations.get(job, 1.0))) ** self.beta
            weights.append(max(tau * eta, 1e-12))
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for job, weight in zip(available, weights):
            cumulative += weight
            if cumulative >= threshold:
                return job
        return available[-1]

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        durations = processing_times(problem)
        pheromones: Dict[int, float] = {job: 1.0 for job in jobs}
        best_sequence = jobs
        best_value = float("inf")

        for _ in range(self.iterations):
            iteration_best_sequence = None
            iteration_best_value = float("inf")
            for _ in range(self.ants):
                available = jobs[:]
                sequence: List[int] = []
                while available:
                    job = self._select_next(available, pheromones, durations, rng)
                    sequence.append(job)
                    available.remove(job)
                value, _ = sequence_objective(problem, sequence, self.objective_weights)
                if value < iteration_best_value:
                    iteration_best_value = value
                    iteration_best_sequence = sequence
            assert iteration_best_sequence is not None

            for job in pheromones:
                pheromones[job] = (1.0 - self.evaporation) * pheromones[job]
                pheromones[job] = max(pheromones[job], 1e-6)
            deposit = 1.0 / (1.0 + iteration_best_value)
            for job in iteration_best_sequence:
                pheromones[job] = pheromones.get(job, 1.0) + deposit

            if iteration_best_value < best_value:
                best_value = iteration_best_value
                best_sequence = iteration_best_sequence

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
