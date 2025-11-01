"""Differential evolution using random keys for job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

from algorithms.metaheuristics.utils import merge_objective_weights, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _keys_to_sequence(keys: Sequence[float], jobs: Sequence[int]) -> List[int]:
    return [job for _, job in sorted(zip(keys, jobs), key=lambda item: item[0])]


class DifferentialEvolution(BaseOptimizer):
    """Classic DE/rand/1/bin adapted to combinatorial scheduling."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 80,
        crossover_rate: float = 0.7,
        differential_weight: float = 0.8,
        seed: int = 19,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            differential_weight=differential_weight,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.differential_weight = differential_weight
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        dimension = len(jobs)
        if dimension == 0:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population: List[List[float]] = [[rng.random() for _ in range(dimension)] for _ in range(self.population_size)]
        scores = [sequence_objective(problem, _keys_to_sequence(individual, jobs), self.objective_weights)[0] for individual in population]

        for _ in range(self.generations):
            for idx in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(idx)
                a, b, c = rng.sample(candidates, 3)
                base = population[a]
                diff1 = population[b]
                diff2 = population[c]
                mutant = [base[d] + self.differential_weight * (diff1[d] - diff2[d]) for d in range(dimension)]
                trial = population[idx][:]
                j_rand = rng.randrange(dimension)
                for d in range(dimension):
                    if rng.random() < self.crossover_rate or d == j_rand:
                        trial[d] = mutant[d]
                trial_score = sequence_objective(problem, _keys_to_sequence(trial, jobs), self.objective_weights)[0]
                if trial_score < scores[idx]:
                    population[idx] = trial
                    scores[idx] = trial_score

        best_index = min(range(self.population_size), key=lambda i: scores[i])
        best_sequence = _keys_to_sequence(population[best_index], jobs)
        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": scores[best_index], "sequence": best_sequence},
        )
