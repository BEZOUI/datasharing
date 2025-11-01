"""Genetic algorithm for sequencing jobs in manufacturing problems."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class GeneticAlgorithm(BaseOptimizer):
    """Order-based genetic algorithm with partially mapped crossover."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 60,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
        seed: int = 42,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elitism=elitism,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _fitness(self, problem: ManufacturingProblem, sequence: Sequence[int]) -> Tuple[float, Dict[str, float]]:
        value, metrics = sequence_objective(problem, sequence, self.objective_weights)
        return value, metrics

    def _tournament(self, population: List[List[int]], scores: List[float], rng: random.Random) -> List[int]:
        candidates = rng.sample(range(len(population)), self.tournament_size)
        best = min(candidates, key=lambda idx: scores[idx])
        return population[best][:]

    def _crossover(self, parent_a: List[int], parent_b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
        size = len(parent_a)
        if size < 2:
            return parent_a[:], parent_b[:]
        start, end = sorted(rng.sample(range(size), 2))
        child_a = [None] * size
        child_b = [None] * size
        child_a[start:end] = parent_a[start:end]
        child_b[start:end] = parent_b[start:end]

        def fill(child: List[int], donor: List[int], start: int, end: int) -> None:
            idx = end
            for gene in donor:
                if gene not in child:
                    if idx >= size:
                        idx = 0
                    child[idx] = gene
                    idx += 1

        fill(child_a, parent_b, start, end)
        fill(child_b, parent_a, start, end)
        return child_a, child_b

    def _mutate(self, sequence: List[int], rng: random.Random) -> None:
        if len(sequence) < 2:
            return
        i, j = rng.sample(range(len(sequence)), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population = [random_sequence(problem, rng) for _ in range(self.population_size)]
        best_sequence = population[0]
        best_value = float("inf")
        best_metrics: Dict[str, float] = {}

        for _ in range(self.generations):
            scores: List[float] = []
            metrics_store: List[Dict[str, float]] = []
            for individual in population:
                value, metrics = self._fitness(problem, individual)
                scores.append(value)
                metrics_store.append(metrics)
                if value < best_value:
                    best_value = value
                    best_sequence = individual[:]
                    best_metrics = metrics

            ranked = sorted(zip(population, scores, metrics_store), key=lambda item: item[1])
            new_population: List[List[int]] = [ind[:] for ind, _, _ in ranked[: self.elitism]]

            while len(new_population) < self.population_size:
                parent_a = self._tournament(population, scores, rng)
                parent_b = self._tournament(population, scores, rng)
                child_a, child_b = parent_a[:], parent_b[:]
                if rng.random() < self.crossover_rate:
                    child_a, child_b = self._crossover(parent_a, parent_b, rng)
                if rng.random() < self.mutation_rate:
                    self._mutate(child_a, rng)
                if rng.random() < self.mutation_rate:
                    self._mutate(child_b, rng)
                new_population.append(child_a)
                if len(new_population) < self.population_size:
                    new_population.append(child_b)

            population = new_population

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
