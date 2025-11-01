"""Light-weight NSGA-II implementation for sequencing problems."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


Individual = Dict[str, object]


def _evaluate(problem: ManufacturingProblem, sequence: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, float]]:
    schedule = problem.build_schedule(sequence)
    metrics = evaluate_schedule(schedule)
    objectives = {key: metrics.get(key, 0.0) for key in ["makespan", "energy", "total_tardiness"]}
    return objectives, metrics


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    better_or_equal = all(a[key] <= b[key] for key in a)
    strictly_better = any(a[key] < b[key] for key in a)
    return better_or_equal and strictly_better


def _fast_nondominated_sort(population: List[Individual]) -> List[List[Individual]]:
    fronts: List[List[Individual]] = []
    for individual in population:
        individual["dominated_set"] = []
        individual["domination_count"] = 0
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if _dominates(p["objectives"], q["objectives"]):
                p["dominated_set"].append(q)
            elif _dominates(q["objectives"], p["objectives"]):
                p["domination_count"] += 1
        if p["domination_count"] == 0:
            p["rank"] = 0
            if not fronts:
                fronts.append([])
            fronts[0].append(p)
    current_rank = 0
    while current_rank < len(fronts):
        next_front: List[Individual] = []
        for p in fronts[current_rank]:
            for q in p["dominated_set"]:
                q["domination_count"] -= 1
                if q["domination_count"] == 0:
                    q["rank"] = current_rank + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current_rank += 1
    return fronts


def _crowding_distance(front: List[Individual], objectives: Sequence[str]) -> None:
    if not front:
        return
    for individual in front:
        individual["crowding_distance"] = 0.0
    for objective in objectives:
        front.sort(key=lambda ind: ind["objectives"][objective])
        front[0]["crowding_distance"] = float("inf")
        front[-1]["crowding_distance"] = float("inf")
        values = [ind["objectives"][objective] for ind in front]
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            continue
        for i in range(1, len(front) - 1):
            prev_val = front[i - 1]["objectives"][objective]
            next_val = front[i + 1]["objectives"][objective]
            front[i]["crowding_distance"] += (next_val - prev_val) / (max_val - min_val)


def _tournament_selection(population: List[Individual], k: int, rng: random.Random) -> Individual:
    contenders = rng.sample(population, k)
    contenders.sort(key=lambda ind: (ind["rank"], -ind["crowding_distance"]))
    return contenders[0]


def _pmx_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    cx_point1, cx_point2 = sorted(rng.sample(range(size), 2))
    child1 = parent1[:]
    child2 = parent2[:]
    child1[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
    child2[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    def repair(child: List[int], segment: List[int], donor: List[int]) -> None:
        mapping = {donor[i]: segment[i] for i in range(cx_point1, cx_point2)}
        for idx in list(range(cx_point1)) + list(range(cx_point2, size)):
            while child[idx] in mapping:
                child[idx] = mapping[child[idx]]

    repair(child1, child1, parent1)
    repair(child2, child2, parent2)
    return child1, child2


def _swap_mutation(sequence: List[int], rng: random.Random) -> List[int]:
    i, j = rng.sample(range(len(sequence)), 2)
    sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence


class NSGAII(BaseOptimizer):
    """A compact NSGA-II optimiser suitable for small instances."""

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 30,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.2,
        tournament_size: int = 2,
        seed: int = 13,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            tournament_size=tournament_size,
            seed=seed,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.seed = seed

    def _create_individual(self, job_indices: List[int], rng: random.Random, problem: ManufacturingProblem) -> Individual:
        sequence = job_indices.copy()
        rng.shuffle(sequence)
        objectives, metrics = _evaluate(problem, sequence)
        return {"sequence": sequence, "objectives": objectives, "metrics": metrics}

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        job_indices = list(problem.jobs.index)
        if not job_indices:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population = [self._create_individual(job_indices, rng, problem) for _ in range(self.population_size)]

        objectives = ["makespan", "energy", "total_tardiness"]

        for _ in range(self.generations):
            fronts = _fast_nondominated_sort(population)
            for front in fronts:
                _crowding_distance(front, objectives)

            mating_pool: List[Individual] = []
            while len(mating_pool) < self.population_size:
                mating_pool.append(_tournament_selection(population, self.tournament_size, rng))

            offspring: List[Individual] = []
            for i in range(0, self.population_size, 2):
                parent1 = mating_pool[i % len(mating_pool)]
                parent2 = mating_pool[(i + 1) % len(mating_pool)]
                seq1 = parent1["sequence"].copy()
                seq2 = parent2["sequence"].copy()
                if rng.random() < self.crossover_probability:
                    seq1, seq2 = _pmx_crossover(seq1, seq2, rng)
                if rng.random() < self.mutation_probability:
                    seq1 = _swap_mutation(seq1, rng)
                if rng.random() < self.mutation_probability:
                    seq2 = _swap_mutation(seq2, rng)
                for seq in (seq1, seq2):
                    objectives_values, metrics = _evaluate(problem, seq)
                    offspring.append({"sequence": seq, "objectives": objectives_values, "metrics": metrics})

            combined = population + offspring
            fronts = _fast_nondominated_sort(combined)
            new_population: List[Individual] = []
            for front in fronts:
                _crowding_distance(front, objectives)
                front.sort(key=lambda ind: (ind["rank"], -ind["crowding_distance"]))
                for individual in front:
                    if len(new_population) < self.population_size:
                        new_population.append(individual)
            population = new_population

        fronts = _fast_nondominated_sort(population)
        pareto_front = [
            {
                "sequence": individual["sequence"],
                "metrics": individual["metrics"],
                "objectives": individual["objectives"],
            }
            for individual in fronts[0]
        ]
        best = min(fronts[0], key=lambda ind: ind["objectives"]["makespan"])
        best_schedule = problem.build_schedule(best["sequence"])
        return ScheduleSolution(
            schedule=best_schedule,
            metrics=best["metrics"],
            metadata={"pareto_front": pareto_front},
        )
