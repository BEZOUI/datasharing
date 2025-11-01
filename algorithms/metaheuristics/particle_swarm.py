"""Particle swarm optimisation for sequencing jobs using random keys."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _position_to_sequence(position: Sequence[float], jobs: Sequence[int]) -> List[int]:
    return [job for _, job in sorted(zip(position, jobs), key=lambda pair: pair[0])]


class ParticleSwarmOptimization(BaseOptimizer):
    """Continuous random-key PSO for combinatorial scheduling."""

    def __init__(
        self,
        swarm_size: int = 30,
        iterations: int = 80,
        inertia: float = 0.72,
        cognitive: float = 1.49,
        social: float = 1.49,
        seed: int = 3,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            swarm_size=swarm_size,
            iterations=iterations,
            inertia=inertia,
            cognitive=cognitive,
            social=social,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        dimension = len(jobs)
        if dimension == 0:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        particles: List[List[float]] = [[rng.random() for _ in range(dimension)] for _ in range(self.swarm_size)]
        velocities: List[List[float]] = [[0.0 for _ in range(dimension)] for _ in range(self.swarm_size)]

        personal_best: List[Tuple[List[float], float]] = []
        best_global_position: List[float] | None = None
        best_global_value = float("inf")

        for position in particles:
            sequence = _position_to_sequence(position, jobs)
            value, _ = sequence_objective(problem, sequence, self.objective_weights)
            personal_best.append((position[:], value))
            if value < best_global_value:
                best_global_value = value
                best_global_position = position[:]

        for _ in range(self.iterations):
            for idx, position in enumerate(particles):
                velocity = velocities[idx]
                pbest_position, pbest_value = personal_best[idx]
                for d in range(dimension):
                    r1 = rng.random()
                    r2 = rng.random()
                    cognitive_term = self.cognitive * r1 * (pbest_position[d] - position[d])
                    social_term = 0.0
                    if best_global_position is not None:
                        social_term = self.social * r2 * (best_global_position[d] - position[d])
                    velocity[d] = self.inertia * velocity[d] + cognitive_term + social_term
                    position[d] += velocity[d]
                sequence = _position_to_sequence(position, jobs)
                value, _ = sequence_objective(problem, sequence, self.objective_weights)
                if value < pbest_value:
                    personal_best[idx] = (position[:], value)
                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position[:]

        assert best_global_position is not None
        best_sequence = _position_to_sequence(best_global_position, jobs)
        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_global_value, "sequence": best_sequence},
        )
