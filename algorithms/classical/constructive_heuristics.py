"""Constructive heuristics for flow-shop style problems."""
from __future__ import annotations

from typing import List

import numpy as np

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class NEHHeuristic(BaseOptimizer):
    """Implementation of the classic Nawaz-Enscore-Ham heuristic."""

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        jobs = problem.jobs.copy()
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column is required for NEH heuristic")

        # Sort jobs by decreasing processing time.
        ordered_indices = list(processing.sort_values(ascending=False).index)
        sequence: List[int] = []

        for job in ordered_indices:
            best_sequence: List[int] | None = None
            best_cost = float("inf")
            for position in range(len(sequence) + 1):
                candidate = sequence[:position] + [job] + sequence[position:]
                schedule = problem.build_schedule(candidate)
                cost = evaluate_schedule(schedule)["makespan"]
                if cost < best_cost:
                    best_cost = cost
                    best_sequence = candidate
            assert best_sequence is not None  # for mypy / static typing
            sequence = best_sequence

        final_schedule = problem.build_schedule(sequence)
        return ScheduleSolution(schedule=final_schedule, metadata={"sequence": sequence})


class PalmerHeuristic(BaseOptimizer):
    """Palmer's slope index heuristic for flow shop scheduling."""

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        jobs = problem.jobs.copy()
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column is required for Palmer heuristic")

        machines = jobs.get("Machine_ID")
        if machines is not None:
            machine_dummies = np.vstack([machines == m for m in sorted(machines.unique())]).astype(float)
            slope_index = machine_dummies.T @ np.linspace(-1, 1, machine_dummies.shape[0])
            slope_index = slope_index.flatten()
        else:
            slope_index = np.linspace(-1, 1, len(jobs))

        priority = slope_index * processing.to_numpy()
        ordered = jobs.assign(_priority=priority).sort_values("_priority", ascending=True)
        schedule = problem.build_schedule(ordered.index)
        return ScheduleSolution(schedule=schedule)
