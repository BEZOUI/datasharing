"""Exact optimisation methods for small instances."""
from __future__ import annotations

from typing import List

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class BranchAndBound(BaseOptimizer):
    """Simple branch-and-bound search exploring job permutations."""

    def __init__(self, max_jobs: int = 8) -> None:
        super().__init__(max_jobs=max_jobs)
        self.max_jobs = max_jobs

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = problem.jobs
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        if len(jobs) > self.max_jobs:
            # Fallback to constructive heuristic for large instances
            from algorithms.classical.constructive_heuristics import NEHHeuristic

            return NEHHeuristic().solve(problem)

        best_sequence: List[int] | None = None
        best_cost = float("inf")
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column required for branch-and-bound optimisation")
        filled_processing = processing.fillna(processing.mean() or 0.0)
        processing_map = filled_processing.to_dict()

        def branch(partial: List[int], remaining: List[int], accumulated: float) -> None:
            nonlocal best_cost, best_sequence
            if not remaining:
                if accumulated < best_cost:
                    best_cost = accumulated
                    best_sequence = partial.copy()
                return

            lower_bound = accumulated + sum(processing_map[idx] for idx in remaining)
            if lower_bound >= best_cost:
                return

            for idx in remaining:
                next_partial = partial + [idx]
                schedule = problem.build_schedule(next_partial)
                cost = evaluate_schedule(schedule)["makespan"]
                if cost >= best_cost:
                    continue
                next_remaining = [j for j in remaining if j != idx]
                branch(next_partial, next_remaining, cost)

        initial_remaining = list(jobs.index)
        branch([], initial_remaining, 0.0)

        if best_sequence is None:
            best_sequence = initial_remaining
        final_schedule = problem.build_schedule(best_sequence)
        return ScheduleSolution(schedule=final_schedule, metadata={"sequence": best_sequence})
