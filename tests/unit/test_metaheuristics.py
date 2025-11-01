from __future__ import annotations

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from algorithms import get_algorithm
from algorithms.metaheuristics import (
    AntColonyOptimization,
    DifferentialEvolution,
    GeneticAlgorithm,
    GuidedLocalSearch,
    IteratedLocalSearch,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    TabuSearch,
    VariableNeighborhoodSearch,
)
from problems.job_shop import create_job_shop_problem


def build_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Job_ID": [f"J{i}" for i in range(8)],
            "Machine_ID": ["M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2"],
            "Scheduled_Start": ["2023-01-01T08:00:00"] * 8,
            "Scheduled_End": ["2023-01-01T09:00:00"] * 8,
            "Processing_Time": [40, 65, 55, 30, 45, 70, 60, 35],
            "Energy_Consumption": [12, 10, 11, 9, 13, 12, 14, 10],
            "Due_Date": [
                "2023-01-01T10:00:00",
                "2023-01-01T09:30:00",
                "2023-01-01T09:45:00",
                "2023-01-01T09:50:00",
                "2023-01-01T10:10:00",
                "2023-01-01T10:05:00",
                "2023-01-01T10:15:00",
                "2023-01-01T09:55:00",
            ],
        }
    )


@pytest.mark.parametrize(
    "optimizer_factory",
    [
        lambda: SimulatedAnnealing(max_iterations=20),
        lambda: GeneticAlgorithm(population_size=20, generations=10),
        lambda: ParticleSwarmOptimization(iterations=20, swarm_size=15),
        lambda: AntColonyOptimization(iterations=15, ants=10),
        lambda: TabuSearch(iterations=40, neighbourhood_size=15),
        lambda: VariableNeighborhoodSearch(max_iterations=20),
        lambda: IteratedLocalSearch(iterations=20, perturbation_strength=2),
        lambda: GuidedLocalSearch(iterations=25, lambda_penalty=0.05),
        lambda: DifferentialEvolution(population_size=15, generations=20),
    ],
)
def test_metaheuristics_produce_valid_schedules(optimizer_factory):
    problem = create_job_shop_problem(build_jobs())
    optimizer = optimizer_factory()
    solution = optimizer.solve(problem)
    assert not solution.schedule.empty
    assert solution.metrics["makespan"] > 0


def test_registry_includes_metaheuristics():
    problem = create_job_shop_problem(build_jobs())
    optimizer = get_algorithm("genetic_algorithm", generations=10, population_size=12)
    solution = optimizer.solve(problem)
    assert solution.metrics["makespan"] > 0
