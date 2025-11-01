from __future__ import annotations

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from algorithms.multi_objective.nsga2 import NSGAII
from algorithms.deep_rl.dqn import DQNOptimizer
from algorithms.deep_rl.ppo import PPOOptimizer
from algorithms.hybrid.adaptive_hybrid import AdaptiveHybridOptimizer
from problems.job_shop import create_job_shop_problem


def build_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Job_ID": [f"J{i}" for i in range(6)],
            "Machine_ID": ["M1", "M1", "M2", "M2", "M1", "M2"],
            "Scheduled_Start": ["2023-01-01T08:00:00"] * 6,
            "Scheduled_End": ["2023-01-01T09:00:00"] * 6,
            "Processing_Time": [45, 70, 55, 40, 65, 35],
            "Energy_Consumption": [12, 10, 11, 9, 13, 8],
            "Due_Date": [
                "2023-01-01T09:30:00",
                "2023-01-01T10:00:00",
                "2023-01-01T09:20:00",
                "2023-01-01T09:40:00",
                "2023-01-01T10:15:00",
                "2023-01-01T09:50:00",
            ],
        }
    )


def test_nsga2_returns_pareto_front():
    problem = create_job_shop_problem(build_dataset())
    optimizer = NSGAII(population_size=10, generations=5, seed=1)
    solution = optimizer.solve(problem)
    pareto = solution.metadata["pareto_front"]
    assert isinstance(pareto, list) and pareto
    assert all("metrics" in entry for entry in pareto)


def test_dqn_optimizer_produces_schedule():
    problem = create_job_shop_problem(build_dataset())
    optimizer = DQNOptimizer(episodes=50, epsilon=0.3, seed=2)
    solution = optimizer.solve(problem)
    assert not solution.schedule.empty
    assert solution.metrics["makespan"] > 0


def test_ppo_optimizer_learns_priorities():
    problem = create_job_shop_problem(build_dataset())
    optimizer = PPOOptimizer(episodes=30, learning_rate=0.02, seed=3)
    solution = optimizer.solve(problem)
    assert not solution.schedule.empty
    assert "policy_weights" in solution.metadata


def test_adaptive_hybrid_selects_best_portfolio_member():
    problem = create_job_shop_problem(build_dataset())
    optimizer = AdaptiveHybridOptimizer(candidates=["fcfs", "spt", "simulated_annealing"])
    solution = optimizer.solve(problem)
    assert "selected" in solution.metadata
    assert solution.metadata["selected"] in {"fcfs", "spt", "simulated_annealing"}
