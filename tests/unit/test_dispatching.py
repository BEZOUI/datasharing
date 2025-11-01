from __future__ import annotations

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from algorithms.classical.dispatching_rules import FCFSRule, SPTRule, EDDRule
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from problems.job_shop import create_job_shop_problem


def test_fcfs_returns_sorted_schedule():
    data = pd.DataFrame(
        {
            "Job_ID": ["A", "B"],
            "Machine_ID": ["M1", "M1"],
            "Scheduled_Start": ["2023-01-01T09:00:00", "2023-01-01T08:00:00"],
            "Scheduled_End": ["2023-01-01T10:00:00", "2023-01-01T09:00:00"],
            "Processing_Time": [60, 120],
            "Due_Date": ["2023-01-01T10:00:00", "2023-01-01T08:30:00"],
        }
    )
    problem = create_job_shop_problem(data)
    optimizer = FCFSRule()
    solution = optimizer.solve(problem)
    assert list(solution.schedule["Job_ID"]) == ["B", "A"]
    assert solution.metrics["makespan"] > 0


def test_spt_improves_makespan_over_fcfs():
    data = pd.DataFrame(
        {
            "Job_ID": ["A", "B", "C"],
            "Machine_ID": ["M1", "M1", "M1"],
            "Scheduled_Start": ["2023-01-01T08:00:00", "2023-01-01T08:05:00", "2023-01-01T08:10:00"],
            "Scheduled_End": ["2023-01-01T10:00:00", "2023-01-01T09:00:00", "2023-01-01T09:10:00"],
            "Processing_Time": [120, 55, 45],
            "Due_Date": ["2023-01-01T12:00:00", "2023-01-01T09:30:00", "2023-01-01T09:20:00"],
        }
    )
    problem = create_job_shop_problem(data)
    fcfs = FCFSRule().solve(problem)
    spt = SPTRule().solve(problem)
    assert spt.metrics["makespan"] <= fcfs.metrics["makespan"]


def test_edd_prioritises_due_dates():
    data = pd.DataFrame(
        {
            "Job_ID": ["A", "B"],
            "Machine_ID": ["M1", "M1"],
            "Scheduled_Start": ["2023-01-01T08:00:00", "2023-01-01T08:10:00"],
            "Scheduled_End": ["2023-01-01T08:30:00", "2023-01-01T09:30:00"],
            "Processing_Time": [30, 90],
            "Due_Date": ["2023-01-01T08:45:00", "2023-01-01T08:40:00"],
        }
    )
    problem = create_job_shop_problem(data)
    solution = EDDRule().solve(problem)
    assert list(solution.schedule["Job_ID"]) == ["B", "A"]


def test_simulated_annealing_finds_better_sequence():
    data = pd.DataFrame(
        {
            "Job_ID": ["A", "B", "C", "D"],
            "Machine_ID": ["M1", "M1", "M1", "M1"],
            "Scheduled_Start": ["2023-01-01T08:00:00"] * 4,
            "Scheduled_End": ["2023-01-01T09:00:00", "2023-01-01T10:00:00", "2023-01-01T11:00:00", "2023-01-01T12:00:00"],
            "Processing_Time": [80, 25, 60, 40],
            "Due_Date": ["2023-01-01T09:10:00", "2023-01-01T08:40:00", "2023-01-01T10:30:00", "2023-01-01T11:00:00"],
        }
    )
    problem = create_job_shop_problem(data)
    baseline = FCFSRule().solve(problem)
    annealed = SimulatedAnnealing(seed=3).solve(problem)
    assert annealed.metrics["makespan"] <= baseline.metrics["makespan"]
