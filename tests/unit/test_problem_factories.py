from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.generator import BenchmarkDataGenerator
from problems import get_problem_factory, list_problem_types


def _load_reference_dataset() -> pd.DataFrame:
    loader = BenchmarkDataGenerator(root=Path("data/benchmarks"))
    return loader.load_instances([loader.available_instances()[0]])[0]


def test_all_problem_factories_generate_instances() -> None:
    dataset = _load_reference_dataset()
    for name in list_problem_types():
        factory = get_problem_factory(name)
        problem = factory(dataset)
        assert len(problem.jobs) > 0
        assert problem.objectives, f"Problem {name} should define objectives"
