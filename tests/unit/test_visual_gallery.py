from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.metrics import evaluate_schedule
from data.generator import BenchmarkDataGenerator
from problems import get_problem_factory
from visualization.gallery import generate_gallery


def test_gallery_generates_over_fifty_figures(tmp_path: Path) -> None:
    loader = BenchmarkDataGenerator(root=Path("data/benchmarks"))
    dataset = loader.load_instances(["taillard_fsp_5x5"])[0]
    problem = get_problem_factory("job_shop")(dataset)
    schedule = problem.build_schedule(problem.job_indices())
    metrics = evaluate_schedule(schedule)
    results = pd.DataFrame(
        [
            {"algorithm": "baseline_dispatch", **metrics},
            {"algorithm": "metaheuristic_a", **{key: value * 0.95 for key, value in metrics.items()}},
            {"algorithm": "metaheuristic_b", **{key: value * 1.05 for key, value in metrics.items()}},
        ]
    )
    paths = generate_gallery(results, schedule, tmp_path)
    assert len(paths) >= 50
    for path in paths:
        assert path.exists()
