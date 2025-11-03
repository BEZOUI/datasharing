"""Unified RMS optimisation pipeline in a single executable module.

This script offers a convenience façade over the modular research
framework contained in this repository.  It orchestrates data loading,
problem construction, optimisation algorithm execution, statistical
validation, reporting, and visual analytics from one entry point.  The
original project intentionally separates these concerns into multiple
packages; however some users prefer a monolithic runner they can launch
without navigating the entire codebase.  `rms_all_in_one.py` fulfils that
requirement while reusing the rigorously tested building blocks.

Usage examples
--------------

Run the full experiment workflow using the default configuration and
produce summary artefacts in ``results/all_in_one``::

    python rms_all_in_one.py --run-experiments

Generate the publication gallery and markdown report for all bundled
problem types and algorithms, exporting outputs to a custom directory::

    python rms_all_in_one.py --run-experiments --generate-gallery \
        --all-problems --algorithms all --output-dir results/full_suite

Launch the interactive dashboard directly from this façade::

    python rms_all_in_one.py --launch-dashboard

The script remains lightweight: it imports modules only when required and
fails gracefully when optional dependencies (for example the Tkinter GUI
stack or SciPy) are unavailable in the current environment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from algorithms import get_algorithm, list_algorithms
from config.base_config import ExperimentalConfig, load_config
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution
from data.generator import BenchmarkDataGenerator, SyntheticDataGenerator, SyntheticScenario
from data.loader import DataLoader, DataPreprocessor
from problems import get_problem_factory, list_problem_types
from reporting.generators import MarkdownReporter
from simulation.monte_carlo import MonteCarloEngine
from simulation.stochastic_models import ProcessingTimeModel
from validation.empirical import confidence_interval, friedman_test
from validation.theoretical import document_complexity
from visualization.gallery import generate_gallery

try:  # pragma: no cover - optional dependency for dashboard usage
    from visualization.dashboard import RMSDashboard, tkinter_available
except Exception:  # pragma: no cover - guard against GUI-less systems
    RMSDashboard = None  # type: ignore
    tkinter_available = lambda: False  # type: ignore


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve_algorithms(config: ExperimentalConfig, override: Sequence[str] | None) -> List[str]:
    if override:
        if len(override) == 1 and override[0].lower() == "all":
            return list_algorithms(include_dispatching=True)
        return [name.lower() for name in override]

    hyper = config.algorithm.hyperparameters or {}
    candidates = hyper.get("candidates")
    if candidates:
        return [name.lower() for name in candidates]

    name = config.algorithm.name.lower()
    if name == "all_dispatching":
        from algorithms.classical.dispatching_rules import list_dispatching_rules

        return list_dispatching_rules()
    return [name]


def _load_dataset(config: ExperimentalConfig, synthetic: bool = False) -> pd.DataFrame:
    loader = DataLoader()
    preprocessor = DataPreprocessor()

    frames: List[pd.DataFrame] = []
    if synthetic:
        scenario = SyntheticScenario(
            num_jobs=240,
            machines=["M01", "M02", "M03", "M04"],
            start_date=pd.Timestamp("2024-01-01"),
            time_between_jobs=pd.Timedelta(minutes=12),
        )
        frames.append(SyntheticDataGenerator().generate(scenario))
    elif config.data.sources:
        sources = [Path(source) for source in config.data.sources]
        data = loader.load(sources)
        frames.append(data)
    else:
        generator = BenchmarkDataGenerator()
        frames.extend(generator.load_instances())

    if not frames:
        raise RuntimeError("No datasets were loaded; provide --synthetic or configure data.sources")

    dataset = pd.concat(frames, ignore_index=True)
    return preprocessor.transform(dataset)


def _build_problem(dataset: pd.DataFrame, problem_name: str, config: ExperimentalConfig) -> ManufacturingProblem:
    factory = get_problem_factory(problem_name)
    problem = factory(dataset.copy())
    problem.metadata = {
        "problem_type": problem_name,
        "objectives": ", ".join(config.optimisation.objectives),
    }
    return problem


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def _run_algorithms(
    problem: ManufacturingProblem,
    algorithm_names: Sequence[str],
    rep: int,
) -> Tuple[pd.DataFrame, ScheduleSolution]:
    records: List[Dict[str, float]] = []
    best_solution: Optional[ScheduleSolution] = None
    best_score = float("inf")

    for name in algorithm_names:
        optimizer = get_algorithm(name)
        solution = optimizer.solve(problem)
        metrics = solution.metrics or evaluate_schedule(solution.schedule)
        record = {"replication": rep, "algorithm": name}
        record.update(metrics)
        records.append(record)
        objective_value = metrics.get("makespan", 0.0)
        if objective_value < best_score:
            best_score = objective_value
            best_solution = solution

    assert best_solution is not None, "At least one algorithm must be executed"
    return pd.DataFrame(records), best_solution


def run_experiments(
    config: ExperimentalConfig,
    dataset: pd.DataFrame,
    problems: Sequence[str],
    algorithm_names: Sequence[str],
    output_dir: Path,
    replications: Optional[int] = None,
    generate_gallery_flag: bool = False,
    run_validation: bool = False,
) -> Dict[str, Dict[str, float]]:
    replications = replications or config.validation.replications
    aggregated_metrics: Dict[str, Dict[str, float]] = {}

    gallery_paths: List[Path] = []
    validation_results: Dict[str, Dict[str, float]] = {}

    for problem_name in problems:
        reporter = MarkdownReporter(output_dir / f"{problem_name}_summary.md")
        problem_records: List[pd.DataFrame] = []
        best_schedule_overall: Optional[pd.DataFrame] = None

        best_problem_makespan = float("inf")

        for rep in range(replications):
            problem_instance = _build_problem(dataset, problem_name, config)
            df_records, best_solution = _run_algorithms(problem_instance, algorithm_names, rep)
            df_records["problem"] = problem_name
            problem_records.append(df_records)

            current_best = df_records.loc[df_records["makespan"].idxmin()]
            if current_best["makespan"] < best_problem_makespan or best_schedule_overall is None:
                best_problem_makespan = float(current_best["makespan"])
                best_schedule_overall = best_solution.schedule.copy()
                best_schedule_overall["Algorithm"] = current_best["algorithm"]

        combined = pd.concat(problem_records, ignore_index=True)
        grouped = (
            combined.groupby("algorithm")
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("makespan")
        )
        aggregated_metrics[problem_name] = grouped.set_index("algorithm").iloc[0].to_dict()

        summary_metrics = {
            f"avg_{metric}": float(mean(grouped[metric]))
            for metric in grouped.columns
            if metric != "algorithm"
        }
        summary_metrics["problem"] = problem_name
        reporter.render(summary_metrics, grouped)

        csv_path = output_dir / f"results_{problem_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(csv_path, index=False)

        if generate_gallery_flag and best_schedule_overall is not None:
            gallery_root = output_dir / "figures" / problem_name
            gallery_root.mkdir(parents=True, exist_ok=True)
            gallery_paths.extend(
                generate_gallery(
                    results=grouped,
                    schedule=best_schedule_overall,
                    output_dir=gallery_root,
                    significance_metric="makespan",
                )
            )

        if run_validation:
            try:
                friedman = friedman_test(combined[["replication", "algorithm", "makespan"]])
            except RuntimeError as exc:
                friedman = {"error": str(exc)}
            validation_results[problem_name] = friedman

            try:
                import numpy as np

                ci = confidence_interval(
                    np.array(combined["makespan"], dtype=float),
                    level=config.validation.confidence_level,
                )
                validation_results[problem_name].update({f"ci_{k}": v for k, v in ci.items()})
            except Exception as exc:  # pragma: no cover - optional deps
                validation_results[problem_name].setdefault("ci_error", str(exc))

    if gallery_paths:
        (output_dir / "figures" / "manifest.json").write_text(
            json.dumps([str(path) for path in gallery_paths], indent=2),
            encoding="utf-8",
        )

    if validation_results:
        (output_dir / "statistics" / "validation.json").write_text(
            json.dumps(validation_results, indent=2),
            encoding="utf-8",
        )

    complexities = []
    complexity_map = {
        "fcfs": ("O(n)", "O(1)"),
        "spt": ("O(n log n)", "O(1)"),
        "lpt": ("O(n log n)", "O(1)"),
        "edd": ("O(n log n)", "O(1)"),
        "slack": ("O(n log n)", "O(1)"),
        "critical_ratio": ("O(n log n)", "O(1)"),
        "wspt": ("O(n log n)", "O(1)"),
        "genetic_algorithm": ("O(g * p * n)", "O(p * n)"),
        "particle_swarm": ("O(i * s * n)", "O(s * n)"),
        "simulated_annealing": ("O(i * n)", "O(n)"),
        "tabu_search": ("O(i * n^2)", "O(n^2)"),
        "ant_colony": ("O(i * a * n^2)", "O(a * n)"),
        "nsga2": ("O(g * p^2)", "O(p * n)"),
        "dqn": ("O(e * b)", "O(b)"),
        "ppo": ("O(e * b)", "O(b)"),
        "adaptive_hybrid": ("O(i * n log n)", "O(n^2)"),
    }
    for name in algorithm_names:
        time_c, space_c = complexity_map.get(name, ("unspecified", "unspecified"))
        complexities.append(document_complexity(name, time_c, space_c))
    (output_dir / "statistics" / "complexity.json").write_text(
        json.dumps(complexities, indent=2),
        encoding="utf-8",
    )

    return aggregated_metrics


# ---------------------------------------------------------------------------
# Simulation façade
# ---------------------------------------------------------------------------


def run_monte_carlo(dataset: pd.DataFrame, config: ExperimentalConfig, output_dir: Path) -> None:
    repetitions = config.simulation.repetitions
    engine = MonteCarloEngine(repetitions)

    rng = random.Random(config.algorithm.seed)

    def _lognormal_distribution(size: int) -> List[float]:
        return [max(1.0, rng.lognormvariate(4.0, 0.35)) for _ in range(size)]

    model = ProcessingTimeModel(distribution=_lognormal_distribution)

    def _simulate_once() -> float:
        samples = model.sample(len(dataset))
        return float(sum(samples))

    estimate = engine.estimate(_simulate_once)
    output = {
        "repetitions": repetitions,
        "jobs": int(len(dataset)),
        "expected_total_processing_time": estimate,
    }
    output_path = output_dir / "statistics" / "monte_carlo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------


def launch_dashboard() -> None:  # pragma: no cover - interactive component
    if not tkinter_available():
        raise RuntimeError("Tkinter is not available in this environment")
    import tkinter as tk

    root = tk.Tk()
    RMSDashboard(root)  # type: ignore[arg-type]
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RMS optimisation runner")
    parser.add_argument("--config", type=Path, default=Path("config/base_config.yaml"), help="Path to configuration file")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Algorithms to execute (use 'all' for the full registry)")
    parser.add_argument("--problem", dest="problems", action="append", help="Problem type to evaluate. Repeat for multiple problems.")
    parser.add_argument("--all-problems", action="store_true", help="Evaluate every bundled problem factory")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic dataset instead of loading from disk")
    parser.add_argument("--output-dir", type=Path, default=Path("results/all_in_one"), help="Directory where artefacts are stored")
    parser.add_argument("--replications", type=int, default=None, help="Number of independent replications per algorithm")
    parser.add_argument("--run-experiments", action="store_true", help="Execute the optimisation experiments")
    parser.add_argument("--generate-gallery", action="store_true", help="Produce the 50+ figure gallery after experiments")
    parser.add_argument("--run-validation", action="store_true", help="Compute statistical validation metrics")
    parser.add_argument("--run-simulation", action="store_true", help="Execute the Monte Carlo processing time study")
    parser.add_argument("--launch-dashboard", action="store_true", help="Start the interactive Tkinter dashboard")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.launch_dashboard:
        launch_dashboard()
        return 0

    dataset = _load_dataset(config, synthetic=args.synthetic)

    problems = list_problem_types() if args.all_problems else (args.problems or ["job_shop"])
    algorithms = _resolve_algorithms(config, args.algorithms)

    summary: Dict[str, Dict[str, float]] = {}
    if args.run_experiments:
        summary = run_experiments(
            config=config,
            dataset=dataset,
            problems=problems,
            algorithm_names=algorithms,
            output_dir=output_dir,
            replications=args.replications,
            generate_gallery_flag=args.generate_gallery,
            run_validation=args.run_validation,
        )

    if args.run_simulation:
        run_monte_carlo(dataset, config, output_dir)

    if summary:
        (output_dir / "statistics" / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - script execution
    sys.exit(main())
