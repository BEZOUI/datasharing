# RMS Optimisation Framework

This repository provides a modular research framework for optimisation in
Reconfigurable Manufacturing Systems (RMS).  The architecture follows a
layered design comprising configuration management, data ingestion,
simulation stubs, algorithmic portfolios, experiment orchestration,
visualisation, reporting, and validation utilities.  The goal is to
enable rapid prototyping of novel optimisation strategies while meeting
reproducibility requirements expected from Q1 journal submissions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiments.py --config config/base_config.yaml
```

The baseline script executes a small suite of dispatching rules on the
configured datasets, exports aggregated metrics, and generates a
publication-ready bar chart together with a markdown summary report.

### Interactive dashboard

Launch the graphical control centre when you want to monitor optimisation
progress in real time and trigger comprehensive visual analytics:

```bash
python scripts/run_dashboard.py
```

The dashboard supports loading the bundled benchmark library or custom
CSV/JSON datasets, selecting any subset of optimisation algorithms,
executing them across one or all supported problem formulations, and
rendering Gantt charts of the leading schedule.  After the run, a single
button generates more than fifty publication-grade figures spanning
distributions, Pareto fronts, convergence trajectories, and utilisation
diagnostics compliant with Q1 journal expectations.

## Project layout

- `config/`: Pydantic-backed configuration models and sample YAML files
- `data/`: Data loading, validation, synthetic generation, caching
- `core/`: Shared domain abstractions (problem, solution, metrics)
- `algorithms/`: Portfolios including classical, metaheuristic, RL, and hybrid stubs
- `experiments/`: Experiment manager orchestrating runs and persistence
- `visualization/`: Publication-quality plotting utilities
- `reporting/`: Automated report generation helpers
- `validation/`: Theoretical and empirical validation skeletons
- `scripts/`: Command-line interfaces for executing experiments

The framework is intentionally modular so additional algorithms,
simulators, or validation routines can be contributed without touching
the existing components.
