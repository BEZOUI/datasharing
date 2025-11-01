"""Problem registry to simplify experiment configuration."""
from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from core.problem import ManufacturingProblem
from problems.flexible_job_shop import create_flexible_job_shop_problem
from problems.flow_shop import create_flow_shop_problem
from problems.job_shop import create_job_shop_problem
from problems.rms_variants import (
    create_distributed_job_shop_problem,
    create_dynamic_job_shop_problem,
    create_hybrid_manufacturing_problem,
)

ProblemFactory = Callable[[pd.DataFrame], ManufacturingProblem]


PROBLEM_FACTORIES: Dict[str, ProblemFactory] = {
    "job_shop": create_job_shop_problem,
    "flow_shop": create_flow_shop_problem,
    "flexible_job_shop": create_flexible_job_shop_problem,
    "dynamic_job_shop": create_dynamic_job_shop_problem,
    "distributed_job_shop": create_distributed_job_shop_problem,
    "hybrid_manufacturing": create_hybrid_manufacturing_problem,
}


def get_problem_factory(name: str) -> ProblemFactory:
    key = name.lower()
    if key not in PROBLEM_FACTORIES:
        raise KeyError(f"Unknown problem factory '{name}'")
    return PROBLEM_FACTORIES[key]


def list_problem_types() -> List[str]:
    return sorted(PROBLEM_FACTORIES.keys())


__all__ = [
    "PROBLEM_FACTORIES",
    "get_problem_factory",
    "list_problem_types",
]
