"""Algorithm registry and utility helpers."""
from __future__ import annotations

from typing import Callable, Dict, List

from algorithms.classical.dispatching_rules import DISPATCHING_RULES, DispatchingRule
from algorithms.classical.constructive_heuristics import NEHHeuristic, PalmerHeuristic
from algorithms.classical.exact_methods import BranchAndBound
from algorithms.deep_rl.dqn import DQNOptimizer
from algorithms.deep_rl.ppo import PPOOptimizer
from algorithms.hybrid.adaptive_hybrid import AdaptiveHybridOptimizer
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
from algorithms.multi_objective.nsga2 import NSGAII
from core.base_optimizer import BaseOptimizer


def get_algorithm(name: str, **kwargs) -> BaseOptimizer:
    """Instantiate an algorithm by name.

    Dispatching rules can be referenced directly by their identifier
    (e.g. ``"spt"``).  Other algorithms expose canonical names matching the
    research roadmap (``"simulated_annealing"``, ``"nsga2"``, ``"dqn"``,
    ``"adaptive_hybrid"``).
    """

    name = name.lower()
    if name in DISPATCHING_RULES:
        return DISPATCHING_RULES[name](**kwargs)

    registry: Dict[str, Callable[..., BaseOptimizer]] = {
        "neh": NEHHeuristic,
        "palmer": PalmerHeuristic,
        "branch_and_bound": BranchAndBound,
        "simulated_annealing": SimulatedAnnealing,
        "genetic_algorithm": GeneticAlgorithm,
        "particle_swarm": ParticleSwarmOptimization,
        "ant_colony": AntColonyOptimization,
        "tabu_search": TabuSearch,
        "variable_neighborhood_search": VariableNeighborhoodSearch,
        "iterated_local_search": IteratedLocalSearch,
        "guided_local_search": GuidedLocalSearch,
        "differential_evolution": DifferentialEvolution,
        "nsga2": NSGAII,
        "dqn": DQNOptimizer,
        "ppo": PPOOptimizer,
        "adaptive_hybrid": AdaptiveHybridOptimizer,
    }
    if name not in registry:
        raise KeyError(f"Unknown algorithm '{name}'")
    return registry[name](**kwargs)


def list_algorithms(include_dispatching: bool = True) -> List[str]:
    """Return the list of registered optimisation algorithms.

    Parameters
    ----------
    include_dispatching:
        When *True*, short-horizon dispatching heuristics are included in
        addition to advanced optimisation methods.  This is particularly
        useful for interactive exploration in the visual dashboard where the
        researcher may want to benchmark simple baselines alongside
        state-of-the-art learners.
    """

    names: List[str] = [
        "neh",
        "palmer",
        "branch_and_bound",
        "simulated_annealing",
        "genetic_algorithm",
        "particle_swarm",
        "ant_colony",
        "tabu_search",
        "variable_neighborhood_search",
        "iterated_local_search",
        "guided_local_search",
        "differential_evolution",
        "nsga2",
        "dqn",
        "ppo",
        "adaptive_hybrid",
    ]
    if include_dispatching:
        names = list(DISPATCHING_RULES.keys()) + names
    return sorted(dict.fromkeys(names))


__all__ = [
    "get_algorithm",
    "DISPATCHING_RULES",
    "DispatchingRule",
    "list_algorithms",
    "NEHHeuristic",
    "PalmerHeuristic",
    "BranchAndBound",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "AntColonyOptimization",
    "TabuSearch",
    "VariableNeighborhoodSearch",
    "IteratedLocalSearch",
    "GuidedLocalSearch",
    "DifferentialEvolution",
    "NSGAII",
    "DQNOptimizer",
    "PPOOptimizer",
    "AdaptiveHybridOptimizer",
]
