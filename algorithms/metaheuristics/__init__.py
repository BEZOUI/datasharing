"""Metaheuristic algorithms available in the framework."""
from algorithms.metaheuristics.ant_colony import AntColonyOptimization
from algorithms.metaheuristics.differential_evolution import DifferentialEvolution
from algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from algorithms.metaheuristics.guided_local_search import GuidedLocalSearch
from algorithms.metaheuristics.iterated_local_search import IteratedLocalSearch
from algorithms.metaheuristics.particle_swarm import ParticleSwarmOptimization
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from algorithms.metaheuristics.tabu_search import TabuSearch
from algorithms.metaheuristics.variable_neighborhood_search import VariableNeighborhoodSearch

__all__ = [
    "AntColonyOptimization",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "GuidedLocalSearch",
    "IteratedLocalSearch",
    "ParticleSwarmOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "VariableNeighborhoodSearch",
]
