"""Monte Carlo simulation helper."""
from __future__ import annotations

from typing import Callable

from statistics import mean


class MonteCarloEngine:
    def __init__(self, repetitions: int) -> None:
        self.repetitions = repetitions

    def estimate(self, func: Callable[[], float]) -> float:
        samples = [func() for _ in range(self.repetitions)]
        return float(mean(samples)) if samples else 0.0
