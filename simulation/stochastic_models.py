"""Stochastic models for manufacturing processes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class ProcessingTimeModel:
    distribution: Callable[[int], List[float]]

    def sample(self, size: int) -> List[float]:
        return self.distribution(size)
