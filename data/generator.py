"""Synthetic data generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import random

import pandas as pd


@dataclass
class SyntheticScenario:
    """Scenario configuration for synthetic dataset creation."""

    num_jobs: int
    machines: Sequence[str]
    start_date: datetime
    time_between_jobs: timedelta


class SyntheticDataGenerator:
    """Generate synthetic manufacturing datasets."""

    def generate(self, scenario: SyntheticScenario) -> pd.DataFrame:
        rng = random.Random()
        timestamps = [
            scenario.start_date + i * scenario.time_between_jobs for i in range(scenario.num_jobs)
        ]
        machine_choices = list(scenario.machines)
        machines = [rng.choice(machine_choices) for _ in range(scenario.num_jobs)]
        processing_time = [rng.randrange(10, 240) for _ in range(scenario.num_jobs)]
        energy = [max(1.0, rng.gauss(15, 5)) for _ in range(scenario.num_jobs)]
        due_dates = [
            ts + timedelta(minutes=int(pt * rng.uniform(1.2, 1.8)))
            for ts, pt in zip(timestamps, processing_time)
        ]
        priorities = [rng.uniform(1.0, 3.0) for _ in range(scenario.num_jobs)]
        data = pd.DataFrame(
            {
                "Job_ID": [f"JOB_{i:05d}" for i in range(scenario.num_jobs)],
                "Machine_ID": machines,
                "Scheduled_Start": timestamps,
                "Scheduled_End": [ts + timedelta(minutes=int(pt)) for ts, pt in zip(timestamps, processing_time)],
                "Processing_Time": processing_time,
                "Energy_Consumption": energy,
                "Due_Date": due_dates,
                "Priority": priorities,
            }
        )
        return data


class BenchmarkDataGenerator:
    """Access curated benchmark datasets shipped with the repository."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).parent / "benchmarks"

    def available_instances(self) -> List[str]:
        return sorted(path.stem for path in Path(self.root).glob("*.csv"))

    def load_instances(self, names: Iterable[str] | None = None) -> List[pd.DataFrame]:
        if names is None:
            names = self.available_instances()
        frames: List[pd.DataFrame] = []
        for name in names:
            path = Path(name)
            if not path.suffix:
                path = Path(self.root) / f"{name}.csv"
            elif not path.is_absolute():
                path = Path(self.root) / path.name
            if not path.exists():
                raise FileNotFoundError(f"Benchmark dataset '{name}' not found at {path}")
            frame = pd.read_csv(path)
            frame["Source_Benchmark"] = [path.stem] * len(frame)
            frames.append(frame)
        return frames
