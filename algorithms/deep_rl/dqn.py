"""Light-weight Deep-Q-inspired scheduler using linear function approximation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _extract_features(job_row: Dict[str, object]) -> np.ndarray:
    processing = float(job_row.get("Processing_Time", 0.0))
    due_date = job_row.get("Due_Date")
    release = job_row.get("Scheduled_Start") or job_row.get("Release_Date")
    energy = float(job_row.get("Energy_Consumption", 0.0))
    due_minutes = 0.0
    release_minutes = 0.0
    if due_date is not None and not pd.isna(due_date):
        due_minutes = pd.to_datetime(due_date).value / 60_000_000_000
    if release is not None and not pd.isna(release):
        release_minutes = pd.to_datetime(release).value / 60_000_000_000
    slack = due_minutes - release_minutes - processing
    return np.array([processing, slack, energy, 1.0], dtype=float)


@dataclass
class LinearQNetwork:
    weights: np.ndarray
    learning_rate: float

    def predict(self, features: np.ndarray) -> float:
        return float(features @ self.weights)

    def update(self, features: np.ndarray, target: float) -> None:
        prediction = self.predict(features)
        error = target - prediction
        self.weights += self.learning_rate * error * features


class DQNOptimizer(BaseOptimizer):
    """A simplified Deep-Q scheduler relying on linear approximation."""

    def __init__(
        self,
        episodes: int = 200,
        discount: float = 0.9,
        learning_rate: float = 1e-3,
        epsilon: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__(episodes=episodes, discount=discount, learning_rate=learning_rate, epsilon=epsilon, seed=seed)
        self.episodes = episodes
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.seed = seed

    def _train(self, problem: ManufacturingProblem) -> LinearQNetwork:
        rng = np.random.default_rng(self.seed)
        weights = rng.normal(loc=0.0, scale=0.01, size=4)
        network = LinearQNetwork(weights=weights, learning_rate=self.learning_rate)
        job_indices = list(problem.jobs.index)
        if not job_indices:
            return network

        for _ in range(self.episodes):
            remaining = job_indices.copy()
            rng.shuffle(remaining)
            current_time = 0.0
            sequence: List[int] = []
            while remaining:
                state_features = []
                for idx in remaining:
                    features = _extract_features(problem.jobs.loc[idx].to_dict())
                    features = features / (np.linalg.norm(features) + 1e-9)
                    state_features.append((idx, features))
                if rng.random() < self.epsilon:
                    action_idx = rng.choice(len(state_features))
                else:
                    q_values = [network.predict(features) for _, features in state_features]
                    action_idx = int(np.argmin(q_values))
                job_id, features = state_features[action_idx]
                sequence.append(job_id)
                remaining.remove(job_id)

                current_time += float(problem.jobs.loc[job_id].get("Processing_Time", 0.0))
                reward = -current_time
                future_estimate = 0.0
                if remaining:
                    next_features = [_extract_features(problem.jobs.loc[idx].to_dict()) for idx in remaining]
                    next_q = [network.predict(feat / (np.linalg.norm(feat) + 1e-9)) for feat in next_features]
                    future_estimate = min(next_q)
                target = reward + self.discount * future_estimate
                network.update(features, target)

        return network

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        network = self._train(problem)
        jobs = problem.jobs
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        features = []
        for idx, row in jobs.iterrows():
            feat = _extract_features(row.to_dict())
            norm_feat = feat / (np.linalg.norm(feat) + 1e-9)
            features.append((idx, network.predict(norm_feat)))
        features.sort(key=lambda item: item[1])
        sequence = [idx for idx, _ in features]
        schedule = problem.build_schedule(sequence)
        metrics = evaluate_schedule(schedule)
        return ScheduleSolution(schedule=schedule, metrics=metrics, metadata={"policy": "linear_dqn"})
