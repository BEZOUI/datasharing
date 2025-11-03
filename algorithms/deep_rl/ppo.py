"""Lightweight proximal policy optimisation for scheduling."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd

from algorithms.metaheuristics.utils import merge_objective_weights, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _job_features(job: pd.Series) -> List[float]:
    processing = float(job.get("Processing_Time", 0.0) or 0.0)
    energy = float(job.get("Energy_Consumption", 0.0) or 0.0)
    due_date = job.get("Due_Date")
    start = job.get("Release_Date", job.get("Scheduled_Start"))
    slack = 0.0
    if pd.notna(due_date) and pd.notna(start):
        due_ts = pd.to_datetime(due_date)
        start_ts = pd.to_datetime(start)
        slack = float((due_ts - start_ts).total_seconds() / 60.0)
    return [processing / 120.0, energy / 50.0, slack / 120.0, 1.0]


def _softmax(scores: Sequence[float]) -> List[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exp_scores]


@dataclass
class Step:
    features: List[List[float]]
    selected: int
    old_prob: float


class PPOOptimizer(BaseOptimizer):
    """Implements a compact PPO variant with linear policy."""

    def __init__(
        self,
        episodes: int = 80,
        learning_rate: float = 0.05,
        clip_ratio: float = 0.2,
        seed: int = 23,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            episodes=episodes,
            learning_rate=learning_rate,
            clip_ratio=clip_ratio,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _policy_scores(self, weights: List[float], feature_sets: List[List[float]]) -> List[float]:
        return [sum(w * f for w, f in zip(weights, features)) for features in feature_sets]

    def _policy_gradient(
        self,
        weights: List[float],
        step: Step,
        advantage: float,
    ) -> List[float]:
        scores = self._policy_scores(weights, step.features)
        probs = _softmax(scores)
        selected_prob = probs[step.selected]
        baseline = [0.0 for _ in weights]
        for prob, features in zip(probs, step.features):
            for idx, feature in enumerate(features):
                baseline[idx] += prob * feature
        gradient = [step.features[step.selected][idx] - baseline[idx] for idx in range(len(weights))]
        ratio = selected_prob / max(step.old_prob, 1e-8)
        clipped_ratio = max(min(ratio, 1.0 + self.clip_ratio), 1.0 - self.clip_ratio)
        scale = clipped_ratio * advantage
        return [g * scale for g in gradient]

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        feature_dim = len(_job_features(problem.jobs.iloc[0]))
        weights = [rng.uniform(-0.5, 0.5) for _ in range(feature_dim)]
        rewards: List[float] = []

        for _ in range(self.episodes):
            available = list(problem.jobs.index)
            step_records: List[Step] = []
            sequence: List[int] = []
            while available:
                feature_sets = [_job_features(problem.jobs.loc[job]) for job in available]
                scores = self._policy_scores(weights, feature_sets)
                probs = _softmax(scores)
                threshold = rng.random()
                cumulative = 0.0
                selected_idx = 0
                for idx, prob in enumerate(probs):
                    cumulative += prob
                    if cumulative >= threshold:
                        selected_idx = idx
                        break
                selected_job = available.pop(selected_idx)
                sequence.append(selected_job)
                step_records.append(Step(features=feature_sets, selected=selected_idx, old_prob=probs[selected_idx]))
            value, metrics = sequence_objective(problem, sequence, self.objective_weights)
            reward = -value
            rewards.append(reward)

            baseline = sum(rewards) / len(rewards)
            for step_record in step_records:
                advantage = reward - baseline
                gradient = self._policy_gradient(weights, step_record, advantage)
                for idx, grad in enumerate(gradient):
                    weights[idx] += self.learning_rate * grad

        available = list(problem.jobs.index)
        greedy_sequence: List[int] = []
        while available:
            feature_sets = [_job_features(problem.jobs.loc[job]) for job in available]
            scores = self._policy_scores(weights, feature_sets)
            probs = _softmax(scores)
            selected_idx = max(range(len(available)), key=lambda idx: probs[idx])
            greedy_sequence.append(available.pop(selected_idx))

        final_schedule = problem.build_schedule(greedy_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"sequence": greedy_sequence, "policy_weights": weights},
        )
