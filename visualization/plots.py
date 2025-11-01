"""Plotting utilities for experiments."""
from __future__ import annotations

import math
from itertools import accumulate
from pathlib import Path
from typing import Dict, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    from visualization import simpleplot as plt  # type: ignore[no-redef]
import pandas as pd


def _save_figure(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _group_metric(results: pd.DataFrame, metric: str) -> Dict[str, list[float]]:
    algorithms = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    series = results[metric].astype(float)
    values = series.to_list() if hasattr(series, "to_list") else list(series)
    grouped: Dict[str, list[float]] = {}
    for algorithm, value in zip(algorithms, values):
        grouped.setdefault(str(algorithm), []).append(float(value))
    return grouped


def bar_performance(results: pd.DataFrame, metric: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    values_series = results[metric].astype(float)
    values = values_series.to_list() if hasattr(values_series, "to_list") else list(values_series)
    ax.bar(categories, values)
    ax.set_ylabel(metric)
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Performance comparison on {metric}")
    ax.grid(True, axis="y", alpha=0.3)
    return _save_figure(fig, output)


def box_performance(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Box plot comparing algorithm distributions for a metric."""

    grouped = _group_metric(results, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(list(grouped.values()), labels=list(grouped.keys()), vert=True, patch_artist=True)
    ax.set_title(f"Distribution of {metric}")
    ax.set_ylabel(metric)
    return _save_figure(fig, output)


def violin_performance(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Violin plot for richer distribution insight."""

    grouped = _group_metric(results, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ax.violinplot(list(grouped.values()), showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(grouped) + 1))
    ax.set_xticklabels(list(grouped.keys()))
    ax.set_title(f"Violin comparison on {metric}")
    ax.set_ylabel(metric)
    return _save_figure(fig, output)


def line_convergence(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Plot convergence curves over iterations for each algorithm."""

    fig, ax = plt.subplots(figsize=(6, 4))
    algorithms = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    iterations = results["iteration"].astype(float)
    iteration_values = iterations.to_list() if hasattr(iterations, "to_list") else list(iterations)
    metric_series = results[metric].astype(float)
    metric_values = metric_series.to_list() if hasattr(metric_series, "to_list") else list(metric_series)
    grouped: Dict[str, list[tuple[float, float]]] = {}
    for algo, iteration, value in zip(algorithms, iteration_values, metric_values):
        grouped.setdefault(str(algo), []).append((float(iteration), float(value)))
    for algorithm, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, label=algorithm)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"Convergence trajectories for {metric}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, output)


def scatter_tradeoff(results: pd.DataFrame, metric_x: str, metric_y: str, output: Path) -> Path:
    """Scatter plot showing trade-offs between two metrics."""

    x_series = results[metric_x].astype(float)
    y_series = results[metric_y].astype(float)
    categories = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    color_codes = {name: idx for idx, name in enumerate(sorted({str(name) for name in categories}))}
    colors = [color_codes[str(name)] for name in categories]
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(x_series.to_list(), y_series.to_list(), c=colors, cmap="viridis")
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title(f"Trade-off: {metric_x} vs {metric_y}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Algorithm index")
    return _save_figure(fig, output)


def pareto_front_plot(results: pd.DataFrame, metric_x: str, metric_y: str, output: Path) -> Path:
    """Plot a two-dimensional Pareto frontier."""

    rows = []
    for idx in range(len(results)):
        rows.append(
            {
                metric_x: float(results[metric_x][idx]),
                metric_y: float(results[metric_y][idx]),
            }
        )
    rows.sort(key=lambda row: (row[metric_x], row[metric_y]))
    pareto_x: list[float] = []
    pareto_y: list[float] = []
    best = math.inf
    for row in rows:
        value = row[metric_y]
        if value < best:
            pareto_x.append(row[metric_x])
            pareto_y.append(value)
            best = value
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(results[metric_x].astype(float).to_list(), results[metric_y].astype(float).to_list(), alpha=0.5, label="Solutions")
    ax.plot(pareto_x, pareto_y, color="red", marker="o", label="Pareto front")
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.legend(loc="best")
    ax.set_title("Pareto front")
    return _save_figure(fig, output)


def pareto_front_3d(results: pd.DataFrame, metrics: Sequence[str], output: Path) -> Path:
    """Visualise a three-dimensional Pareto surface."""

    if len(metrics) != 3:
        raise ValueError("Three metrics are required for 3D Pareto plots")
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        results[metrics[0]].astype(float).to_list(),
        results[metrics[1]].astype(float).to_list(),
        results[metrics[2]].astype(float).to_list(),
        c="steelblue",
        alpha=0.7,
    )
    ax.set_xlabel(metrics[0])
    ax.set_ylabel(metrics[1])
    ax.set_zlabel(metrics[2])
    ax.set_title("3D Pareto frontier")
    return _save_figure(fig, output)


def parallel_coordinates_plot(results: pd.DataFrame, metrics: Sequence[str], output: Path) -> Path:
    """Parallel coordinates for multi-objective comparison."""

    spans: Dict[str, tuple[float, float]] = {}
    for metric in metrics:
        series = results[metric].astype(float)
        values = series.to_list()
        min_value = min(values) if values else 0.0
        max_value = max(values) if values else 0.0
        span = max_value - min_value
        spans[metric] = (min_value, span)
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in range(len(results)):
        row_values = []
        for metric in metrics:
            value = float(results[metric][idx])
            min_value, span = spans[metric]
            row_values.append(0.0 if span == 0 else (value - min_value) / span)
        ax.plot(range(len(metrics)), row_values, alpha=0.6)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Normalised value")
    ax.set_title("Parallel coordinates of objectives")
    return _save_figure(fig, output)


def radar_performance_plot(results: pd.DataFrame, metrics: Sequence[str], algorithm: str, output: Path) -> Path:
    """Generate a radar chart for a specific algorithm across metrics."""

    target_index = None
    algorithms = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    for idx, name in enumerate(algorithms):
        if str(name) == algorithm:
            target_index = idx
            break
    if target_index is None:
        raise ValueError(f"Algorithm {algorithm} not found in results")
    values = [float(results[metric][target_index]) for metric in metrics]
    span_values = []
    for metric in metrics:
        series = results[metric].astype(float)
        values_series = series.to_list()
        min_value = min(values_series) if values_series else 0.0
        max_value = max(values_series) if values_series else 0.0
        span = max_value - min_value
        span_values.append(0.0 if span == 0 else (float(results[metric][target_index]) - min_value) / span)
    angles = [n / float(len(metrics)) * 2 * math.pi for n in range(len(metrics))]
    angles += angles[:1]
    span_values += span_values[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, span_values, linewidth=2, label=algorithm)
    ax.fill(angles, span_values, alpha=0.25)
    ax.set_xticks([n / float(len(metrics)) * 2 * math.pi for n in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Radar profile for {algorithm}")
    ax.legend(loc="upper right")
    return _save_figure(fig, output)


def heatmap_correlation(results: pd.DataFrame, metrics: Sequence[str], output: Path) -> Path:
    """Correlation heatmap between metrics."""

    corr_matrix: list[list[float]] = []
    value_cache: Dict[str, list[float]] = {}
    for metric in metrics:
        series = results[metric].astype(float)
        value_cache[metric] = series.to_list()
    for metric_a in metrics:
        row: list[float] = []
        values_a = value_cache[metric_a]
        mean_a = sum(values_a) / len(values_a) if values_a else 0.0
        var_a = sum((value - mean_a) ** 2 for value in values_a) if values_a else 0.0
        for metric_b in metrics:
            values_b = value_cache[metric_b]
            mean_b = sum(values_b) / len(values_b) if values_b else 0.0
            covariance = sum((va - mean_a) * (vb - mean_b) for va, vb in zip(values_a, values_b)) if values_a else 0.0
            var_b = sum((value - mean_b) ** 2 for value in values_b) if values_b else 0.0
            denominator = math.sqrt(var_a * var_b) if var_a and var_b else 1.0
            row.append(covariance / denominator if denominator else 0.0)
        corr_matrix.append(row)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr_matrix[i][j]:.2f}", va="center", ha="center", color="black")
    ax.set_title("Metric correlation heatmap")
    return _save_figure(fig, output)


def heatmap_significance(p_values: pd.DataFrame, output: Path) -> Path:
    """Heatmap showing statistical significance levels."""

    matrix = []
    for i in range(len(p_values.index)):
        row_series = p_values.iloc[i]
        row = [float(row_series[col]) for col in p_values.columns]
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap="viridis_r", vmin=0, vmax=0.1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    ax.set_xticks(range(len(p_values.columns)))
    ax.set_xticklabels(p_values.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(p_values.index)))
    ax.set_yticklabels(p_values.index)
    for i in range(len(p_values.index)):
        for j in range(len(p_values.columns)):
            ax.text(j, i, f"{p_values.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    ax.set_title("Significance matrix")
    return _save_figure(fig, output)


def gantt_chart(schedule: pd.DataFrame, output: Path) -> Path:
    """Generate a Gantt chart from a schedule."""

    fig, ax = plt.subplots(figsize=(8, 4))
    machines_series = schedule.get("Machine_ID", pd.Series(["M0"] * len(schedule)))
    machines = machines_series.to_list() if hasattr(machines_series, "to_list") else list(machines_series)
    unique_machines = list(dict.fromkeys(machines))
    for idx, (_, row) in enumerate(schedule.iterrows()):
        machine = row.get("Machine_ID", "M0")
        start = pd.to_datetime(row.get("Scheduled_Start"))
        end = pd.to_datetime(row.get("Scheduled_End"))
        duration = (end - start).total_seconds() / 3600 if pd.notna(end) and pd.notna(start) else 0
        y = unique_machines.index(machine)
        left = 0.0
        if pd.notna(start):
            midnight = start.normalize()
            left = (start - midnight).total_seconds() / 3600
        ax.barh(y, duration, left=left, height=0.4)
        ax.text(
            (start - start.normalize()).total_seconds() / 3600 if pd.notna(start) else 0,
            y,
            str(row.get("Job_ID", idx)),
            va="center",
            ha="left",
        )
    ax.set_yticks(range(len(unique_machines)))
    ax.set_yticklabels(unique_machines)
    ax.set_xlabel("Hours within day")
    ax.set_title("Schedule Gantt chart")
    return _save_figure(fig, output)


def stacked_area_utilization(timeseries: pd.DataFrame, output: Path) -> Path:
    """Plot stacked area chart for resource utilisation over time."""

    time_series = pd.to_datetime(timeseries["timestamp"])
    base_time = time_series.iloc[0] if len(time_series) else pd.Timestamp("1970-01-01")
    time = [float((timestamp - base_time).total_seconds() / 3600) for timestamp in time_series.to_list()]
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = [col for col in timeseries.columns if col != "timestamp"]
    data_series = []
    for metric in metrics:
        series = timeseries[metric].astype(float)
        data_series.append(series.to_list())
    ax.stackplot(time, data_series, labels=metrics, alpha=0.8)
    ax.legend(loc="upper left")
    ax.set_ylabel("Utilisation")
    ax.set_xlabel("Time")
    ax.set_title("Resource utilisation")
    return _save_figure(fig, output)


def histogram_metric(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Histogram for a performance metric."""

    fig, ax = plt.subplots(figsize=(6, 4))
    series = results[metric].astype(float)
    ax.hist(series.to_list(), bins=20, color="tab:blue", alpha=0.7)
    ax.set_title(f"Histogram of {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    return _save_figure(fig, output)


def density_plot_metric(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Density-style plot using a smooth histogram."""

    fig, ax = plt.subplots(figsize=(6, 4))
    series = results[metric].astype(float)
    ax.hist(series.to_list(), bins=30, density=True, alpha=0.6, color="tab:green")
    ax.set_title(f"Density estimate for {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    return _save_figure(fig, output)


def cdf_metric_plot(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Empirical cumulative distribution function plot."""

    series = results[metric].astype(float)
    values = sorted(series.to_list())
    cumulative = [i / len(values) for i in range(1, len(values) + 1)] if values else []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(values, cumulative, where="post")
    ax.set_xlabel(metric)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF of {metric}")
    return _save_figure(fig, output)


def rug_plot_metric(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Rug plot to visualise value concentration."""

    values_series = results[metric].astype(float)
    values = values_series.to_list()
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.scatter(values, [0] * len(values), marker="|", s=120)
    ax.set_yticks([])
    ax.set_xlabel(metric)
    ax.set_title(f"Rug plot of {metric}")
    return _save_figure(fig, output)


def bubble_chart(results: pd.DataFrame, metric_x: str, metric_y: str, size_metric: str, output: Path) -> Path:
    """Bubble chart for tri-variate comparisons."""

    size_series = results[size_metric].astype(float)
    size_values = size_series.to_list()
    min_size = min(size_values) if size_values else 0.0
    size_scaled = [(value - min_size + 1.0) * 50 for value in size_values]
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(results[metric_x].astype(float).to_list(), results[metric_y].astype(float).to_list(), s=size_scaled, alpha=0.6)
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title(f"Bubble chart with bubble size from {size_metric}")
    fig.colorbar(scatter, ax=ax, label=size_metric)
    return _save_figure(fig, output)


def slope_graph(data: pd.DataFrame, category: str, start: str, end: str, output: Path) -> Path:
    """Slope graph showing changes between two scenarios."""

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in data.iterrows():
        start_value = float(row[start])
        end_value = float(row[end])
        ax.plot([0, 1], [start_value, end_value], marker="o")
        ax.text(-0.02, start_value, str(row[category]), ha="right", va="center")
        ax.text(1.02, end_value, str(row[category]), ha="left", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([start, end])
    ax.set_ylabel("Value")
    ax.set_title("Slope graph comparison")
    return _save_figure(fig, output)


def throughput_timeline(results: pd.DataFrame, time_column: str, count_column: str, output: Path) -> Path:
    """Timeline plot for throughput or completed jobs."""

    fig, ax = plt.subplots(figsize=(6, 4))
    time_series = pd.to_datetime(results[time_column])
    base = time_series.iloc[0] if len(time_series) else pd.Timestamp("1970-01-01")
    time = [float((timestamp - base).total_seconds() / 3600) for timestamp in time_series.to_list()]
    count_series = results[count_column].astype(float)
    ax.step(time, count_series.to_list(), where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel(count_column)
    ax.set_title("Throughput over time")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, output)


def stacked_bar_objectives(results: pd.DataFrame, metrics: Sequence[str], output: Path) -> Path:
    """Stacked bar chart for multiple objectives per algorithm."""

    fig, ax = plt.subplots(figsize=(7, 4))
    algorithms = results["algorithm"].to_list() if hasattr(results["algorithm"], "to_list") else list(results["algorithm"])
    bottom = [0.0] * len(algorithms)
    for metric in metrics:
        series = results[metric].astype(float)
        values = series.to_list()
        ax.bar(algorithms, values, bottom=bottom, label=metric)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("Aggregated value")
    ax.set_title("Stacked objectives per algorithm")
    ax.legend(loc="upper right")
    return _save_figure(fig, output)


def cumulative_improvement(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Plot cumulative improvements across experiments."""

    sorted_values = sorted(results[metric].astype(float).to_list())
    improvements = list(accumulate(sorted_values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(improvements) + 1), improvements, marker="o")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(f"Cumulative {metric}")
    ax.set_title("Cumulative improvements")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, output)


def boxen_schedule_variability(results: pd.DataFrame, metric: str, output: Path) -> Path:
    """Boxen-style layered box plot to emphasise variability."""

    grouped = _group_metric(results, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(list(grouped.values()), labels=list(grouped.keys()), showfliers=False)
    ax.set_title(f"Boxen approximation for {metric}")
    ax.set_ylabel(metric)
    return _save_figure(fig, output)


def waterfall_breakdown(components: pd.DataFrame, output: Path) -> Path:
    """Waterfall chart illustrating contribution of components."""

    fig, ax = plt.subplots(figsize=(7, 4))
    indices = []
    values = []
    colors = []
    for _, row in components.iterrows():
        indices.append(row["component"])
        values.append(row["value"])
        colors.append("tab:green" if row["value"] >= 0 else "tab:red")
    totals = list(accumulate(values))
    starts = [0.0] + totals[:-1]
    for idx, (start, value, label, color) in enumerate(zip(starts, values, indices, colors)):
        ax.bar(idx, value, bottom=start, color=color)
        ax.text(idx, start + value / 2, f"{value:.2f}", ha="center", va="center", color="white")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices, rotation=45, ha="right")
    ax.set_ylabel("Contribution")
    ax.set_title("Waterfall breakdown")
    return _save_figure(fig, output)
