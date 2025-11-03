from __future__ import annotations

from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from visualization import plots


def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "algorithm": ["A", "B", "C"],
            "makespan": [120.0, 110.0, 130.0],
            "energy": [50.0, 55.0, 45.0],
            "total_tardiness": [12.0, 9.0, 15.0],
        }
    )


def sample_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="H"),
            "machine_a": [0.5, 0.6, 0.7, 0.5, 0.4],
            "machine_b": [0.4, 0.5, 0.6, 0.4, 0.3],
        }
    )


def sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Job_ID": ["J1", "J2"],
            "Machine_ID": ["M1", "M2"],
            "Scheduled_Start": ["2023-01-01T08:00:00", "2023-01-01T09:00:00"],
            "Scheduled_End": ["2023-01-01T09:00:00", "2023-01-01T10:00:00"],
        }
    )


def test_generate_multiple_plots(tmp_path: Path) -> None:
    results = sample_results()
    plots.bar_performance(results, "makespan", tmp_path / "bar.png")
    plots.box_performance(results, "makespan", tmp_path / "box.png")
    plots.violin_performance(results, "makespan", tmp_path / "violin.png")
    plots.pareto_front_plot(results, "makespan", "energy", tmp_path / "pareto.png")
    plots.parallel_coordinates_plot(results, ["makespan", "energy", "total_tardiness"], tmp_path / "parallel.png")
    plots.radar_performance_plot(results, ["makespan", "energy"], "A", tmp_path / "radar.png")
    plots.heatmap_correlation(results, ["makespan", "energy", "total_tardiness"], tmp_path / "corr.png")
    plots.histogram_metric(results, "makespan", tmp_path / "hist.png")
    plots.cdf_metric_plot(results, "makespan", tmp_path / "cdf.png")
    plots.stacked_bar_objectives(results, ["makespan", "energy"], tmp_path / "stacked.png")
    assert (tmp_path / "bar.png").exists()


def test_schedule_and_timeseries_plots(tmp_path: Path) -> None:
    schedule = sample_schedule()
    plots.gantt_chart(schedule, tmp_path / "gantt.png")
    timeseries = sample_timeseries()
    plots.stacked_area_utilization(timeseries, tmp_path / "util.png")
    plots.throughput_timeline(
        pd.DataFrame({"time": pd.date_range("2023-01-01", periods=4, freq="H"), "jobs": [0, 2, 4, 6]}),
        "time",
        "jobs",
        tmp_path / "throughput.png",
    )
    assert (tmp_path / "gantt.png").exists()
