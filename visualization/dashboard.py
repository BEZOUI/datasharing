"""Interactive dashboard for real-time optimisation monitoring."""
from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Iterable, List, Optional

import pandas as pd

from algorithms import get_algorithm, list_algorithms
from core.metrics import evaluate_schedule
from core.solution import ScheduleSolution
from data.generator import BenchmarkDataGenerator
from data.loader import DataLoader, DataPreprocessor
from problems import get_problem_factory, list_problem_types
from visualization.gallery import generate_gallery

try:  # pragma: no cover - optional dependency for GUI environments
    import tkinter as tk
    from tkinter import filedialog, ttk
except ModuleNotFoundError:  # pragma: no cover - guard for headless systems
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]


def tkinter_available() -> bool:
    """Return *True* when Tkinter can be used in the current environment."""

    return tk is not None


class RMSDashboard:
    """Interactive control centre for benchmarking optimisation algorithms."""

    def __init__(self, root: tk.Tk) -> None:  # type: ignore[type-arg]
        if tk is None:  # pragma: no cover - defensive programming
            raise RuntimeError("Tkinter is not available in this environment")

        self.root = root
        self.root.title("RMS Optimisation Control Centre")
        self.root.geometry("1400x780")

        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.benchmark_loader = BenchmarkDataGenerator()
        self.loaded_data: Optional[pd.DataFrame] = None
        self.latest_schedule: Optional[pd.DataFrame] = None
        self.latest_results: Optional[pd.DataFrame] = None

        self._ui_queue: "Queue[Callable[[], None]]" = Queue()
        self._build_layout()
        self._poll_queue()

    # ------------------------------------------------------------------ UI --
    def _build_layout(self) -> None:
        if tk is None or ttk is None:  # pragma: no cover
            return

        container = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(container, padding=10)
        display_frame = ttk.Frame(container, padding=10)
        container.add(control_frame, weight=1)
        container.add(display_frame, weight=2)

        # Data selection controls
        data_section = ttk.LabelFrame(control_frame, text="Dataset", padding=10)
        data_section.pack(fill=tk.X, expand=False)

        self.dataset_path_var = tk.StringVar()
        path_entry = ttk.Entry(data_section, textvariable=self.dataset_path_var, width=48)
        path_entry.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=5)

        browse_button = ttk.Button(data_section, text="Browse", command=self._browse_dataset)
        browse_button.grid(row=0, column=1, padx=5, pady=5)

        benchmark_names = self.benchmark_loader.available_instances()
        ttk.Label(data_section, text="Benchmark").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.benchmark_var = tk.StringVar()
        benchmark_combo = ttk.Combobox(data_section, textvariable=self.benchmark_var, values=benchmark_names, state="readonly")
        benchmark_combo.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        benchmark_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_benchmark())

        load_button = ttk.Button(data_section, text="Load Dataset", command=self._load_dataset)
        load_button.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        # Problem selection
        problem_section = ttk.LabelFrame(control_frame, text="Problem configuration", padding=10)
        problem_section.pack(fill=tk.X, expand=False, pady=(10, 0))
        ttk.Label(problem_section, text="Problem type").grid(row=0, column=0, sticky=tk.W)
        self.problem_var = tk.StringVar(value=list_problem_types()[0])
        problem_combo = ttk.Combobox(
            problem_section,
            textvariable=self.problem_var,
            values=list_problem_types(),
            state="readonly",
        )
        problem_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)

        self.run_all_problems = tk.BooleanVar(value=False)
        ttk.Checkbutton(problem_section, text="Run all problem types", variable=self.run_all_problems).grid(
            row=1, column=0, columnspan=2, sticky=tk.W
        )

        # Algorithm selection
        algorithm_section = ttk.LabelFrame(control_frame, text="Algorithms", padding=10)
        algorithm_section.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(algorithm_section, text="Select algorithms to execute").pack(anchor=tk.W)
        self.algorithm_listbox = tk.Listbox(algorithm_section, selectmode=tk.MULTIPLE, exportselection=False, height=12)
        for name in list_algorithms():
            self.algorithm_listbox.insert(tk.END, name)
        self.algorithm_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        select_all_button = ttk.Button(algorithm_section, text="Select All", command=self._select_all_algorithms)
        select_all_button.pack(fill=tk.X, padx=5, pady=2)

        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
        ttk.Button(action_frame, text="Run Optimisation", command=self._run_async).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Generate Figure Suite", command=self._generate_figures_async).pack(fill=tk.X, padx=5, pady=2)

        # Display section
        log_section = ttk.LabelFrame(display_frame, text="Experiment log", padding=10)
        log_section.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_section, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        gantt_section = ttk.LabelFrame(display_frame, text="Gantt visualisation", padding=10)
        gantt_section.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.gantt_canvas = tk.Canvas(gantt_section, background="#1f2933", height=280)
        self.gantt_canvas.pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------- helpers --
    def _poll_queue(self) -> None:
        if tk is None:  # pragma: no cover
            return
        try:
            while True:
                callback = self._ui_queue.get_nowait()
                callback()
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _enqueue(self, callback: Callable[[], None]) -> None:
        self._ui_queue.put(callback)

    def _append_log(self, message: str) -> None:
        def _write() -> None:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)

        self._enqueue(_write)

    def _browse_dataset(self) -> None:
        if filedialog is None:  # pragma: no cover
            return
        filename = filedialog.askopenfilename(filetypes=(("CSV", "*.csv"), ("JSON", "*.json")))
        if filename:
            self.dataset_path_var.set(filename)

    def _load_benchmark(self) -> None:
        selection = self.benchmark_var.get()
        if not selection:
            return
        try:
            frame = self.benchmark_loader.load_instances([selection])[0]
        except FileNotFoundError as exc:
            self._append_log(str(exc))
            return
        self.loaded_data = frame
        self.dataset_path_var.set(str(Path(self.benchmark_loader.root) / f"{selection}.csv"))
        self._append_log(f"Loaded benchmark dataset '{selection}' with {len(frame)} records")

    def _load_dataset(self) -> None:
        path = self.dataset_path_var.get()
        if not path:
            self._append_log("No dataset path provided")
            return
        try:
            frame = self.loader.load([Path(path)])
        except Exception as exc:  # pragma: no cover - error surfaces through log
            self._append_log(f"Failed to load dataset: {exc}")
            return
        self.loaded_data = self.preprocessor.transform(frame)
        self._append_log(f"Dataset loaded successfully ({len(self.loaded_data)} rows)")

    def _select_all_algorithms(self) -> None:
        if tk is None:  # pragma: no cover
            return
        self.algorithm_listbox.select_set(0, tk.END)

    def _run_async(self) -> None:
        worker = threading.Thread(target=self._run_experiments, daemon=True)
        worker.start()

    def _generate_figures_async(self) -> None:
        worker = threading.Thread(target=self._generate_figures, daemon=True)
        worker.start()

    # ------------------------------------------------------------- execution --
    def _selected_algorithms(self) -> List[str]:
        if tk is None:  # pragma: no cover
            return []
        selection = self.algorithm_listbox.curselection()
        if not selection:
            return list_algorithms()
        return [self.algorithm_listbox.get(index) for index in selection]

    def _problem_names(self) -> Iterable[str]:
        if self.run_all_problems.get():
            return list_problem_types()
        return [self.problem_var.get()]

    def _run_experiments(self) -> None:
        if self.loaded_data is None:
            self._append_log("Please load a dataset before running experiments")
            return

        algorithms = self._selected_algorithms()
        self._append_log(f"Launching optimisation with algorithms: {', '.join(algorithms)}")

        results_records: List[dict] = []
        best_solution: Optional[ScheduleSolution] = None
        best_makespan = float("inf")

        for problem_name in self._problem_names():
            try:
                factory = get_problem_factory(problem_name)
            except KeyError as exc:
                self._append_log(str(exc))
                continue
            problem = factory(self.loaded_data)
            self._append_log(f"Evaluating problem '{problem_name}' with {len(problem.jobs)} operations")
            for algorithm_name in algorithms:
                try:
                    optimizer = get_algorithm(algorithm_name)
                    solution = optimizer.solve(problem)
                    metrics = evaluate_schedule(solution.schedule)
                    record = {"algorithm": algorithm_name, "problem": problem_name, **metrics}
                    results_records.append(record)
                    self._append_log(
                        f"{algorithm_name} | makespan={metrics.get('makespan', 0):.2f} | tardiness={metrics.get('total_tardiness', 0):.2f}"
                    )
                    if metrics.get("makespan", float("inf")) < best_makespan:
                        best_makespan = metrics.get("makespan", float("inf"))
                        best_solution = solution
                except Exception as exc:
                    self._append_log(f"Algorithm '{algorithm_name}' failed: {exc}")

        if not results_records:
            self._append_log("No successful runs were recorded")
            return

        results = pd.DataFrame(results_records)
        self.latest_results = results
        if best_solution is not None:
            self.latest_schedule = best_solution.schedule
            self._enqueue(lambda: self._draw_gantt(best_solution.schedule))
        self._append_log("Optimisation run completed")

    def _generate_figures(self) -> None:
        if self.latest_results is None or self.latest_schedule is None:
            self._append_log("Run an optimisation first to produce figures")
            return
        output_dir = Path("results") / "dashboard_gallery"
        try:
            figures = generate_gallery(self.latest_results, self.latest_schedule, output_dir)
        except Exception as exc:
            self._append_log(f"Failed to generate figure suite: {exc}")
            return
        self._append_log(f"Generated {len(figures)} figures in {output_dir}")

    # --------------------------------------------------------------- drawing --
    def _draw_gantt(self, schedule: pd.DataFrame) -> None:
        if tk is None:  # pragma: no cover
            return
        self.gantt_canvas.delete("all")
        if schedule.empty:
            return
        start_times = pd.to_datetime(schedule["Scheduled_Start"]).fillna(method="ffill").fillna(method="bfill")
        end_times = pd.to_datetime(schedule["Scheduled_End"]).fillna(method="ffill").fillna(method="bfill")
        min_start = start_times.min()
        max_end = end_times.max()
        total_seconds = max((max_end - min_start).total_seconds(), 1.0)

        machines = schedule.get("Machine_ID", pd.Series(["M0"] * len(schedule)))
        unique_machines = list(dict.fromkeys(machines.astype(str)))
        height_per_machine = max(self.gantt_canvas.winfo_height() // max(len(unique_machines), 1), 40)
        canvas_width = max(self.gantt_canvas.winfo_width(), 600)

        for _, row in schedule.iterrows():
            job = str(row.get("Job_ID", "JOB"))
            machine = str(row.get("Machine_ID", "M0"))
            start = pd.to_datetime(row.get("Scheduled_Start", min_start))
            end = pd.to_datetime(row.get("Scheduled_End", start))
            offset = (start - min_start).total_seconds() / total_seconds * canvas_width
            duration = max((end - start).total_seconds(), 60.0) / total_seconds * canvas_width
            y_index = unique_machines.index(machine)
            top = y_index * height_per_machine + 10
            bottom = top + height_per_machine - 20
            self.gantt_canvas.create_rectangle(offset, top, offset + duration, bottom, fill="#38bdf8", outline="#0f172a")
            self.gantt_canvas.create_text(offset + 5, (top + bottom) / 2, anchor="w", text=job, fill="#0f172a")
        for idx, machine in enumerate(unique_machines):
            y = idx * height_per_machine + 5
            self.gantt_canvas.create_text(5, y, anchor="nw", text=machine, fill="#f8fafc")


def launch_dashboard() -> None:
    if tk is None:
        raise RuntimeError("Tkinter is not available; install tkinter to use the dashboard")
    root = tk.Tk()
    RMSDashboard(root)
    root.mainloop()


__all__ = ["RMSDashboard", "launch_dashboard", "tkinter_available"]
