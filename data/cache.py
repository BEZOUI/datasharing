"""Simple caching utilities for large datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

try:
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore
    import pickle


class DataCache:
    """Persist dataframes using joblib for quick reloads."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_compute(self, name: str, factory: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.pkl"
        if path.exists():
            if joblib is not None:
                return joblib.load(path)
            with path.open('rb') as handle:
                return pickle.load(handle)
        dataframe = factory()
        if joblib is not None:
            joblib.dump(dataframe, path)
        else:
            with path.open('wb') as handle:
                pickle.dump(dataframe, handle)
        return dataframe
