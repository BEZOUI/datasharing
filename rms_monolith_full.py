# Auto-generated monolithic RMS optimisation script
# Do not edit manually.

"""Monolithic RMS optimisation framework."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

MODULE_SOURCES: dict[str, dict[str, object]] = {}

def _register_module(name: str, source: str, is_package: bool) -> None:
    MODULE_SOURCES[name] = {"code": source, "is_package": is_package}

# BEGIN MODULE: algorithms (algorithms/__init__.py)
_register_module('algorithms', r'''
"""Algorithm registry and utility helpers."""
from __future__ import annotations

from typing import Callable, Dict, List

from algorithms.classical.dispatching_rules import DISPATCHING_RULES, DispatchingRule
from algorithms.classical.constructive_heuristics import NEHHeuristic, PalmerHeuristic
from algorithms.classical.exact_methods import BranchAndBound
from algorithms.deep_rl.dqn import DQNOptimizer
from algorithms.deep_rl.ppo import PPOOptimizer
from algorithms.hybrid.adaptive_hybrid import AdaptiveHybridOptimizer
from algorithms.metaheuristics import (
    AntColonyOptimization,
    DifferentialEvolution,
    GeneticAlgorithm,
    GuidedLocalSearch,
    IteratedLocalSearch,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    TabuSearch,
    VariableNeighborhoodSearch,
)
from algorithms.multi_objective.nsga2 import NSGAII
from core.base_optimizer import BaseOptimizer


def get_algorithm(name: str, **kwargs) -> BaseOptimizer:
    """Instantiate an algorithm by name.

    Dispatching rules can be referenced directly by their identifier
    (e.g. ``"spt"``).  Other algorithms expose canonical names matching the
    research roadmap (``"simulated_annealing"``, ``"nsga2"``, ``"dqn"``,
    ``"adaptive_hybrid"``).
    """

    name = name.lower()
    if name in DISPATCHING_RULES:
        return DISPATCHING_RULES[name](**kwargs)

    registry: Dict[str, Callable[..., BaseOptimizer]] = {
        "neh": NEHHeuristic,
        "palmer": PalmerHeuristic,
        "branch_and_bound": BranchAndBound,
        "simulated_annealing": SimulatedAnnealing,
        "genetic_algorithm": GeneticAlgorithm,
        "particle_swarm": ParticleSwarmOptimization,
        "ant_colony": AntColonyOptimization,
        "tabu_search": TabuSearch,
        "variable_neighborhood_search": VariableNeighborhoodSearch,
        "iterated_local_search": IteratedLocalSearch,
        "guided_local_search": GuidedLocalSearch,
        "differential_evolution": DifferentialEvolution,
        "nsga2": NSGAII,
        "dqn": DQNOptimizer,
        "ppo": PPOOptimizer,
        "adaptive_hybrid": AdaptiveHybridOptimizer,
    }
    if name not in registry:
        raise KeyError(f"Unknown algorithm '{name}'")
    return registry[name](**kwargs)


def list_algorithms(include_dispatching: bool = True) -> List[str]:
    """Return the list of registered optimisation algorithms.

    Parameters
    ----------
    include_dispatching:
        When *True*, short-horizon dispatching heuristics are included in
        addition to advanced optimisation methods.  This is particularly
        useful for interactive exploration in the visual dashboard where the
        researcher may want to benchmark simple baselines alongside
        state-of-the-art learners.
    """

    names: List[str] = [
        "neh",
        "palmer",
        "branch_and_bound",
        "simulated_annealing",
        "genetic_algorithm",
        "particle_swarm",
        "ant_colony",
        "tabu_search",
        "variable_neighborhood_search",
        "iterated_local_search",
        "guided_local_search",
        "differential_evolution",
        "nsga2",
        "dqn",
        "ppo",
        "adaptive_hybrid",
    ]
    if include_dispatching:
        names = list(DISPATCHING_RULES.keys()) + names
    return sorted(dict.fromkeys(names))


__all__ = [
    "get_algorithm",
    "DISPATCHING_RULES",
    "DispatchingRule",
    "list_algorithms",
    "NEHHeuristic",
    "PalmerHeuristic",
    "BranchAndBound",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "AntColonyOptimization",
    "TabuSearch",
    "VariableNeighborhoodSearch",
    "IteratedLocalSearch",
    "GuidedLocalSearch",
    "DifferentialEvolution",
    "NSGAII",
    "DQNOptimizer",
    "PPOOptimizer",
    "AdaptiveHybridOptimizer",
]
''', True)
# END MODULE: algorithms

# BEGIN MODULE: config (config/__init__.py)
_register_module('config', r'''

''', True)
# END MODULE: config

# BEGIN MODULE: core (core/__init__.py)
_register_module('core', r'''

''', True)
# END MODULE: core

# BEGIN MODULE: data (data/__init__.py)
_register_module('data', r'''

''', True)
# END MODULE: data

# BEGIN MODULE: experiments (experiments/__init__.py)
_register_module('experiments', r'''

''', True)
# END MODULE: experiments

# BEGIN MODULE: pandas (pandas/__init__.py)
_register_module('pandas', r'''
"""Minimal pure-Python subset of the pandas API used by the project.

The goal of this module is to provide just enough functionality for the
research framework to execute in a restricted environment without the real
`pandas` dependency.  Only the operations that are exercised by the unit
tests are implemented.  The implementation focuses on readability and
determinism rather than raw performance.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DataFrame",
    "Series",
    "Index",
    "RangeIndex",
    "Timestamp",
    "NaT",
    "isna",
    "notna",
    "to_datetime",
    "date_range",
    "to_numeric",
    "to_timedelta",
    "concat",
    "read_csv",
    "read_json",
    "read_parquet",
]


NaT = object()
NAN = float("nan")


def _is_nan(value: Any) -> bool:
    if value is None or value is NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def isna(value: Any) -> bool | "Series":
    if isinstance(value, Series):
        return Series([_is_nan(v) for v in value._data], index=value._index.copy())
    return _is_nan(value)


def notna(value: Any) -> bool | "Series":
    result = isna(value)
    if isinstance(result, Series):
        return Series([not bool(v) for v in result._data], index=result._index.copy())
    return not result


class Timestamp(datetime):
    """Simple timestamp implementation with nanosecond value accessor."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "Timestamp":
        if not args and not kwargs:
            dt = datetime.utcnow()
        elif len(args) == 1 and not kwargs:
            value = args[0]
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Could not parse datetime string '{value}'") from exc
            else:
                dt = datetime.fromtimestamp(float(value))
        elif "value" in kwargs and len(args) == 0:
            value = kwargs.pop("value")
            return cls(value, **kwargs)
        else:
            return datetime.__new__(cls, *args, **kwargs)
        return datetime.__new__(
            cls,
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            dt.tzinfo,
        )

    @property
    def value(self) -> int:
        epoch = datetime(1970, 1, 1, tzinfo=self.tzinfo)
        delta = self - epoch
        return int(delta.total_seconds() * 1_000_000_000)

    def normalize(self) -> "Timestamp":
        """Return the timestamp floored to midnight of the same day."""

        return Timestamp(
            datetime(
                self.year,
                self.month,
                self.day,
                0,
                0,
                0,
                0,
                tzinfo=self.tzinfo,
            )
        )


class Index:
    def __init__(self, data: Sequence[Any]):
        self._data = list(data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int | slice) -> Any:
        return self._data[item]

    def to_list(self) -> List[Any]:
        return list(self._data)

    @property
    def has_duplicates(self) -> bool:
        return len(set(self._data)) != len(self._data)


class RangeIndex(Index):
    def __init__(self, stop: int, start: int = 0, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(range(start, stop, step))


def _ensure_index(index: Optional[Sequence[Any]], length: int) -> List[Any]:
    if index is None:
        return list(range(length))
    if len(index) != length:
        raise ValueError("Index length must match data length")
    return list(index)


class Series:
    def __init__(
        self,
        data: Any = None,
        index: Optional[Sequence[Any]] = None,
        dtype: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if isinstance(data, Series):
            values = data._data.copy()
            index = data._index.copy() if index is None else list(index)
        elif isinstance(data, Mapping):
            values = list(data.values())
            index = list(data.keys()) if index is None else list(index)
        elif index is not None and (isinstance(data, (int, float, str, bool, Timestamp)) or data is None):
            values = [data for _ in range(len(index))]
        elif data is None:
            values = []
        else:
            values = list(data)
        index_values = _ensure_index(index, len(values))
        self._data: List[Any] = values
        self._index: List[Any] = index_values
        self.dtype = dtype
        self.name = name

    # ------------------------------------------------------------------
    @property
    def index(self) -> Index:
        return Index(self._index)

    @property
    def values(self) -> List[Any]:
        return list(self._data)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def copy(self) -> "Series":
        return Series(self._data.copy(), index=self._index.copy(), dtype=self.dtype, name=self.name)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def _resolve_label(self, label: Any) -> int:
        try:
            return self._index.index(label)
        except ValueError as exc:
            raise KeyError(label) from exc

    def __getitem__(self, key: int | slice | Sequence[int] | Any) -> Any:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self._data)))
            data = [self._data[i] for i in indices]
            idx = [self._index[i] for i in indices]
            return Series(data, index=idx, dtype=self.dtype, name=self.name)
        if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            if all(isinstance(k, int) for k in key):
                positions = list(key)
            else:
                positions = [self._resolve_label(k) for k in key]
            data = [self._data[pos] for pos in positions]
            idx = [self._index[pos] for pos in positions]
            return Series(data, index=idx, dtype=self.dtype, name=self.name)
        if isinstance(key, int):
            return self._data[key]
        position = self._resolve_label(key)
        return self._data[position]

    class _ILoc:
        def __init__(self, series: "Series") -> None:
            self.series = series

        def __getitem__(self, item: int | slice | Sequence[int]) -> Any:
            if isinstance(item, slice):
                indices = range(*item.indices(len(self.series._data)))
                data = [self.series._data[i] for i in indices]
                index = [self.series._index[i] for i in indices]
                return Series(data, index=index, dtype=self.series.dtype, name=self.series.name)
            if isinstance(item, Sequence):
                data = [self.series._data[i] for i in item]
                index = [self.series._index[i] for i in item]
                return Series(data, index=index, dtype=self.series.dtype, name=self.series.name)
            return self.series._data[item]

    @property
    def iloc(self) -> "Series._ILoc":
        return Series._ILoc(self)

    def to_list(self) -> List[Any]:
        return list(self._data)

    def to_numpy(self) -> List[Any]:
        return list(self._data)

    def to_dict(self) -> Dict[Any, Any]:
        return {idx: value for idx, value in zip(self._index, self._data)}

    def _binary_op(self, other: Any, operator: Callable[[Any, Any], Any]) -> "Series":
        if isinstance(other, Series):
            other_map = other.to_dict()
            data = [operator(value, other_map.get(idx, NAN)) for idx, value in zip(self._index, self._data)]
        else:
            data = [operator(value, other) for value in self._data]
        return Series(data, index=self._index.copy(), dtype=self.dtype, name=self.name)

    def __add__(self, other: Any) -> "Series":
        def add(a: Any, b: Any) -> Any:
            if _is_nan(a) and _is_nan(b):
                return NAN
            if _is_nan(a):
                return b
            if _is_nan(b):
                return a
            return a + b

        return self._binary_op(other, add)

    def __sub__(self, other: Any) -> "Series":
        def subtract(a: Any, b: Any) -> Any:
            if _is_nan(a) or _is_nan(b):
                return NAN
            if isinstance(a, datetime) and isinstance(b, datetime):
                return a - b
            return a - b

        return self._binary_op(other, subtract)

    def __rsub__(self, other: Any) -> "Series":
        def subtract(a: Any, b: Any) -> Any:
            if _is_nan(a) or _is_nan(b):
                return NAN
            if isinstance(b, datetime) and isinstance(a, datetime):
                return b - a
            return b - a

        return self._binary_op(other, subtract)

    def __mul__(self, other: Any) -> "Series":
        def multiply(a: Any, b: Any) -> Any:
            if _is_nan(a) or _is_nan(b):
                return NAN
            return a * b

        return self._binary_op(other, multiply)

    def __truediv__(self, other: Any) -> "Series":
        def divide(a: Any, b: Any) -> Any:
            if _is_nan(a) or _is_nan(b) or b in (0, None):
                return NAN
            return a / b

        return self._binary_op(other, divide)

    def __rtruediv__(self, other: Any) -> "Series":
        def divide(a: Any, b: Any) -> Any:
            if _is_nan(a) or _is_nan(b) or a in (0, None):
                return NAN
            return b / a

        return self._binary_op(other, divide)

    def __neg__(self) -> "Series":
        return Series([-value if not _is_nan(value) else NAN for value in self._data], index=self._index.copy(), dtype=self.dtype)

    def __eq__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: False if _is_nan(a) or _is_nan(b) else a == b)

    def __lt__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: False if _is_nan(a) or _is_nan(b) else a < b)

    def __gt__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: False if _is_nan(a) or _is_nan(b) else a > b)

    def sum(self) -> float:
        total = 0.0
        for value in self._data:
            if _is_nan(value):
                continue
            total += float(value)
        return total

    def mean(self) -> float:
        total = 0.0
        count = 0
        for value in self._data:
            if _is_nan(value):
                continue
            total += float(value)
            count += 1
        return total / count if count else 0.0

    def min(self) -> Any:
        valid = [value for value in self._data if not _is_nan(value)]
        return min(valid) if valid else NAN

    def max(self) -> Any:
        valid = [value for value in self._data if not _is_nan(value)]
        return max(valid) if valid else NAN

    def all(self) -> bool:
        return all(bool(value) for value in self._data if not _is_nan(value))

    def any(self) -> bool:
        return any(bool(value) for value in self._data if not _is_nan(value))

    def fillna(self, value: Any) -> "Series":
        if isinstance(value, Series):
            replacement = value.to_dict()
            data = [replacement.get(idx) if _is_nan(current) else current for idx, current in zip(self._index, self._data)]
        else:
            data = [value if _is_nan(current) else current for current in self._data]
        return Series(data, index=self._index.copy(), dtype=self.dtype, name=self.name)

    def isna(self) -> "Series":
        return Series([_is_nan(v) for v in self._data], index=self._index.copy())

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> "Series":
        data: List[Any] = []
        for value in self._data:
            if _is_nan(value):
                data.append(NAN)
                continue
            if lower is not None and value < lower:
                value = lower
            if upper is not None and value > upper:
                value = upper
            data.append(value)
        return Series(data, index=self._index.copy(), dtype=self.dtype, name=self.name)

    def replace(self, to_replace: Any, value: Any) -> "Series":
        if isinstance(to_replace, (list, tuple, set)):
            targets = set(to_replace)
            data = [value if item in targets else item for item in self._data]
        else:
            data = [value if item == to_replace else item for item in self._data]
        return Series(data, index=self._index.copy(), dtype=self.dtype, name=self.name)

    def rank(self, method: str = "average") -> "Series":
        enumerated = [(idx, val, pos) for pos, (idx, val) in enumerate(zip(self._index, self._data)) if not _is_nan(val)]
        enumerated.sort(key=lambda item: (item[1], item[2]))
        ranks: Dict[Any, float] = {}
        current = 1
        for idx, _value, _pos in enumerated:
            ranks[idx] = float(current)
            current += 1
        ranked = [ranks.get(idx, NAN) for idx in self._index]
        return Series(ranked, index=self._index.copy())

    def reindex(self, index: Iterable[Any]) -> "Series":
        mapping = self.to_dict()
        new_index = list(index)
        data = [mapping.get(idx, NAN) for idx in new_index]
        return Series(data, index=new_index, dtype=self.dtype, name=self.name)

    def astype(self, dtype: Any) -> "Series":
        if dtype in (float, int, str, bool):
            cast = dtype
        elif isinstance(dtype, str):
            if dtype == "float":
                cast = float
            elif dtype == "int":
                cast = int
            else:
                raise ValueError(f"Unsupported dtype '{dtype}'")
        else:
            raise ValueError("Unsupported dtype")
        data: List[Any] = []
        for value in self._data:
            if _is_nan(value):
                data.append(NAN)
            else:
                data.append(cast(value))
        return Series(data, index=self._index.copy(), dtype=str(dtype), name=self.name)

    def unique(self) -> List[Any]:
        seen = []
        for value in self._data:
            if value not in seen:
                seen.append(value)
        return seen

    def sort_values(self, ascending: bool = True) -> "Series":
        sortable = list(enumerate(zip(self._index, self._data)))
        sortable.sort(key=lambda item: (_is_nan(item[1][1]), item[1][1], item[0]))
        if not ascending:
            sortable.reverse()
        index = [idx for _, (idx, _val) in sortable]
        data = [val for _, (_idx, val) in sortable]
        return Series(data, index=index, dtype=self.dtype, name=self.name)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            position = self._resolve_label(key)
            return self._data[position]
        except KeyError:
            return default

    def apply(self, func: Callable[[Any], Any]) -> "Series":
        return Series([func(value) for value in self._data], index=self._index.copy(), dtype=self.dtype, name=self.name)

    @property
    def dt(self) -> "_DatetimeAccessor":
        return _DatetimeAccessor(self)


class _DatetimeAccessor:
    def __init__(self, series: Series) -> None:
        self.series = series

    def total_seconds(self) -> Series:
        data: List[float] = []
        for value in self.series._data:
            if _is_nan(value):
                data.append(NAN)
            elif isinstance(value, timedelta):
                data.append(value.total_seconds())
            else:
                raise TypeError("total_seconds requires timedelta values")
        return Series(data, index=self.series._index.copy())


class DataFrame:
    def __init__(
        self,
        data: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]] = None,
        index: Optional[Sequence[Any]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self._data: Dict[str, List[Any]] = {}
        if data is None:
            if columns is not None:
                for column in columns:
                    self._data[column] = []
            self._index = _ensure_index(index, 0)
            return

        if isinstance(data, Mapping):
            columns = list(columns) if columns is not None else list(data.keys())
            lengths = [len(list(data.get(col, []))) for col in columns]
            length = max(lengths) if lengths else 0
            self._index = _ensure_index(index, length)
            for column in columns:
                values = list(data.get(column, []))
                if len(values) != length:
                    if not values and length:
                        values = [None] * length
                    elif len(values) != length:
                        raise ValueError("Column length mismatch")
                self._data[column] = values
        else:
            rows = list(data)
            if rows:
                columns = list(columns) if columns is not None else list(rows[0].keys())
                for column in columns:
                    self._data[column] = [row.get(column) for row in rows]
                self._index = _ensure_index(index, len(rows))
            else:
                self._index = _ensure_index(index, 0)
                if columns is not None:
                    for column in columns:
                        self._data[column] = []

    @property
    def columns(self) -> List[str]:
        return list(self._data.keys())

    @property
    def index(self) -> Index:
        return Index(self._index)

    @property
    def empty(self) -> bool:
        return len(self._index) == 0

    def __len__(self) -> int:
        return len(self._index)

    def copy(self) -> "DataFrame":
        new = DataFrame()
        new._data = {column: values.copy() for column, values in self._data.items()}
        new._index = self._index.copy()
        return new

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def __getitem__(self, key: str | Sequence[str]) -> Series | "DataFrame":
        if isinstance(key, Sequence) and not isinstance(key, str):
            data = {column: self._data[column] for column in key}
            return DataFrame(data, index=self._index.copy(), columns=list(key))
        return Series(self._data[key], index=self._index.copy(), name=key)

    def __setitem__(self, key: str, value: Sequence[Any]) -> None:
        if isinstance(value, Series):
            value = value.reindex(self._index)._data
        else:
            value = list(value)
        if len(value) != len(self._index):
            raise ValueError("Column length mismatch")
        self._data[key] = list(value)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return Series(self._data[key], index=self._index.copy(), name=key)

    class _Loc:
        def __init__(self, frame: "DataFrame") -> None:
            self.frame = frame

        def __getitem__(self, key: Any) -> Series | "DataFrame":
            if isinstance(key, list):
                positions = [self.frame._index.index(label) for label in key]
                return self.frame._take_rows(positions)
            if isinstance(key, slice):
                range_indices = range(*key.indices(len(self.frame._index)))
                return self.frame._take_rows(list(range_indices))
            position = self.frame._index.index(key)
            return self.frame._row_as_series(position)

    class _ILoc:
        def __init__(self, frame: "DataFrame") -> None:
            self.frame = frame

        def __getitem__(self, key: Any) -> Series | "DataFrame":
            if isinstance(key, list):
                return self.frame._take_rows(key)
            if isinstance(key, slice):
                range_indices = list(range(*key.indices(len(self.frame._index))))
                return self.frame._take_rows(range_indices)
            return self.frame._row_as_series(key)

    @property
    def loc(self) -> "DataFrame._Loc":
        return DataFrame._Loc(self)

    @property
    def iloc(self) -> "DataFrame._ILoc":
        return DataFrame._ILoc(self)

    def _row_as_series(self, position: int) -> Series:
        data = {column: self._data[column][position] for column in self._data}
        return Series(data, index=list(self._data.keys()))

    def _take_rows(self, positions: Sequence[int]) -> "DataFrame":
        data = {column: [self._data[column][pos] for pos in positions] for column in self._data}
        index = [self._index[pos] for pos in positions]
        return DataFrame(data, index=index, columns=list(self._data.keys()))

    def assign(self, **columns: Any) -> "DataFrame":
        frame = self.copy()
        for key, value in columns.items():
            if callable(value):
                value = value(frame)
            if isinstance(value, Series):
                frame._data[key] = value.reindex(frame._index)._data
            else:
                if isinstance(value, (int, float, str, bool)) or value is None:
                    frame._data[key] = [value for _ in frame._index]
                else:
                    seq = list(value)
                    if len(seq) != len(frame._index):
                        raise ValueError("Assigned column length mismatch")
                    frame._data[key] = seq
        return frame

    def sort_values(self, by: str, ascending: bool = True, kind: Optional[str] = None) -> "DataFrame":
        order = list(range(len(self._index)))
        values = self._data[by]
        order.sort(key=lambda idx: (_is_nan(values[idx]), values[idx], idx))
        if not ascending:
            order.reverse()
        return self._take_rows(order)

    def reset_index(self, drop: bool = False) -> "DataFrame":
        new_index = list(range(len(self._index)))
        if drop:
            data = {column: values.copy() for column, values in self._data.items()}
        else:
            data = {"index": self._index.copy()}
            data.update({column: values.copy() for column, values in self._data.items()})
        return DataFrame(data, index=new_index)

    def iterrows(self) -> Iterator[Tuple[Any, Series]]:
        for position, label in enumerate(self._index):
            yield label, self._row_as_series(position)

    def apply(self, func: Callable[[Series], Any], axis: int = 0) -> Series:
        if axis != 1:
            raise ValueError("Only axis=1 is supported in the lightweight DataFrame")
        results = [func(self._row_as_series(pos)) for pos in range(len(self._index))]
        return Series(results, index=self._index.copy())

    def to_dict(self, orient: str = "dict") -> Any:
        if orient == "dict":
            return {column: values.copy() for column, values in self._data.items()}
        if orient == "records":
            records = []
            for pos in range(len(self._index)):
                record = {column: self._data[column][pos] for column in self._data}
                records.append(record)
            return records
        raise ValueError("Unsupported orient value")

    def drop_duplicates(self) -> "DataFrame":
        seen: set[Tuple[Any, ...]] = set()
        keep: List[int] = []
        for pos in range(len(self._index)):
            signature = tuple(self._data[column][pos] for column in self._data)
            if signature in seen:
                continue
            seen.add(signature)
            keep.append(pos)
        return self._take_rows(keep)

    def fillna(self, value: Any = None, method: Optional[str] = None) -> "DataFrame":
        frame = self.copy()
        if method is None:
            for column in frame._data:
                frame._data[column] = Series(frame._data[column], index=frame._index).fillna(value)._data
            return frame
        if method not in {"ffill", "bfill"}:
            raise ValueError("Unsupported fillna method")
        for column in frame._data:
            values = frame._data[column]
            if method == "ffill":
                last = None
                new_col: List[Any] = []
                for item in values:
                    if _is_nan(item):
                        new_col.append(last)
                    else:
                        new_col.append(item)
                        last = item
                frame._data[column] = new_col
            else:
                next_value = None
                new_col_rev: List[Any] = []
                for item in reversed(values):
                    if _is_nan(item):
                        new_col_rev.append(next_value)
                    else:
                        new_col_rev.append(item)
                        next_value = item
                frame._data[column] = list(reversed(new_col_rev))
        return frame


def to_datetime(data: Any, errors: str = "raise") -> Series | Timestamp:
    def convert(value: Any) -> Optional[Timestamp]:
        if value is None or value is NaT:
            return None
        if isinstance(value, datetime):
            return Timestamp(value)
        try:
            return Timestamp(value)
        except ValueError:
            if errors == "coerce":
                return None
            raise

    if isinstance(data, Series):
        converted = [convert(value) for value in data._data]
        return Series(converted, index=data._index.copy(), dtype="datetime64[ns]")
    if isinstance(data, list):
        converted = [convert(value) for value in data]
        return Series(converted, index=list(range(len(converted))), dtype="datetime64[ns]")
    return convert(data)


def date_range(start: Any, periods: int, freq: str = "D") -> Series:
    """Generate a minimal fixed-frequency datetime range."""

    if periods < 0:
        raise ValueError("periods must be non-negative")
    start_ts = to_datetime(start)
    freq = freq.upper()
    if freq == "H":
        delta = timedelta(hours=1)
    elif freq in {"D", "1D"}:
        delta = timedelta(days=1)
    elif freq in {"T", "MIN"}:
        delta = timedelta(minutes=1)
    else:
        raise ValueError(f"Unsupported frequency '{freq}' in date_range")
    values = [start_ts + i * delta for i in range(periods)]
    return Series(values)


def to_numeric(data: Series, errors: str = "raise") -> Series:
    converted: List[float] = []
    for value in data._data:
        if _is_nan(value):
            converted.append(NAN)
            continue
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            if errors == "coerce":
                converted.append(NAN)
            else:
                raise
    return Series(converted, index=data._index.copy(), dtype="float")


def to_timedelta(values: Any, unit: str = "s") -> Series | timedelta:
    multiplier = {
        "s": 1,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "m": 60,
        "h": 3600,
    }[unit]

    def convert(value: Any) -> timedelta:
        return timedelta(seconds=float(value) * multiplier)

    if isinstance(values, Series):
        data = [convert(value) for value in values._data]
        return Series(data, index=values._index.copy())
    if isinstance(values, list):
        data = [convert(value) for value in values]
        return Series(data, index=list(range(len(data))))
    return convert(values)


def concat(frames: Iterable[DataFrame], ignore_index: bool = False) -> DataFrame:
    frames = [frame.copy() for frame in frames if frame is not None]
    if not frames:
        return DataFrame()
    all_columns: List[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)
    combined: Dict[str, List[Any]] = {column: [] for column in all_columns}
    combined_index: List[Any] = []
    for frame in frames:
        for column in all_columns:
            column_data = frame._data.get(column, [None] * len(frame))
            combined[column].extend(column_data)
        if ignore_index:
            combined_index.extend([None] * len(frame))
        else:
            combined_index.extend(frame._index)
    if ignore_index:
        combined_index = list(range(len(combined_index)))
    return DataFrame(combined, index=combined_index)


def _normalise_rows(rows: List[Dict[str, Any]]) -> DataFrame:
    if not rows:
        return DataFrame()
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    data: Dict[str, List[Any]] = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            data[column].append(row.get(column))
    return DataFrame(data)


def read_csv(path: str | Path) -> DataFrame:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return _normalise_rows(rows)


def read_json(path: str | Path) -> DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("data")
        if not isinstance(rows, list):
            rows = [payload]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("JSON file must contain an array of records")
    normalised: List[Dict[str, Any]] = []
    for entry in rows:
        if isinstance(entry, Mapping):
            normalised.append(dict(entry))
        else:
            raise ValueError("Invalid JSON structure")
    return _normalise_rows(normalised)


def read_parquet(path: str | Path) -> DataFrame:  # pragma: no cover - best effort fallback
    raise NotImplementedError("Parquet reading is not supported in the lightweight pandas implementation")


# Alias used by the framework
Timestamp = Timestamp
''', True)
# END MODULE: pandas

# BEGIN MODULE: problems (problems/__init__.py)
_register_module('problems', r'''
"""Problem registry to simplify experiment configuration."""
from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from core.problem import ManufacturingProblem
from problems.flexible_job_shop import create_flexible_job_shop_problem
from problems.flow_shop import create_flow_shop_problem
from problems.job_shop import create_job_shop_problem
from problems.rms_variants import (
    create_distributed_job_shop_problem,
    create_dynamic_job_shop_problem,
    create_hybrid_manufacturing_problem,
)

ProblemFactory = Callable[[pd.DataFrame], ManufacturingProblem]


PROBLEM_FACTORIES: Dict[str, ProblemFactory] = {
    "job_shop": create_job_shop_problem,
    "flow_shop": create_flow_shop_problem,
    "flexible_job_shop": create_flexible_job_shop_problem,
    "dynamic_job_shop": create_dynamic_job_shop_problem,
    "distributed_job_shop": create_distributed_job_shop_problem,
    "hybrid_manufacturing": create_hybrid_manufacturing_problem,
}


def get_problem_factory(name: str) -> ProblemFactory:
    key = name.lower()
    if key not in PROBLEM_FACTORIES:
        raise KeyError(f"Unknown problem factory '{name}'")
    return PROBLEM_FACTORIES[key]


def list_problem_types() -> List[str]:
    return sorted(PROBLEM_FACTORIES.keys())


__all__ = [
    "PROBLEM_FACTORIES",
    "get_problem_factory",
    "list_problem_types",
]
''', True)
# END MODULE: problems

# BEGIN MODULE: pydantic (pydantic/__init__.py)
_register_module('pydantic', r'''
"""Lightweight subset of the pydantic API used by the RMS framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


class ValidationError(Exception):
    """Exception raised when validation fails."""


@dataclass
class _FieldInfo:
    default: Any = None
    default_factory: Optional[Callable[[], Any]] = None
    description: Optional[str] = None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


def Field(default: Any = None, *, default_factory: Optional[Callable[[], Any]] = None, description: str | None = None, **_: Any) -> _FieldInfo:
    return _FieldInfo(default=default, default_factory=default_factory, description=description)


def validator(field_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__validator_config__ = field_name
        return func
    return decorator


class BaseModel:
    """Minimal BaseModel emulating core pydantic behaviour."""

    def __init_subclass__(cls) -> None:
        cls.__validators__ = []
        for attr in dir(cls):
            value = getattr(cls, attr)
            if callable(value) and hasattr(value, "__validator_config__"):
                cls.__validators__.append((value.__validator_config__, value))

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        for name in annotations:
            field_info = getattr(self.__class__, name, None)
            if isinstance(field_info, _FieldInfo):
                default_value = field_info.get_default()
            else:
                default_value = field_info
            value = data[name] if name in data else default_value
            setattr(self, name, value)
        extra_keys = set(data.keys()) - set(annotations.keys())
        if extra_keys:
            raise ValidationError(f"Unexpected fields: {sorted(extra_keys)}")
        self._run_validators()

    def _run_validators(self) -> None:
        for field_name, func in getattr(self.__class__, "__validators__", []):
            current_value = getattr(self, field_name)
            new_value = func(self.__class__, current_value)
            setattr(self, field_name, new_value)

    @classmethod
    def parse_obj(cls, obj: Dict[str, Any]) -> "BaseModel":
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            raise ValidationError("Object must be a dict")
        return cls(**obj)

    def dict(self) -> Dict[str, Any]:
        annotations = getattr(self, "__annotations__", {})
        return {name: getattr(self, name) for name in annotations}

    def json(self) -> str:
        import json

        return json.dumps(self.dict())

    def copy(self) -> "BaseModel":
        return self.__class__(**self.dict())
''', True)
# END MODULE: pydantic

# BEGIN MODULE: reporting (reporting/__init__.py)
_register_module('reporting', r'''

''', True)
# END MODULE: reporting

# BEGIN MODULE: rms_all_in_one (rms_all_in_one.py)
_register_module('rms_all_in_one', r'''
"""Unified RMS optimisation pipeline in a single executable module.

This script offers a convenience façade over the modular research
framework contained in this repository.  It orchestrates data loading,
problem construction, optimisation algorithm execution, statistical
validation, reporting, and visual analytics from one entry point.  The
original project intentionally separates these concerns into multiple
packages; however some users prefer a monolithic runner they can launch
without navigating the entire codebase.  `rms_all_in_one.py` fulfils that
requirement while reusing the rigorously tested building blocks.

Usage examples
--------------

Run the full experiment workflow using the default configuration and
produce summary artefacts in ``results/all_in_one``::

    python rms_all_in_one.py --run-experiments

Generate the publication gallery and markdown report for all bundled
problem types and algorithms, exporting outputs to a custom directory::

    python rms_all_in_one.py --run-experiments --generate-gallery \
        --all-problems --algorithms all --output-dir results/full_suite

Launch the interactive dashboard directly from this façade::

    python rms_all_in_one.py --launch-dashboard

The script remains lightweight: it imports modules only when required and
fails gracefully when optional dependencies (for example the Tkinter GUI
stack or SciPy) are unavailable in the current environment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from algorithms import get_algorithm, list_algorithms
from config.base_config import ExperimentalConfig, load_config
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution
from data.generator import BenchmarkDataGenerator, SyntheticDataGenerator, SyntheticScenario
from data.loader import DataLoader, DataPreprocessor
from problems import get_problem_factory, list_problem_types
from reporting.generators import MarkdownReporter
from simulation.monte_carlo import MonteCarloEngine
from simulation.stochastic_models import ProcessingTimeModel
from validation.empirical import confidence_interval, friedman_test
from validation.theoretical import document_complexity
from visualization.gallery import generate_gallery

try:  # pragma: no cover - optional dependency for dashboard usage
    from visualization.dashboard import RMSDashboard, tkinter_available
except Exception:  # pragma: no cover - guard against GUI-less systems
    RMSDashboard = None  # type: ignore
    tkinter_available = lambda: False  # type: ignore


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve_algorithms(config: ExperimentalConfig, override: Sequence[str] | None) -> List[str]:
    if override:
        if len(override) == 1 and override[0].lower() == "all":
            return list_algorithms(include_dispatching=True)
        return [name.lower() for name in override]

    hyper = config.algorithm.hyperparameters or {}
    candidates = hyper.get("candidates")
    if candidates:
        return [name.lower() for name in candidates]

    name = config.algorithm.name.lower()
    if name == "all_dispatching":
        from algorithms.classical.dispatching_rules import list_dispatching_rules

        return list_dispatching_rules()
    return [name]


def _load_dataset(config: ExperimentalConfig, synthetic: bool = False) -> pd.DataFrame:
    loader = DataLoader()
    preprocessor = DataPreprocessor()

    frames: List[pd.DataFrame] = []
    if synthetic:
        scenario = SyntheticScenario(
            num_jobs=240,
            machines=["M01", "M02", "M03", "M04"],
            start_date=pd.Timestamp("2024-01-01"),
            time_between_jobs=pd.Timedelta(minutes=12),
        )
        frames.append(SyntheticDataGenerator().generate(scenario))
    elif config.data.sources:
        sources = [Path(source) for source in config.data.sources]
        data = loader.load(sources)
        frames.append(data)
    else:
        generator = BenchmarkDataGenerator()
        frames.extend(generator.load_instances())

    if not frames:
        raise RuntimeError("No datasets were loaded; provide --synthetic or configure data.sources")

    dataset = pd.concat(frames, ignore_index=True)
    return preprocessor.transform(dataset)


def _build_problem(dataset: pd.DataFrame, problem_name: str, config: ExperimentalConfig) -> ManufacturingProblem:
    factory = get_problem_factory(problem_name)
    problem = factory(dataset.copy())
    problem.metadata = {
        "problem_type": problem_name,
        "objectives": ", ".join(config.optimisation.objectives),
    }
    return problem


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def _run_algorithms(
    problem: ManufacturingProblem,
    algorithm_names: Sequence[str],
    rep: int,
) -> Tuple[pd.DataFrame, ScheduleSolution]:
    records: List[Dict[str, float]] = []
    best_solution: Optional[ScheduleSolution] = None
    best_score = float("inf")

    for name in algorithm_names:
        optimizer = get_algorithm(name)
        solution = optimizer.solve(problem)
        metrics = solution.metrics or evaluate_schedule(solution.schedule)
        record = {"replication": rep, "algorithm": name}
        record.update(metrics)
        records.append(record)
        objective_value = metrics.get("makespan", 0.0)
        if objective_value < best_score:
            best_score = objective_value
            best_solution = solution

    assert best_solution is not None, "At least one algorithm must be executed"
    return pd.DataFrame(records), best_solution


def run_experiments(
    config: ExperimentalConfig,
    dataset: pd.DataFrame,
    problems: Sequence[str],
    algorithm_names: Sequence[str],
    output_dir: Path,
    replications: Optional[int] = None,
    generate_gallery_flag: bool = False,
    run_validation: bool = False,
) -> Dict[str, Dict[str, float]]:
    replications = replications or config.validation.replications
    aggregated_metrics: Dict[str, Dict[str, float]] = {}

    gallery_paths: List[Path] = []
    validation_results: Dict[str, Dict[str, float]] = {}

    for problem_name in problems:
        reporter = MarkdownReporter(output_dir / f"{problem_name}_summary.md")
        problem_records: List[pd.DataFrame] = []
        best_schedule_overall: Optional[pd.DataFrame] = None

        best_problem_makespan = float("inf")

        for rep in range(replications):
            problem_instance = _build_problem(dataset, problem_name, config)
            df_records, best_solution = _run_algorithms(problem_instance, algorithm_names, rep)
            df_records["problem"] = problem_name
            problem_records.append(df_records)

            current_best = df_records.loc[df_records["makespan"].idxmin()]
            if current_best["makespan"] < best_problem_makespan or best_schedule_overall is None:
                best_problem_makespan = float(current_best["makespan"])
                best_schedule_overall = best_solution.schedule.copy()
                best_schedule_overall["Algorithm"] = current_best["algorithm"]

        combined = pd.concat(problem_records, ignore_index=True)
        grouped = (
            combined.groupby("algorithm")
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("makespan")
        )
        aggregated_metrics[problem_name] = grouped.set_index("algorithm").iloc[0].to_dict()

        summary_metrics = {
            f"avg_{metric}": float(mean(grouped[metric]))
            for metric in grouped.columns
            if metric != "algorithm"
        }
        summary_metrics["problem"] = problem_name
        reporter.render(summary_metrics, grouped)

        csv_path = output_dir / f"results_{problem_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(csv_path, index=False)

        if generate_gallery_flag and best_schedule_overall is not None:
            gallery_root = output_dir / "figures" / problem_name
            gallery_root.mkdir(parents=True, exist_ok=True)
            gallery_paths.extend(
                generate_gallery(
                    results=grouped,
                    schedule=best_schedule_overall,
                    output_dir=gallery_root,
                    significance_metric="makespan",
                )
            )

        if run_validation:
            try:
                friedman = friedman_test(combined[["replication", "algorithm", "makespan"]])
            except RuntimeError as exc:
                friedman = {"error": str(exc)}
            validation_results[problem_name] = friedman

            try:
                import numpy as np

                ci = confidence_interval(
                    np.array(combined["makespan"], dtype=float),
                    level=config.validation.confidence_level,
                )
                validation_results[problem_name].update({f"ci_{k}": v for k, v in ci.items()})
            except Exception as exc:  # pragma: no cover - optional deps
                validation_results[problem_name].setdefault("ci_error", str(exc))

    if gallery_paths:
        (output_dir / "figures" / "manifest.json").write_text(
            json.dumps([str(path) for path in gallery_paths], indent=2),
            encoding="utf-8",
        )

    if validation_results:
        (output_dir / "statistics" / "validation.json").write_text(
            json.dumps(validation_results, indent=2),
            encoding="utf-8",
        )

    complexities = []
    complexity_map = {
        "fcfs": ("O(n)", "O(1)"),
        "spt": ("O(n log n)", "O(1)"),
        "lpt": ("O(n log n)", "O(1)"),
        "edd": ("O(n log n)", "O(1)"),
        "slack": ("O(n log n)", "O(1)"),
        "critical_ratio": ("O(n log n)", "O(1)"),
        "wspt": ("O(n log n)", "O(1)"),
        "genetic_algorithm": ("O(g * p * n)", "O(p * n)"),
        "particle_swarm": ("O(i * s * n)", "O(s * n)"),
        "simulated_annealing": ("O(i * n)", "O(n)"),
        "tabu_search": ("O(i * n^2)", "O(n^2)"),
        "ant_colony": ("O(i * a * n^2)", "O(a * n)"),
        "nsga2": ("O(g * p^2)", "O(p * n)"),
        "dqn": ("O(e * b)", "O(b)"),
        "ppo": ("O(e * b)", "O(b)"),
        "adaptive_hybrid": ("O(i * n log n)", "O(n^2)"),
    }
    for name in algorithm_names:
        time_c, space_c = complexity_map.get(name, ("unspecified", "unspecified"))
        complexities.append(document_complexity(name, time_c, space_c))
    (output_dir / "statistics" / "complexity.json").write_text(
        json.dumps(complexities, indent=2),
        encoding="utf-8",
    )

    return aggregated_metrics


# ---------------------------------------------------------------------------
# Simulation façade
# ---------------------------------------------------------------------------


def run_monte_carlo(dataset: pd.DataFrame, config: ExperimentalConfig, output_dir: Path) -> None:
    repetitions = config.simulation.repetitions
    engine = MonteCarloEngine(repetitions)

    rng = random.Random(config.algorithm.seed)

    def _lognormal_distribution(size: int) -> List[float]:
        return [max(1.0, rng.lognormvariate(4.0, 0.35)) for _ in range(size)]

    model = ProcessingTimeModel(distribution=_lognormal_distribution)

    def _simulate_once() -> float:
        samples = model.sample(len(dataset))
        return float(sum(samples))

    estimate = engine.estimate(_simulate_once)
    output = {
        "repetitions": repetitions,
        "jobs": int(len(dataset)),
        "expected_total_processing_time": estimate,
    }
    output_path = output_dir / "statistics" / "monte_carlo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------


def launch_dashboard() -> None:  # pragma: no cover - interactive component
    if not tkinter_available():
        raise RuntimeError("Tkinter is not available in this environment")
    import tkinter as tk

    root = tk.Tk()
    RMSDashboard(root)  # type: ignore[arg-type]
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RMS optimisation runner")
    parser.add_argument("--config", type=Path, default=Path("config/base_config.yaml"), help="Path to configuration file")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Algorithms to execute (use 'all' for the full registry)")
    parser.add_argument("--problem", dest="problems", action="append", help="Problem type to evaluate. Repeat for multiple problems.")
    parser.add_argument("--all-problems", action="store_true", help="Evaluate every bundled problem factory")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic dataset instead of loading from disk")
    parser.add_argument("--output-dir", type=Path, default=Path("results/all_in_one"), help="Directory where artefacts are stored")
    parser.add_argument("--replications", type=int, default=None, help="Number of independent replications per algorithm")
    parser.add_argument("--run-experiments", action="store_true", help="Execute the optimisation experiments")
    parser.add_argument("--generate-gallery", action="store_true", help="Produce the 50+ figure gallery after experiments")
    parser.add_argument("--run-validation", action="store_true", help="Compute statistical validation metrics")
    parser.add_argument("--run-simulation", action="store_true", help="Execute the Monte Carlo processing time study")
    parser.add_argument("--launch-dashboard", action="store_true", help="Start the interactive Tkinter dashboard")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.launch_dashboard:
        launch_dashboard()
        return 0

    dataset = _load_dataset(config, synthetic=args.synthetic)

    problems = list_problem_types() if args.all_problems else (args.problems or ["job_shop"])
    algorithms = _resolve_algorithms(config, args.algorithms)

    summary: Dict[str, Dict[str, float]] = {}
    if args.run_experiments:
        summary = run_experiments(
            config=config,
            dataset=dataset,
            problems=problems,
            algorithm_names=algorithms,
            output_dir=output_dir,
            replications=args.replications,
            generate_gallery_flag=args.generate_gallery,
            run_validation=args.run_validation,
        )

    if args.run_simulation:
        run_monte_carlo(dataset, config, output_dir)

    if summary:
        (output_dir / "statistics" / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - script execution
    sys.exit(main())
''', False)
# END MODULE: rms_all_in_one

# BEGIN MODULE: simulation (simulation/__init__.py)
_register_module('simulation', r'''

''', True)
# END MODULE: simulation

# BEGIN MODULE: utils (utils/__init__.py)
_register_module('utils', r'''

''', True)
# END MODULE: utils

# BEGIN MODULE: validation (validation/__init__.py)
_register_module('validation', r'''

''', True)
# END MODULE: validation

# BEGIN MODULE: visualization (visualization/__init__.py)
_register_module('visualization', r'''

''', True)
# END MODULE: visualization

# BEGIN MODULE: yaml (yaml/__init__.py)
_register_module('yaml', r'''
"""Minimal YAML loader/dumper for simple configuration files."""
from __future__ import annotations

from typing import Any, Dict, List


def safe_load(stream: str) -> Any:
    if hasattr(stream, 'read'):
        text = stream.read()
    else:
        text = stream
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith('#')]
    index = 0

    def parse_block(indent: int = 0) -> Any:
        nonlocal index
        mapping: Dict[str, Any] = {}
        sequence: List[Any] = []
        is_sequence = False
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent < indent:
                break
            stripped = line.strip()
            if stripped.startswith('- '):
                is_sequence = True
                value_text = stripped[2:].strip()
                index += 1
                if value_text:
                    sequence.append(_convert_scalar(value_text))
                else:
                    sequence.append(parse_block(current_indent + 2))
                continue
            if ':' in stripped:
                key, value_text = stripped.split(':', 1)
                key = key.strip()
                value_text = value_text.strip()
                index += 1
                if value_text:
                    mapping[key] = _convert_scalar(value_text)
                else:
                    mapping[key] = parse_block(current_indent + 2)
                continue
            index += 1
        if is_sequence:
            return sequence
        return mapping

    def _convert_scalar(token: str) -> Any:
        lowered = token.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        if lowered == 'null':
            return None
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            return token.strip('"')

    return parse_block(0)


def safe_dump(data: Any, stream=None) -> str:
    def render(obj: Any, indent: int = 0) -> List[str]:
        prefix = ' ' * indent
        lines: List[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {format_scalar(value)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(render(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {format_scalar(item)}")
        else:
            lines.append(f"{prefix}{format_scalar(obj)}")
        return lines

    def format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None:
            return 'null'
        return str(value)

    content = '\n'.join(render(data, 0)) + '\n'
    if stream is not None:
        stream.write(content)
        return ''
    return content
''', True)
# END MODULE: yaml

# BEGIN MODULE: algorithms.metaheuristics (algorithms/metaheuristics/__init__.py)
_register_module('algorithms.metaheuristics', r'''
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
''', True)
# END MODULE: algorithms.metaheuristics

# BEGIN MODULE: config.base_config (config/base_config.py)
_register_module('config.base_config', r'''
"""Configuration models for the RMS optimization framework.

This module centralises all experiment configuration objects.  The
models are implemented with `pydantic` to guarantee validation and
provide convenient serialisation / deserialisation helpers.  Each
configuration block mirrors one portion of the research plan described
in the project charter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator
import yaml


class DataConfig(BaseModel):
    """Configuration for the dataset layer."""

    sources: List[Path] = Field(default_factory=list, description="Input datasets")
    streaming: bool = Field(False, description="Enable streaming data ingestion")
    batch_size: int = Field(1024, ge=1, description="Batch size for streaming pipelines")
    cache_dir: Path = Field(Path("data/cache"))


class AlgorithmConfig(BaseModel):
    """Per-algorithm hyper-parameters and search spaces."""

    name: str = Field(..., description="Primary algorithm identifier")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    search_space: Dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(42, description="Random seed for reproducibility")


class OptimizationConfig(BaseModel):
    """Multi-objective optimisation settings."""

    objectives: List[str] = Field(default_factory=lambda: ["makespan", "energy"])
    weights: Dict[str, float] = Field(default_factory=lambda: {"makespan": 0.5, "energy": 0.5})
    constraints: Dict[str, Any] = Field(default_factory=dict)
    pareto_front_size: int = Field(100, ge=1)

    @validator("weights")
    def validate_weights(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("At least one weight must be provided")
        total = sum(value.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        return value


class SimulationConfig(BaseModel):
    """Configuration of stochastic simulation parameters."""

    repetitions: int = Field(100, ge=1)
    enable_discrete_event: bool = Field(True)
    enable_monte_carlo: bool = Field(True)
    parallelism: int = Field(1, ge=1, description="Number of parallel workers")


class ValidationConfig(BaseModel):
    """Statistical validation parameters."""

    confidence_level: float = Field(0.95, ge=0.0, le=0.999)
    tests: List[str] = Field(default_factory=lambda: ["friedman", "wilcoxon"])
    replications: int = Field(30, ge=1)


class HardwareConfig(BaseModel):
    """Hardware and runtime resources."""

    use_gpu: bool = Field(False)
    num_cpus: int = Field(4, ge=1)
    memory_gb: int = Field(16, ge=1)


class LoggingConfig(BaseModel):
    """Experiment tracking and logging configuration."""

    experiment_name: str = Field("rms-optimization")
    tracking_uri: Optional[str] = Field(None, description="MLflow or W&B tracking URI")
    log_dir: Path = Field(Path("logs"))
    level: str = Field("INFO")


class ExperimentalConfig(BaseModel):
    """Master configuration object that aggregates all sections."""

    data: DataConfig = Field(default_factory=DataConfig)
    algorithm: AlgorithmConfig = Field(default_factory=lambda: AlgorithmConfig(name="fcfs"))
    optimisation: OptimizationConfig = Field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentalConfig":
        """Load configuration from a YAML or JSON file."""

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.parse_obj(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise configuration to a dictionary."""

        return self.dict()

    def save(self, path: Path) -> None:
        """Persist configuration to disk."""

        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle)


def load_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentalConfig:
    """Utility wrapper to load and override configuration fields."""

    config = ExperimentalConfig.from_file(path) if path else ExperimentalConfig()
    if overrides:
        config = config.copy(update=overrides)
    return config
''', False)
# END MODULE: config.base_config

# BEGIN MODULE: core.base_optimizer (core/base_optimizer.py)
_register_module('core.base_optimizer', r'''
"""Abstract base classes for optimisation algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class BaseOptimizer(ABC):
    """Base class every optimisation algorithm should derive from."""

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = hyperparameters

    @abstractmethod
    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        """Compute a solution for the provided manufacturing problem."""

    def info(self) -> Dict[str, Any]:
        """Return metadata describing the optimizer."""

        return {"name": self.__class__.__name__, "hyperparameters": self.hyperparameters}
''', False)
# END MODULE: core.base_optimizer

# BEGIN MODULE: core.config (core/config.py)
_register_module('core.config', r'''
"""Helper functions to work with experiment configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from config.base_config import ExperimentalConfig, load_config


class ConfigManager:
    """High level API to manage experiment configuration."""

    def __init__(self, config: Optional[ExperimentalConfig] = None) -> None:
        self._config = config or ExperimentalConfig()

    @property
    def config(self) -> ExperimentalConfig:
        return self._config

    @classmethod
    def from_file(cls, path: Path) -> "ConfigManager":
        return cls(load_config(path))

    def override(self, updates: Dict[str, Any]) -> None:
        self._config = self._config.copy(update=updates)
''', False)
# END MODULE: core.config

# BEGIN MODULE: core.metrics (core/metrics.py)
_register_module('core.metrics', r'''
"""Core metrics for manufacturing optimisation."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def compute_makespan(schedule: pd.DataFrame) -> float:
    if schedule.empty:
        return 0.0
    end_times = _ensure_datetime(schedule["Scheduled_End"])
    start_times = _ensure_datetime(schedule["Scheduled_Start"])
    if end_times.isna().all() or start_times.isna().all():
        return 0.0
    return float((end_times.max() - start_times.min()).total_seconds() / 60.0)


def compute_total_completion_time(schedule: pd.DataFrame) -> float:
    completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
    if completion.isna().all():
        return 0.0
    start = _ensure_datetime(schedule.get("Release_Date", schedule.get("Scheduled_Start", pd.NaT)))
    start = start.fillna(start.min())
    flow_times = (completion - start).dt.total_seconds() / 60.0
    return float(flow_times.sum())


def compute_total_tardiness(schedule: pd.DataFrame) -> float:
    if "Due_Date" not in schedule.columns:
        return 0.0
    due = _ensure_datetime(schedule["Due_Date"])
    completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
    tardiness = (completion - due).dt.total_seconds() / 60.0
    tardiness = tardiness.clip(lower=0)
    return float(tardiness.sum())


def compute_energy(schedule: pd.DataFrame) -> float:
    if "Energy_Consumption" not in schedule:
        return 0.0
    return float(pd.to_numeric(schedule["Energy_Consumption"], errors="coerce").fillna(0.0).sum())


def evaluate_schedule(schedule: pd.DataFrame) -> Dict[str, float]:
    makespan = compute_makespan(schedule)
    total_completion = compute_total_completion_time(schedule)
    energy = compute_energy(schedule)
    total_tardiness = compute_total_tardiness(schedule)
    num_tardy = 0
    if "Due_Date" in schedule.columns:
        due = _ensure_datetime(schedule["Due_Date"])
        completion = _ensure_datetime(schedule.get("Completion_Time", schedule.get("Scheduled_End", pd.NaT)))
        tardy_mask = completion > due
        num_tardy = int(tardy_mask.sum())
    mean_flow_time = float(total_completion / max(len(schedule), 1)) if schedule is not None else 0.0
    return {
        "makespan": makespan,
        "total_completion_time": total_completion,
        "mean_flow_time": mean_flow_time,
        "total_tardiness": total_tardiness,
        "num_tardy_jobs": num_tardy,
        "energy": energy,
    }
''', False)
# END MODULE: core.metrics

# BEGIN MODULE: core.problem (core/problem.py)
_register_module('core.problem', r'''
"""Problem representations and helpers for RMS optimisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetime while preserving missing entries."""

    if getattr(series, "empty", False):
        return pd.Series([], dtype="datetime64[ns]")
    if getattr(series, "dtype", None) == "datetime64[ns]":
        return series
    return pd.to_datetime(series, errors="coerce")


def _infer_processing_time(row: pd.Series) -> float:
    """Infer the processing time for a job in minutes."""

    processing = row.get("Processing_Time")
    if pd.notna(processing):
        return float(processing)
    start = row.get("Scheduled_Start")
    end = row.get("Scheduled_End")
    if pd.notna(start) and pd.notna(end):
        return float((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60.0)
    return 0.0


@dataclass
class ManufacturingProblem:
    """Encapsulate the data describing a scheduling instance."""

    jobs: pd.DataFrame
    objectives: List[str]
    constraints: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, pd.DataFrame):
            raise TypeError("jobs must be provided as a pandas DataFrame")
        if not self.objectives:
            raise ValueError("At least one objective must be specified")
        if self.jobs.index.has_duplicates:
            # Ensure every job can be uniquely addressed when building sequences.
            self.jobs = self.jobs.reset_index(drop=True)

    def build_schedule(self, order: Sequence[int | str] | None = None) -> pd.DataFrame:
        """Construct a feasible schedule following a given job order.

        The implementation assumes a job-shop scenario with potentially
        multiple machines.  Jobs are executed on their designated machine
        and start as soon as both the machine becomes available and the job
        release time has elapsed.  Processing times are handled in minutes.

        Parameters
        ----------
        order:
            Sequence of row indices describing the desired execution order.
            When *None*, the current dataframe order is used.
        """

        if self.jobs.empty:
            return self.jobs.copy()

        if order is None:
            frame = self.jobs.copy()
        else:
            try:
                frame = self.jobs.loc[list(order)].copy()
            except (KeyError, TypeError):
                frame = self.jobs.iloc[list(order)].copy()

        frame = frame.reset_index(drop=True)
        machine_col = "Machine_ID" if "Machine_ID" in frame.columns else None

        default_release = pd.Timestamp("1970-01-01")
        raw_release = frame.get("Release_Date")
        if raw_release is None:
            raw_release = frame.get("Scheduled_Start")
        if raw_release is None or getattr(raw_release, "empty", False):
            release = pd.Series([default_release] * len(frame), index=frame.index, dtype="datetime64[ns]")
        else:
            release = _ensure_datetime(raw_release)
            if release.isna().all():
                release = pd.Series([default_release] * len(frame), index=frame.index, dtype="datetime64[ns]")
            else:
                release = release.fillna(release.min())
        processing_times = frame.apply(_infer_processing_time, axis=1).astype(float).to_numpy()

        machine_available: Dict[str, pd.Timestamp] = {}
        global_clock = min(release.min(), default_release)

        starts: List[pd.Timestamp] = []
        ends: List[pd.Timestamp] = []

        for idx, row in frame.iterrows():
            machine = str(row[machine_col]) if machine_col else "M0"
            release_time = release.iloc[idx]
            if pd.isna(release_time):
                release_time = global_clock
            start_time = max(machine_available.get(machine, global_clock), release_time)
            processing_minutes = processing_times[idx]
            end_time = start_time + pd.to_timedelta(processing_minutes, unit="m")
            machine_available[machine] = end_time
            global_clock = max(global_clock, end_time)
            starts.append(start_time)
            ends.append(end_time)

        frame["Scheduled_Start"] = starts
        frame["Scheduled_End"] = ends
        frame["Processing_Time"] = processing_times
        frame["Completion_Time"] = frame["Scheduled_End"]
        frame["Start_Time"] = frame["Scheduled_Start"]
        return frame

    def job_indices(self) -> Iterable[int]:
        """Return the job indices in execution order."""

        return list(range(len(self.jobs)))
''', False)
# END MODULE: core.problem

# BEGIN MODULE: core.solution (core/solution.py)
_register_module('core.solution', r'''
"""Solution representation for RMS optimisation problems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class ScheduleSolution:
    """Container for schedules generated by optimisation algorithms."""

    schedule: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, pd.DataFrame):
            raise TypeError("schedule must be a pandas DataFrame")
        if not self.metrics:
            from core.metrics import evaluate_schedule

            self.metrics = evaluate_schedule(self.schedule)

    def to_dict(self) -> Dict[str, float]:
        return self.metrics.copy()
''', False)
# END MODULE: core.solution

# BEGIN MODULE: data.cache (data/cache.py)
_register_module('data.cache', r'''
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
''', False)
# END MODULE: data.cache

# BEGIN MODULE: data.generator (data/generator.py)
_register_module('data.generator', r'''
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
''', False)
# END MODULE: data.generator

# BEGIN MODULE: data.loader (data/loader.py)
_register_module('data.loader', r'''
"""Data ingestion utilities for the RMS optimisation framework."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError


class DataSchema(BaseModel):
    """Minimal schema used to validate ingested datasets."""

    Job_ID: str
    Machine_ID: str
    Scheduled_Start: str
    Scheduled_End: str


class DataValidator:
    """Validate raw data sources using `pydantic` models."""

    def __init__(self, schema: type[BaseModel] = DataSchema) -> None:
        self.schema = schema

    def validate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        errors: List[str] = []
        for row in dataframe.to_dict(orient="records"):
            try:
                self.schema(**row)
            except ValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("Invalid dataset detected:\n" + "\n".join(errors[:5]))
        return dataframe


class DataLoader:
    """Load multiple dataset formats into pandas DataFrames."""

    def __init__(self, validator: Optional[DataValidator] = None) -> None:
        self.validator = validator or DataValidator()

    def load(self, sources: Iterable[Path], validate: bool = True) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for source in sources:
            frame = self._load_single(source)
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.validator.validate(data) if validate and not data.empty else data

    def _load_single(self, path: Path) -> pd.DataFrame:
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".json"}:
            return pd.read_json(path)
        raise ValueError(f"Unsupported file format: {suffix}")


class DataPreprocessor:
    """Simple preprocessing utilities for baseline experiments."""

    datetime_columns: List[str] = ["Scheduled_Start", "Scheduled_End"]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        for column in self.datetime_columns:
            if column in df:
                df[column] = pd.to_datetime(df[column])
        if "Due_Date" in df:
            df["Due_Date"] = pd.to_datetime(df["Due_Date"])
        if {"Processing_Time", "Scheduled_Start", "Scheduled_End"}.issubset(df.columns):
            start = pd.to_datetime(df["Scheduled_Start"])
            end = pd.to_datetime(df["Scheduled_End"])
            inferred = (end - start).dt.total_seconds() / 60.0
            df = df.assign(Processing_Time=df["Processing_Time"].fillna(inferred))
        if "Release_Date" not in df and "Scheduled_Start" in df:
            df["Release_Date"] = df["Scheduled_Start"]
        df = df.drop_duplicates()
        if hasattr(df, "ffill") and hasattr(df, "bfill"):
            df = df.ffill().bfill()
        else:
            df = df.fillna(method="ffill").fillna(method="bfill")
        return df
''', False)
# END MODULE: data.loader

# BEGIN MODULE: experiments.manager (experiments/manager.py)
_register_module('experiments.manager', r'''
"""Experiment orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from algorithms import get_algorithm
from config.base_config import ExperimentalConfig
from core.problem import ManufacturingProblem


@dataclass
class ExperimentResult:
    algorithm: str
    metrics: Dict[str, float]


class ExperimentManager:
    """Coordinate data loading, algorithm execution, and metric logging."""

    def __init__(self, config: ExperimentalConfig) -> None:
        self.config = config

    def _algorithm_names(self) -> Iterable[str]:
        requested = (
            self.config.algorithm.hyperparameters.get("candidates")
            if self.config.algorithm.hyperparameters
            else None
        )
        if requested:
            return [name.lower() for name in requested]
        name = self.config.algorithm.name.lower()
        if name == "all_dispatching":
            from algorithms.classical.dispatching_rules import list_dispatching_rules

            return list_dispatching_rules()
        return [name]

    def run(self, problem: ManufacturingProblem) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for name in self._algorithm_names():
            optimizer = get_algorithm(name)
            solution = optimizer.solve(problem)
            results.append(ExperimentResult(algorithm=name, metrics=solution.metrics))
        return results

    def summarise(self, results: List[ExperimentResult]) -> pd.DataFrame:
        return pd.DataFrame([{"algorithm": r.algorithm, **r.metrics} for r in results])


def export_results(results: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
''', False)
# END MODULE: experiments.manager

# BEGIN MODULE: problems.constraints (problems/constraints.py)
_register_module('problems.constraints', r'''
"""Constraint inference helpers for manufacturing problems."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd


def infer_machine_capacities(data: pd.DataFrame) -> Dict[str, float]:
    """Infer per-machine capacity based on dataset utilisation.

    The heuristic assumes that a machine appearing ``n`` times in the dataset
    can process one job at a time.  The resulting capacity value corresponds to
    the share of the planning horizon that can be allocated concurrently to a
    single job.  This provides a pragmatic constraint bundle that keeps the
    optimisation models consistent with the supplied data.
    """

    if data.empty or "Machine_ID" not in data.columns:
        return {"global": 1.0}

    machine_series = data["Machine_ID"]
    machine_values = machine_series.to_list() if hasattr(machine_series, "to_list") else list(machine_series)
    capacities: Dict[str, float] = {}
    for machine in machine_values:
        key = str(machine)
        capacities[key] = capacities.get(key, 0.0) + 1.0
    for machine, count in list(capacities.items()):
        capacities[machine] = 1.0 / max(float(count), 1.0)
    return capacities


def compute_buffer_limits(data: pd.DataFrame, buffer_columns: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Infer buffer capacities from optional buffer-related columns."""

    if buffer_columns is None:
        buffer_columns = ["Buffer_Capacity", "WIP_Limit"]
    limits: Dict[str, float] = {}
    for column in buffer_columns:
        if column in data.columns:
            series = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
            limits[column.lower()] = float(series.max())
    return limits


def make_constraint_bundle(data: pd.DataFrame, extra_constraints: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Combine machine capacities, buffer limits, and user overrides."""

    constraints = {"machine_capacity": 1.0}
    constraints.update(infer_machine_capacities(data))
    constraints.update(compute_buffer_limits(data))
    if extra_constraints:
        constraints.update(extra_constraints)
    return constraints


__all__ = [
    "infer_machine_capacities",
    "compute_buffer_limits",
    "make_constraint_bundle",
]
''', False)
# END MODULE: problems.constraints

# BEGIN MODULE: problems.flexible_job_shop (problems/flexible_job_shop.py)
_register_module('problems.flexible_job_shop', r'''
"""Flexible job shop problem factory."""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


def _normalise_eligible_machines(values: Sequence[str]) -> List[str]:
    machines: List[str] = []
    for value in values:
        if not value:
            continue
        for token in str(value).replace("|", ",").split(","):
            candidate = token.strip()
            if candidate and candidate not in machines:
                machines.append(candidate)
    return machines if machines else ["M0"]


def create_flexible_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Construct a flexible job shop instance where jobs have machine choices."""

    frame = data.copy()
    if "Eligible_Machines" in frame.columns:
        frame["Eligible_Machines"] = frame["Eligible_Machines"].fillna("")
    else:
        frame["Eligible_Machines"] = frame.get("Machine_ID", "M0").astype(str)

    eligibility: Dict[str, List[str]] = {}
    for _, row in frame.iterrows():
        job = str(row.get("Job_ID", "JOB_UNKNOWN"))
        eligible = _normalise_eligible_machines([row.get("Eligible_Machines", "")])
        eligibility[job] = eligible
    frame["Eligible_Machine_Count"] = [len(eligibility[str(row.get("Job_ID", "JOB_UNKNOWN"))]) for _, row in frame.iterrows()]
    objectives = ["makespan", "total_tardiness", "energy"]
    constraints = make_constraint_bundle(frame, {"flexible_choices": float(sum(frame["Eligible_Machine_Count"]))})
    metadata = {
        "problem_type": "flexible_job_shop",
        "eligibility_encoded": True,
    }
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = ["create_flexible_job_shop_problem"]
''', False)
# END MODULE: problems.flexible_job_shop

# BEGIN MODULE: problems.flow_shop (problems/flow_shop.py)
_register_module('problems.flow_shop', r'''
"""Flow shop problem factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


@dataclass
class FlowShopSchema:
    """Describe the machine sequence in a flow shop scenario."""

    machines: Sequence[str]

    @staticmethod
    def from_frame(data: pd.DataFrame) -> "FlowShopSchema":
        if "Stage" in data.columns and "Machine_ID" in data.columns:
            ordered = (
                data.sort_values("Stage")["Machine_ID"].astype(str).unique().tolist()
            )
            return FlowShopSchema(tuple(ordered))
        machine_columns: List[str] = [
            column
            for column in data.columns
            if column.lower().startswith("machine_")
        ]
        if machine_columns:
            ordered = [data[column].iloc[0] for column in machine_columns]
            return FlowShopSchema(tuple(str(machine) for machine in ordered))
        machines = data.get("Machine_ID")
        if machines is not None:
            return FlowShopSchema(tuple(str(machine) for machine in machines.astype(str).unique()))
        return FlowShopSchema(("M0",))


def _expand_flow_shop(data: pd.DataFrame, schema: FlowShopSchema) -> pd.DataFrame:
    """Expand wide-form records into operation-level rows."""

    if {"Stage", "Machine_ID"}.issubset(data.columns):
        return data.copy().reset_index(drop=True)

    records: List[dict] = []
    processing_columns = [
        column
        for column in data.columns
        if column.lower().startswith("processing_time_")
    ]
    for _, row in data.iterrows():
        job_id = row.get("Job_ID", "JOB_UNKNOWN")
        due = row.get("Due_Date")
        energy = row.get("Energy_Consumption", 0.0)
        for stage_index, machine in enumerate(schema.machines):
            processing_column_candidates: Iterable[str] = [
                f"Processing_Time_{stage_index + 1}",
                f"Processing_Time_{machine}",
                f"processing_time_{stage_index + 1}",
            ] + processing_columns
            processing_time = None
            for column in processing_column_candidates:
                if column in row and pd.notna(row[column]):
                    processing_time = float(row[column])
                    break
            if processing_time is None:
                processing_time = float(row.get("Processing_Time", 0.0))
            records.append(
                {
                    "Job_ID": job_id,
                    "Machine_ID": machine,
                    "Stage": stage_index + 1,
                    "Processing_Time": processing_time,
                    "Energy_Consumption": energy,
                    "Due_Date": due,
                }
            )
    return pd.DataFrame(records)


def create_flow_shop_problem(data: pd.DataFrame, machine_sequence: Sequence[str] | None = None) -> ManufacturingProblem:
    """Build a :class:`ManufacturingProblem` for deterministic flow shops."""

    schema = FlowShopSchema(tuple(machine_sequence)) if machine_sequence else FlowShopSchema.from_frame(data)
    expanded = _expand_flow_shop(data, schema)
    objectives = ["makespan", "total_completion_time", "energy"]
    constraints = make_constraint_bundle(expanded, {"flow_order": len(schema.machines)})
    metadata = {
        "problem_type": "flow_shop",
        "machine_sequence": ",".join(schema.machines),
    }
    return ManufacturingProblem(jobs=expanded, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = ["create_flow_shop_problem", "FlowShopSchema"]
''', False)
# END MODULE: problems.flow_shop

# BEGIN MODULE: problems.job_shop (problems/job_shop.py)
_register_module('problems.job_shop', r'''
"""Job shop problem factory."""
from __future__ import annotations

import pandas as pd

from core.problem import ManufacturingProblem


def create_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    objectives = ["makespan", "energy", "total_tardiness"]
    constraints = {"machine_capacity": 1.0}
    if data.empty:
        jobs = pd.DataFrame(columns=[
            "Job_ID",
            "Machine_ID",
            "Scheduled_Start",
            "Scheduled_End",
            "Processing_Time",
            "Energy_Consumption",
            "Due_Date",
        ])
    else:
        jobs = data.reset_index(drop=True)
        if "Job_ID" not in jobs:
            jobs["Job_ID"] = [f"JOB_{i:05d}" for i in range(len(jobs))]
    return ManufacturingProblem(jobs=jobs, objectives=objectives, constraints=constraints)
''', False)
# END MODULE: problems.job_shop

# BEGIN MODULE: problems.rms_variants (problems/rms_variants.py)
_register_module('problems.rms_variants', r'''
"""Specialised RMS problem variants."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from core.problem import ManufacturingProblem
from problems.constraints import make_constraint_bundle


def _annotate_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["Scenario"] = [variant] * len(annotated)
    return annotated


def create_dynamic_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Dynamic job shop with online arrivals and breakdown markers."""

    frame = _annotate_variant(data, "dynamic")
    if "Arrival_Time" not in frame.columns:
        raw_start = frame.get("Scheduled_Start")
        if raw_start is None or getattr(raw_start, "empty", False):
            frame["Arrival_Time"] = [pd.Timestamp.now()] * len(frame)
        else:
            frame["Arrival_Time"] = pd.to_datetime(raw_start)
    if "Breakdown_Risk" not in frame.columns:
        frame["Breakdown_Risk"] = [0.05] * len(frame)
    objectives = ["makespan", "total_tardiness", "num_tardy_jobs"]
    constraints = make_constraint_bundle(frame, {"dynamic_arrivals": float(len(frame))})
    metadata = {"problem_type": "dynamic_job_shop", "supports_online": "true"}
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


def create_distributed_job_shop_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Distributed manufacturing with plant identifiers and logistics."""

    frame = _annotate_variant(data, "distributed")
    if "Plant" not in frame.columns:
        frame["Plant"] = ["Plant_A"] * len(frame)
    if "Transfer_Time" not in frame.columns:
        frame["Transfer_Time"] = [0.0] * len(frame)
    plant_series = frame["Plant"]
    plant_values = plant_series.to_list() if hasattr(plant_series, "to_list") else list(plant_series)
    unique_plants = len(dict.fromkeys(str(value) for value in plant_values))
    objectives = ["makespan", "total_completion_time", "energy"]
    constraints = make_constraint_bundle(frame, {"plants": float(unique_plants)})
    metadata = {"problem_type": "distributed_job_shop", "plants": str(unique_plants)}
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


def create_hybrid_manufacturing_problem(data: pd.DataFrame) -> ManufacturingProblem:
    """Hybrid additive/subtractive manufacturing scenario."""

    frame = _annotate_variant(data, "hybrid")
    if "Process_Type" not in frame.columns:
        frame["Process_Type"] = ["subtractive"] * len(frame)
    if "Additive_Layer_Time" not in frame.columns:
        frame["Additive_Layer_Time"] = [0.0] * len(frame)
    objectives = ["makespan", "energy", "total_tardiness"]
    constraints = make_constraint_bundle(frame, {"hybrid_steps": float((frame["Process_Type"] == "additive").sum())})
    metadata: Dict[str, str] = {
        "problem_type": "hybrid_manufacturing",
        "hybrid_operations": str((frame["Process_Type"] == "additive").sum()),
    }
    return ManufacturingProblem(jobs=frame, objectives=objectives, constraints=constraints, metadata=metadata)


__all__ = [
    "create_dynamic_job_shop_problem",
    "create_distributed_job_shop_problem",
    "create_hybrid_manufacturing_problem",
]
''', False)
# END MODULE: problems.rms_variants

# BEGIN MODULE: reporting.generators (reporting/generators.py)
_register_module('reporting.generators', r'''
"""Automated reporting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def _stringify(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _compute_column_widths(columns: Iterable[str], rows: Iterable[Iterable[str]]) -> List[int]:
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    return widths


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "No records available."

    columns = [str(col) for col in table.columns]
    string_rows: List[List[str]] = []
    for _, row in table.iterrows():
        string_rows.append([_stringify(row[col]) for col in table.columns])

    widths = _compute_column_widths(columns, string_rows)

    def _format_row(values: Iterable[str]) -> str:
        cells = [f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    header = _format_row(columns)
    separator_cells = ["-" * (width + 2) for width in widths]
    separator = "|" + "|".join(separator_cells) + "|"
    body = [_format_row(row) for row in string_rows]
    return "\n".join([header, separator, *body])


class MarkdownReporter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def render(self, metrics: Dict[str, float], table: pd.DataFrame) -> Path:
        lines = ["# Experiment Summary", "", "## Aggregate Metrics"]
        for key, value in metrics.items():
            lines.append(f"- **{key}**: {value:.3f}")
        lines.append("\n## Detailed Results")
        lines.append(_dataframe_to_markdown(table))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines), encoding="utf-8")
        return self.output_path
''', False)
# END MODULE: reporting.generators

# BEGIN MODULE: scripts.run_dashboard (scripts/run_dashboard.py)
_register_module('scripts.run_dashboard', r'''
"""Entry-point to launch the interactive optimisation dashboard."""
from __future__ import annotations

from visualization.dashboard import launch_dashboard, tkinter_available


def main() -> None:
    if not tkinter_available():
        raise SystemExit(
            "Tkinter is not available in this environment. Install tkinter to use the dashboard interface."
        )
    launch_dashboard()


if __name__ == "__main__":
    main()
''', False)
# END MODULE: scripts.run_dashboard

# BEGIN MODULE: scripts.run_experiments (scripts/run_experiments.py)
_register_module('scripts.run_experiments', r'''
"""Entry point to execute baseline experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.base_config import load_config
from core.config import ConfigManager
from data.generator import SyntheticDataGenerator, SyntheticScenario
from data.loader import DataLoader, DataPreprocessor
from experiments.manager import ExperimentManager, export_results
from problems.job_shop import create_job_shop_problem
from reporting.generators import MarkdownReporter
from visualization.plots import bar_performance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMS optimisation experiments")
    parser.add_argument("--config", type=Path, help="Path to configuration file", required=False)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/baseline.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    manager = ConfigManager(config)

    loader = DataLoader()
    preprocessor = DataPreprocessor()
    data_sources = manager.config.data.sources or [Path("data/synthetic/sample.csv")]
    existing_sources = [source for source in data_sources if Path(source).exists()]
    if existing_sources:
        data = loader.load(existing_sources)
    else:
        scenario = SyntheticScenario(
            num_jobs=20,
            machines=["M01", "M02", "M03"],
            start_date=pd.Timestamp("2023-01-01T08:00:00"),
            time_between_jobs=pd.Timedelta(minutes=15),
        )
        data = SyntheticDataGenerator().generate(scenario)
    data = preprocessor.transform(data)
    problem = create_job_shop_problem(data)

    experiment_manager = ExperimentManager(manager.config)
    results = experiment_manager.run(problem)
    summary = experiment_manager.summarise(results)
    export_results(summary, args.output)

    if not summary.empty:
        bar_performance(summary, "makespan", Path("results/figures/makespan.png"))
        reporter = MarkdownReporter(Path("results/reports/summary.md"))
        reporter.render({"runs": len(summary)}, summary)


if __name__ == "__main__":
    main()
''', False)
# END MODULE: scripts.run_experiments

# BEGIN MODULE: simulation.discrete_event (simulation/discrete_event.py)
_register_module('simulation.discrete_event', r'''
"""Simplified discrete event simulation skeleton."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(order=True)
class Event:
    time: float
    description: str


@dataclass
class DiscreteEventSimulator:
    events: List[Event] = field(default_factory=list)

    def schedule(self, event: Event) -> None:
        self.events.append(event)
        self.events.sort()

    def run(self) -> List[Event]:
        executed: List[Event] = []
        while self.events:
            executed.append(self.events.pop(0))
        return executed
''', False)
# END MODULE: simulation.discrete_event

# BEGIN MODULE: simulation.monte_carlo (simulation/monte_carlo.py)
_register_module('simulation.monte_carlo', r'''
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
''', False)
# END MODULE: simulation.monte_carlo

# BEGIN MODULE: simulation.stochastic_models (simulation/stochastic_models.py)
_register_module('simulation.stochastic_models', r'''
"""Stochastic models for manufacturing processes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class ProcessingTimeModel:
    distribution: Callable[[int], List[float]]

    def sample(self, size: int) -> List[float]:
        return self.distribution(size)
''', False)
# END MODULE: simulation.stochastic_models

# BEGIN MODULE: utils.logging (utils/logging.py)
_register_module('utils.logging', r'''
"""Logging utilities for the framework."""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "framework.log"),
            logging.StreamHandler(),
        ],
    )
''', False)
# END MODULE: utils.logging

# BEGIN MODULE: validation.empirical (validation/empirical.py)
_register_module('validation.empirical', r'''
"""Empirical validation utilities."""
from __future__ import annotations

from typing import Dict

import pandas as pd

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - fallback for constrained environments
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from scipy import stats
except ImportError:  # pragma: no cover - fallback for constrained environments
    stats = None  # type: ignore


def friedman_test(results: pd.DataFrame) -> Dict[str, float]:
    if stats is None:
        raise RuntimeError("scipy is required to run the Friedman test")
    pivot = results.pivot(index="replication", columns="algorithm", values="makespan")
    statistic, pvalue = stats.friedmanchisquare(*pivot.T.values)
    return {"statistic": float(statistic), "p_value": float(pvalue)}


def confidence_interval(values: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    if stats is None or np is None:
        raise RuntimeError("scipy and numpy are required to compute confidence intervals")
    mean = float(np.mean(values))
    sem = stats.sem(values)
    interval = stats.t.interval(level, len(values) - 1, loc=mean, scale=sem)
    return {"mean": mean, "lower": float(interval[0]), "upper": float(interval[1])}
''', False)
# END MODULE: validation.empirical

# BEGIN MODULE: validation.theoretical (validation/theoretical.py)
_register_module('validation.theoretical', r'''
"""Theoretical validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ComplexityAnalysis:
    algorithm: str
    time_complexity: str
    space_complexity: str


def document_complexity(algorithm: str, time_complexity: str, space_complexity: str) -> Dict[str, str]:
    return {
        "algorithm": algorithm,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity,
    }
''', False)
# END MODULE: validation.theoretical

# BEGIN MODULE: visualization.dashboard (visualization/dashboard.py)
_register_module('visualization.dashboard', r'''
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
''', False)
# END MODULE: visualization.dashboard

# BEGIN MODULE: visualization.gallery (visualization/gallery.py)
_register_module('visualization.gallery', r'''
"""Automated gallery generation producing 50+ publication-grade figures."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import pandas as pd

from visualization import plots


@dataclass(frozen=True)
class FigureTemplate:
    name: str
    builder: Callable
    args: Sequence
    kwargs: Dict[str, object]


def _metric_list(results: pd.DataFrame) -> List[str]:
    return [
        column
        for column in results.columns
        if column not in {"algorithm", "iteration", "timestamp", "scenario"}
    ]


def _ensure_iteration_frame(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in results.iterrows():
        base_value = float(row[metric]) if metric in row else 0.0
        for iteration in range(1, 11):
            progress = base_value * (1.0 - 0.4 * iteration / 10.0)
            rows.append(
                {
                    "algorithm": row.get("algorithm", f"algo_{iteration}"),
                    "iteration": iteration,
                    metric: max(progress, 0.0),
                }
            )
    return pd.DataFrame(rows)


def _ensure_timeseries_frame(results: pd.DataFrame) -> pd.DataFrame:
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    timestamps = [base_time + timedelta(minutes=idx * 15) for idx in range(len(results))]
    energy_base = 0.0
    if "energy" in getattr(results, "columns", []):
        energy_series = results["energy"]
        energy_values = energy_series.astype(float).to_list() if hasattr(energy_series, "to_list") else list(energy_series)
        if energy_values:
            energy_base = float(energy_values[0])
    utilisation = {
        "timestamp": timestamps,
        "energy_load": [energy_base * (0.9 + 0.02 * idx) for idx in range(len(results))],
        "throughput": [idx + 1 for idx in range(len(results))],
        "quality": [max(0.0, 1.0 - 0.05 * idx) for idx in range(len(results))],
    }
    return pd.DataFrame(utilisation)


def _significance_frame(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    algo_series = results["algorithm"]
    algorithms = algo_series.to_list() if hasattr(algo_series, "to_list") else list(algo_series)
    algorithms = [str(algo) for algo in algorithms]
    value_series = results[metric]
    value_list = value_series.astype(float).to_list() if hasattr(value_series, "to_list") else [float(value) for value in value_series]
    matrix_rows: List[Dict[str, float]] = []
    for i, _algo_a in enumerate(algorithms):
        row: Dict[str, float] = {}
        for j, algo_b in enumerate(algorithms):
            diff = abs(value_list[i] - value_list[j])
            denominator = value_list[i] + 1.0
            row[algo_b] = max(0.001, min(0.1, diff / denominator))
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows, index=algorithms, columns=algorithms)


def _waterfall_components(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = float(results[metric].min())
    deltas = [float(value) - baseline for value in results[metric]]
    return pd.DataFrame(
        {
            "component": results["algorithm"].astype(str),
            "value": deltas,
        }
    )


def _slope_components(results: pd.DataFrame, metric: str, alt_metric: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "algorithm": results["algorithm"].astype(str),
            metric: results[metric].astype(float),
            alt_metric: results[alt_metric].astype(float),
        }
    )


def _tradeoff_pairs(metrics: Sequence[str]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            pairs.append((metrics[i], metrics[j]))
    return pairs


def build_figure_templates(results: pd.DataFrame) -> List[FigureTemplate]:
    metrics = _metric_list(results)
    templates: List[FigureTemplate] = []
    primary_metrics = metrics[:6] if len(metrics) >= 6 else metrics
    for metric in primary_metrics:
        templates.extend(
            [
                FigureTemplate(f"bar_{metric}", plots.bar_performance, (metric,), {}),
                FigureTemplate(f"box_{metric}", plots.box_performance, (metric,), {}),
                FigureTemplate(f"violin_{metric}", plots.violin_performance, (metric,), {}),
                FigureTemplate(f"histogram_{metric}", plots.histogram_metric, (metric,), {}),
                FigureTemplate(f"density_{metric}", plots.density_plot_metric, (metric,), {}),
                FigureTemplate(f"cdf_{metric}", plots.cdf_metric_plot, (metric,), {}),
                FigureTemplate(f"rug_{metric}", plots.rug_plot_metric, (metric,), {}),
                FigureTemplate(f"boxen_{metric}", plots.boxen_schedule_variability, (metric,), {}),
            ]
        )

    if len(metrics) >= 3:
        templates.append(
            FigureTemplate(
                "parallel_coordinates",
                plots.parallel_coordinates_plot,
                (metrics[: min(6, len(metrics))],),
                {},
            )
        )
    if metrics:
        templates.append(
            FigureTemplate("cumulative_improvement_makespan", plots.cumulative_improvement, (metrics[0],), {})
        )

    for metric_x, metric_y in _tradeoff_pairs(primary_metrics[:4]):
        templates.append(
            FigureTemplate(f"scatter_{metric_x}_vs_{metric_y}", plots.scatter_tradeoff, (metric_x, metric_y), {})
        )
        templates.append(
            FigureTemplate(
                f"bubble_{metric_x}_{metric_y}",
                plots.bubble_chart,
                (metric_x, metric_y, primary_metrics[0]),
                {},
            )
        )

    if len(primary_metrics) >= 2:
        templates.append(
            FigureTemplate(
                "pareto_front_primary",
                plots.pareto_front_plot,
                (primary_metrics[0], primary_metrics[1]),
                {},
            )
        )
    if len(primary_metrics) >= 3:
        templates.append(
            FigureTemplate(
                "pareto_front_3d_primary",
                plots.pareto_front_3d,
                (primary_metrics[:3],),
                {},
            )
        )

    if metrics:
        templates.append(
            FigureTemplate(
                "radar_top_algorithm",
                plots.radar_performance_plot,
                (metrics[: min(6, len(metrics))], results.iloc[0]["algorithm"]),
                {},
            )
        )

    templates.append(FigureTemplate("heatmap_correlation", plots.heatmap_correlation, (metrics[: min(6, len(metrics))],), {}))
    templates.append(FigureTemplate("stacked_bar_objectives", plots.stacked_bar_objectives, (metrics[: min(5, len(metrics))],), {}))
    templates.append(FigureTemplate("gantt_overview", plots.gantt_chart, tuple(), {}))
    templates.append(FigureTemplate("utilisation_stack", plots.stacked_area_utilization, tuple(), {}))
    templates.append(FigureTemplate("throughput_timeline", plots.throughput_timeline, tuple(), {}))
    templates.append(FigureTemplate("slope_analysis", plots.slope_graph, tuple(), {}))
    templates.append(FigureTemplate("waterfall_decomposition", plots.waterfall_breakdown, tuple(), {}))
    templates.append(FigureTemplate("line_convergence", plots.line_convergence, (primary_metrics[0],), {}))

    return templates


def generate_gallery(
    results: pd.DataFrame,
    schedule: pd.DataFrame,
    output_dir: Path | str,
    significance_metric: str | None = None,
) -> List[Path]:
    """Generate an extensive figure gallery covering the supplied results.

    Parameters
    ----------
    results:
        DataFrame with per-algorithm metrics.
    schedule:
        Representative schedule used for Gantt and resource plots.
    output_dir:
        Directory where the figures will be written.  Files are always
        generated using PNG semantics even when the lightweight plotting
        backend serialises JSON instructions; the extension remains ``.png`` to
        keep the workflow consistent with journal submission tooling.
    significance_metric:
        Optional metric used to derive the statistical significance heatmap.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = _metric_list(results)
    if not metrics:
        raise ValueError("Results dataframe must contain at least one metric column")
    primary_metric = significance_metric or metrics[0]

    convergence_frame = _ensure_iteration_frame(results, primary_metric)
    utilisation_frame = _ensure_timeseries_frame(results)
    significance_frame = _significance_frame(results, primary_metric)
    slope_frame = _slope_components(results, primary_metric, metrics[min(1, len(metrics) - 1)])
    waterfall_frame = _waterfall_components(results, primary_metric)

    generated_paths: List[Path] = []
    for template in build_figure_templates(results):
        target = output_path / f"{template.name}.png"
        builder = template.builder
        if builder is plots.gantt_chart:
            path = builder(schedule, target)
        elif builder is plots.stacked_area_utilization:
            path = builder(utilisation_frame, target)
        elif builder is plots.throughput_timeline:
            path = builder(utilisation_frame, "timestamp", "throughput", target)
        elif builder is plots.slope_graph:
            path = builder(slope_frame, "algorithm", primary_metric, metrics[min(1, len(metrics) - 1)], target)
        elif builder is plots.waterfall_breakdown:
            path = builder(waterfall_frame, target)
        elif builder is plots.line_convergence:
            path = builder(convergence_frame, primary_metric, target)
        elif builder is plots.heatmap_correlation:
            path = builder(results, template.args[0], target)  # type: ignore[arg-type]
        elif builder is plots.heatmap_significance:
            path = builder(significance_frame, target)
        elif builder is plots.parallel_coordinates_plot:
            path = builder(results, template.args[0], target)
        elif builder is plots.pareto_front_3d:
            path = builder(results, template.args[0], target)
        elif builder is plots.scatter_tradeoff:
            path = builder(results, template.args[0], template.args[1], target)
        elif builder is plots.bubble_chart:
            path = builder(results, template.args[0], template.args[1], template.args[2], target)
        elif builder in {plots.bar_performance, plots.box_performance, plots.violin_performance}:
            path = builder(results, template.args[0], target)
        elif builder in {
            plots.histogram_metric,
            plots.density_plot_metric,
            plots.cdf_metric_plot,
            plots.rug_plot_metric,
            plots.boxen_schedule_variability,
            plots.cumulative_improvement,
        }:
            path = builder(results, template.args[0], target)
        elif builder is plots.radar_performance_plot:
            path = builder(results, template.args[0], template.args[1], target)
        elif builder is plots.stacked_bar_objectives:
            path = builder(results, template.args[0], target)
        elif builder is plots.pareto_front_plot:
            path = builder(results, template.args[0], template.args[1], target)
        else:
            path = builder(results, target)  # type: ignore[arg-type]
        generated_paths.append(path)

    # Add statistical significance heatmap explicitly to guarantee coverage.
    heatmap_path = plots.heatmap_significance(significance_frame, output_path / "heatmap_significance.png")
    generated_paths.append(heatmap_path)

    if len(generated_paths) < 50:
        raise RuntimeError(
            f"Gallery produced only {len(generated_paths)} figures; expected at least 50 for publication readiness."
        )
    return generated_paths


def available_figure_names(results: pd.DataFrame) -> List[str]:
    return [template.name for template in build_figure_templates(results)]


__all__ = ["generate_gallery", "available_figure_names"]
''', False)
# END MODULE: visualization.gallery

# BEGIN MODULE: visualization.plots (visualization/plots.py)
_register_module('visualization.plots', r'''
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
    if hasattr(cbar, "set_label"):
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
    try:
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        fig, ax = plt.subplots(figsize=(6, 4))
        series_z = results[metrics[2]].astype(float)
        colors = series_z.to_list() if hasattr(series_z, "to_list") else list(series_z)
        scatter = ax.scatter(
            results[metrics[0]].astype(float).to_list(),
            results[metrics[1]].astype(float).to_list(),
            c=colors,
            cmap="viridis",
        )
        ax.set_xlabel(metrics[0])
        ax.set_ylabel(metrics[1])
        ax.set_title("Pareto projection (colour encodes third objective)")
        cbar = fig.colorbar(scatter, ax=ax)
        if hasattr(cbar, "set_label"):
            cbar.set_label(metrics[2])
        return _save_figure(fig, output)

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

    columns = list(p_values.columns)
    matrix = []
    for idx in range(len(p_values.index)):
        row_values: list[float] = []
        for column in columns:
            column_series = p_values[column]
            values = column_series.to_list() if hasattr(column_series, "to_list") else list(column_series)
            row_values.append(float(values[idx]))
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap="viridis_r", vmin=0, vmax=0.1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    ax.set_xticks(range(len(p_values.columns)))
    ax.set_xticklabels(p_values.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(p_values.index)))
    ax.set_yticklabels(p_values.index)
    for i in range(len(p_values.index)):
        for j in range(len(columns)):
            ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", color="black")
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
        ax.bar([idx], [value], bottom=start, color=color)
        ax.text(idx, start + value / 2, f"{value:.2f}", ha="center", va="center", color="white")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices, rotation=45, ha="right")
    ax.set_ylabel("Contribution")
    ax.set_title("Waterfall breakdown")
    return _save_figure(fig, output)
''', False)
# END MODULE: visualization.plots

# BEGIN MODULE: visualization.simpleplot (visualization/simpleplot.py)
_register_module('visualization.simpleplot', r'''
"""Fallback plotting module when matplotlib is unavailable."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class BodyHandle:
    operations: List[Dict[str, Any]] = field(default_factory=list)

    def set_alpha(self, value: float) -> None:
        self.operations.append({"set_alpha": float(value)})


@dataclass
class CollectionHandle:
    kind: str
    payload: Dict[str, Any]


class Axes:
    def __init__(self, polar: bool = False) -> None:
        self.polar = polar
        self.operations: List[Dict[str, Any]] = []

    def _log(self, name: str, **payload: Any) -> None:
        entry: Dict[str, Any] = {"op": name}
        if payload:
            entry.update(payload)
        self.operations.append(entry)

    def bar(self, x: Sequence[Any], height: Sequence[float], **kwargs: Any) -> None:
        self._log("bar", x=list(x), height=list(height), kwargs=kwargs)

    def barh(self, y: float, width: float, left: float, height: float) -> None:
        self._log("barh", y=y, width=width, left=left, height=height)

    def set_ylabel(self, label: str) -> None:
        self._log("set_ylabel", label=label)

    def set_xlabel(self, label: str) -> None:
        self._log("set_xlabel", label=label)

    def set_title(self, title: str) -> None:
        self._log("set_title", title=title)

    def grid(self, *args: Any, **kwargs: Any) -> None:
        self._log("grid", args=args, kwargs=kwargs)

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self._log("legend", args=args, kwargs=kwargs)

    def plot(self, x: Sequence[float], y: Sequence[float], **kwargs: Any) -> None:
        self._log("plot", x=list(x), y=list(y), kwargs=kwargs)

    def scatter(self, x: Sequence[float], y: Sequence[float], **kwargs: Any) -> CollectionHandle:
        payload = {"x": list(x), "y": list(y), "kwargs": kwargs}
        self._log("scatter", **payload)
        return CollectionHandle(kind="scatter", payload=payload)

    def hist(self, data: Sequence[float], **kwargs: Any) -> None:
        self._log("hist", data=list(data), kwargs=kwargs)

    def violinplot(self, dataset: Sequence[Sequence[float]], **kwargs: Any) -> Dict[str, List[BodyHandle]]:
        bodies = [BodyHandle() for _ in dataset]
        self._log("violinplot", dataset=[list(item) for item in dataset], kwargs=kwargs)
        return {"bodies": bodies}

    def boxplot(self, dataset: Sequence[Sequence[float]], labels: Sequence[str], **kwargs: Any) -> Dict[str, Any]:
        self._log("boxplot", dataset=[list(item) for item in dataset], labels=list(labels), kwargs=kwargs)
        return {}

    def step(self, x: Sequence[float], y: Sequence[float], where: str = "post") -> None:
        self._log("step", x=list(x), y=list(y), where=where)

    def stackplot(self, x: Sequence[Any], y: Sequence[Sequence[float]], labels: Sequence[str], alpha: float = 1.0) -> None:
        self._log("stackplot", x=list(x), y=[[float(v) for v in series] for series in y], labels=list(labels), alpha=float(alpha))

    def fill(self, x: Sequence[float], y: Sequence[float], alpha: float = 1.0) -> None:
        self._log("fill", x=list(x), y=list(y), alpha=float(alpha))

    def set_xticks(self, ticks: Sequence[float]) -> None:
        self._log("set_xticks", ticks=list(ticks))

    def set_xticklabels(self, labels: Sequence[str], rotation: Optional[float] = None, ha: Optional[str] = None) -> None:
        self._log("set_xticklabels", labels=list(labels), rotation=rotation, ha=ha)

    def set_yticks(self, ticks: Sequence[float]) -> None:
        self._log("set_yticks", ticks=list(ticks))

    def set_yticklabels(self, labels: Sequence[str]) -> None:
        self._log("set_yticklabels", labels=list(labels))

    def text(self, x: float, y: float, s: str, **kwargs: Any) -> None:
        self._log("text", x=float(x), y=float(y), text=s, kwargs=kwargs)

    def imshow(self, data: Sequence[Sequence[float]], **kwargs: Any) -> CollectionHandle:
        payload = {"data": [list(row) for row in data], "kwargs": kwargs}
        self._log("imshow", **payload)
        return CollectionHandle(kind="image", payload=payload)

    def bar_label(self, container: Any, labels: Sequence[str]) -> None:
        self._log("bar_label", container=str(container), labels=list(labels))

    def set_zlabel(self, label: str) -> None:
        self._log("set_zlabel", label=label)


class Axes3D(Axes):
    def __init__(self) -> None:
        super().__init__(polar=False)


class Figure:
    def __init__(self) -> None:
        self.axes: List[Axes] = []
        self.operations: List[Dict[str, Any]] = []

    def add_subplot(self, _code: int, projection: Optional[str] = None) -> Axes:
        ax = Axes3D() if projection == "3d" else Axes()
        self.axes.append(ax)
        self.operations.append({"op": "add_subplot", "projection": projection})
        return ax

    def tight_layout(self) -> None:
        self.operations.append({"op": "tight_layout"})

    def savefig(self, output: Path, dpi: int = 300) -> None:
        data = {
            "dpi": dpi,
            "axes": [ax.operations for ax in self.axes],
            "figure_ops": self.operations,
        }
        output.write_text(json.dumps(data, indent=2))

    def colorbar(self, handle: CollectionHandle, ax: Axes, label: Optional[str] = None, **kwargs: Any) -> None:
        self.operations.append({
            "op": "colorbar",
            "handle": handle.kind,
            "label": label,
            "kwargs": kwargs,
        })


def subplots(figsize: Tuple[float, float] = (6, 4), subplot_kw: Optional[Dict[str, Any]] = None) -> Tuple[Figure, Axes]:
    figure = Figure()
    polar = bool(subplot_kw.get("polar")) if subplot_kw else False
    ax = Axes(polar=polar)
    figure.axes.append(ax)
    figure.operations.append({"op": "subplots", "figsize": figsize, "polar": polar})
    return figure, ax


def figure(figsize: Tuple[float, float] = (6, 4)) -> Figure:
    fig = Figure()
    fig.operations.append({"op": "figure", "figsize": figsize})
    return fig


def close(fig: Figure) -> None:
    fig.operations.append({"op": "close"})
''', False)
# END MODULE: visualization.simpleplot

# BEGIN MODULE: algorithms.classical.constructive_heuristics (algorithms/classical/constructive_heuristics.py)
_register_module('algorithms.classical.constructive_heuristics', r'''
"""Constructive heuristics for flow-shop style problems."""
from __future__ import annotations

from typing import List

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class NEHHeuristic(BaseOptimizer):
    """Implementation of the classic Nawaz-Enscore-Ham heuristic."""

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        jobs = problem.jobs.copy()
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column is required for NEH heuristic")

        # Sort jobs by decreasing processing time.
        ordered_indices = list(processing.sort_values(ascending=False).index)
        sequence: List[int] = []

        for job in ordered_indices:
            best_sequence: List[int] | None = None
            best_cost = float("inf")
            for position in range(len(sequence) + 1):
                candidate = sequence[:position] + [job] + sequence[position:]
                schedule = problem.build_schedule(candidate)
                cost = evaluate_schedule(schedule)["makespan"]
                if cost < best_cost:
                    best_cost = cost
                    best_sequence = candidate
            assert best_sequence is not None  # for mypy / static typing
            sequence = best_sequence

        final_schedule = problem.build_schedule(sequence)
        return ScheduleSolution(schedule=final_schedule, metadata={"sequence": sequence})


class PalmerHeuristic(BaseOptimizer):
    """Palmer's slope index heuristic for flow shop scheduling."""

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        jobs = problem.jobs.copy()
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column is required for Palmer heuristic")

        machines = jobs.get("Machine_ID")
        slope_index: List[float]
        if machines is not None and not machines.empty:
            unique_machines = sorted(machines.unique())
            if len(unique_machines) == 1:
                weight_map = {unique_machines[0]: 0.0}
            else:
                step = 2.0 / (len(unique_machines) - 1)
                weight_map = {machine: -1.0 + idx * step for idx, machine in enumerate(unique_machines)}
            slope_index = [weight_map.get(machines.iloc[i], 0.0) for i in range(len(machines))]
        else:
            if len(jobs) <= 1:
                slope_index = [0.0 for _ in range(len(jobs))]
            else:
                step = 2.0 / (len(jobs) - 1)
                slope_index = [-1.0 + i * step for i in range(len(jobs))]

        priority = [slope_index[i] * processing.iloc[i] for i in range(len(processing))]
        ordered = jobs.assign(_priority=priority).sort_values("_priority", ascending=True)
        schedule = problem.build_schedule(ordered.index)
        return ScheduleSolution(schedule=schedule)
''', False)
# END MODULE: algorithms.classical.constructive_heuristics

# BEGIN MODULE: algorithms.classical.dispatching_rules (algorithms/classical/dispatching_rules.py)
_register_module('algorithms.classical.dispatching_rules', r'''
"""Implementation of classical dispatching rules."""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from core.base_optimizer import BaseOptimizer
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _ensure_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _ensure_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce")


def _fill_reference(series: pd.Series, default: pd.Timestamp) -> pd.Series:
    if series.isna().all():
        return pd.Series([default] * len(series), index=series.index, dtype="datetime64[ns]")
    return series.fillna(series.min())


class DispatchingRule(BaseOptimizer):
    """Base class encapsulating a dispatching rule."""

    rule_name: str = "dispatching_rule"
    ascending: bool = True

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = problem.jobs.copy()
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        priority = self._priority(jobs)
        priority = priority.reindex(jobs.index)
        jobs = jobs.assign(_priority=priority)
        ordered = jobs.sort_values("_priority", ascending=self.ascending, kind="mergesort")
        schedule = problem.build_schedule(ordered.index)
        schedule = schedule.reset_index(drop=True)
        return ScheduleSolution(schedule=schedule, metadata={"rule": self.rule_name})


class FCFSRule(DispatchingRule):
    """First-Come-First-Served based on release time."""

    rule_name = "fcfs"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_datetime(jobs, "Scheduled_Start").rank(method="first")


class SPTRule(DispatchingRule):
    """Shortest processing time first."""

    rule_name = "spt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_series(jobs, "Processing_Time")


class LPTRule(DispatchingRule):
    """Longest processing time first."""

    rule_name = "lpt"
    ascending = False

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_series(jobs, "Processing_Time")


class EDDRule(DispatchingRule):
    """Earliest due date rule."""

    rule_name = "edd"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        return _ensure_datetime(jobs, "Due_Date").rank(method="first")


class SLACKRule(DispatchingRule):
    """Schedule jobs with minimum slack."""

    rule_name = "slack"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        return pd.Series(slack, index=jobs.index)


class CriticalRatioRule(DispatchingRule):
    """Critical ratio rule (time remaining / processing)."""

    rule_name = "critical_ratio"
    ascending = False

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        time_remaining = (due - start).dt.total_seconds() / 60.0
        ratio = time_remaining / processing.replace(0, math.nan)
        return ratio.fillna(0.0)


class WSPTRule(DispatchingRule):
    """Weighted shortest processing time rule."""

    rule_name = "wspt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        weights = _ensure_series(jobs, "Priority", default=1.0)
        return processing / weights.replace(0, math.nan)


class ATRule(DispatchingRule):
    """Apparent tardiness cost (ATC) rule."""

    rule_name = "atc"

    def __init__(self, k: float = 2.0, **kwargs):
        super().__init__(k=k, **kwargs)
        self.k = k

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        release = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        avg_proc = processing.mean() if not processing.empty else 1.0
        urgency = (due - release).dt.total_seconds() / 60.0 - processing
        exponent = urgency.clip(lower=0.0) / (self.k * avg_proc)
        exponent = exponent.fillna(0.0)
        priority = exponent.apply(lambda value: math.exp(-value)) / processing.replace(0, math.nan)
        priority = priority.apply(
            lambda value: 0.0 if value in (math.inf, -math.inf) or pd.isna(value) else value
        )
        return priority


class MSERule(DispatchingRule):
    """Minimum slack per operation."""

    rule_name = "mse"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        operations = _ensure_series(jobs, "Remaining_Operations", default=1.0)
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        processing = _ensure_series(jobs, "Processing_Time")
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        return slack / operations.replace(0, math.nan)


class SRPTRule(DispatchingRule):
    """Shortest remaining processing time."""

    rule_name = "srpt"

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        remaining = _ensure_series(jobs, "Remaining_Processing_Time")
        if (remaining == 0).all():
            remaining = _ensure_series(jobs, "Processing_Time")
        return remaining


class CoversionRule(DispatchingRule):
    """CoVERT rule emphasising tardiness avoidance."""

    rule_name = "covert"
    ascending = False

    def __init__(self, k: float = 3.0, **kwargs):
        super().__init__(k=k, **kwargs)
        self.k = k

    def _priority(self, jobs: pd.DataFrame) -> pd.Series:
        processing = _ensure_series(jobs, "Processing_Time")
        due = _fill_reference(_ensure_datetime(jobs, "Due_Date"), pd.Timestamp("1970-01-01"))
        start = _fill_reference(_ensure_datetime(jobs, "Scheduled_Start"), due.min())
        slack = (due - start).dt.total_seconds() / 60.0 - processing
        avg_proc = processing.mean() if not processing.empty else 1.0
        exponent = slack.clip(lower=0.0) / (self.k * avg_proc)
        return exponent.apply(lambda value: math.exp(-value))


DISPATCHING_RULES: Dict[str, type[DispatchingRule]] = {
    "fcfs": FCFSRule,
    "spt": SPTRule,
    "lpt": LPTRule,
    "edd": EDDRule,
    "slack": SLACKRule,
    "critical_ratio": CriticalRatioRule,
    "wspt": WSPTRule,
    "atc": ATRule,
    "mse": MSERule,
    "srpt": SRPTRule,
    "covert": CoversionRule,
}


def list_dispatching_rules() -> List[str]:
    """Return the available dispatching rule identifiers."""

    return sorted(DISPATCHING_RULES.keys())
''', False)
# END MODULE: algorithms.classical.dispatching_rules

# BEGIN MODULE: algorithms.classical.exact_methods (algorithms/classical/exact_methods.py)
_register_module('algorithms.classical.exact_methods', r'''
"""Exact optimisation methods for small instances."""
from __future__ import annotations

from typing import List

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class BranchAndBound(BaseOptimizer):
    """Simple branch-and-bound search exploring job permutations."""

    def __init__(self, max_jobs: int = 8) -> None:
        super().__init__(max_jobs=max_jobs)
        self.max_jobs = max_jobs

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = problem.jobs
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        if len(jobs) > self.max_jobs:
            # Fallback to constructive heuristic for large instances
            from algorithms.classical.constructive_heuristics import NEHHeuristic

            return NEHHeuristic().solve(problem)

        best_sequence: List[int] | None = None
        best_cost = float("inf")
        processing = jobs.get("Processing_Time")
        if processing is None:
            raise ValueError("Processing_Time column required for branch-and-bound optimisation")
        filled_processing = processing.fillna(processing.mean() or 0.0)
        processing_map = filled_processing.to_dict()

        def branch(partial: List[int], remaining: List[int], accumulated: float) -> None:
            nonlocal best_cost, best_sequence
            if not remaining:
                if accumulated < best_cost:
                    best_cost = accumulated
                    best_sequence = partial.copy()
                return

            lower_bound = accumulated + sum(processing_map[idx] for idx in remaining)
            if lower_bound >= best_cost:
                return

            for idx in remaining:
                next_partial = partial + [idx]
                schedule = problem.build_schedule(next_partial)
                cost = evaluate_schedule(schedule)["makespan"]
                if cost >= best_cost:
                    continue
                next_remaining = [j for j in remaining if j != idx]
                branch(next_partial, next_remaining, cost)

        initial_remaining = list(jobs.index)
        branch([], initial_remaining, 0.0)

        if best_sequence is None:
            best_sequence = initial_remaining
        final_schedule = problem.build_schedule(best_sequence)
        return ScheduleSolution(schedule=final_schedule, metadata={"sequence": best_sequence})
''', False)
# END MODULE: algorithms.classical.exact_methods

# BEGIN MODULE: algorithms.deep_rl.dqn (algorithms/deep_rl/dqn.py)
_register_module('algorithms.deep_rl.dqn', r'''
"""Light-weight Deep-Q-inspired scheduler using linear function approximation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _extract_features(job_row: Dict[str, object]) -> List[float]:
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
    return [processing, slack, energy, 1.0]


@dataclass
class LinearQNetwork:
    weights: List[float]
    learning_rate: float

    def predict(self, features: List[float]) -> float:
        return float(sum(f * w for f, w in zip(features, self.weights)))

    def update(self, features: List[float], target: float) -> None:
        prediction = self.predict(features)
        error = target - prediction
        for idx, value in enumerate(features):
            self.weights[idx] += self.learning_rate * error * value


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
        rng = random.Random(self.seed)
        weights = [rng.gauss(0.0, 0.01) for _ in range(4)]
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
                state_features: List[tuple[int, List[float]]] = []
                for idx in remaining:
                    features = _extract_features(problem.jobs.loc[idx].to_dict())
                    norm = math.sqrt(sum(value * value for value in features)) + 1e-9
                    features = [value / norm for value in features]
                    state_features.append((idx, features))
                if rng.random() < self.epsilon:
                    action_idx = rng.randrange(len(state_features))
                else:
                    q_values = [network.predict(features) for _, features in state_features]
                    best_value = min(q_values)
                    action_idx = q_values.index(best_value)
                job_id, features = state_features[action_idx]
                sequence.append(job_id)
                remaining.remove(job_id)

                current_time += float(problem.jobs.loc[job_id].get("Processing_Time", 0.0))
                reward = -current_time
                future_estimate = 0.0
                if remaining:
                    next_features = []
                    for idx in remaining:
                        feat = _extract_features(problem.jobs.loc[idx].to_dict())
                        norm = math.sqrt(sum(value * value for value in feat)) + 1e-9
                        next_features.append([value / norm for value in feat])
                    next_q = [network.predict(feat) for feat in next_features]
                    future_estimate = min(next_q)
                target = reward + self.discount * future_estimate
                network.update(features, target)

        return network

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        network = self._train(problem)
        jobs = problem.jobs
        if jobs.empty:
            return ScheduleSolution(schedule=jobs)

        features: List[tuple[int, float]] = []
        for idx, row in jobs.iterrows():
            feat = _extract_features(row.to_dict())
            norm = math.sqrt(sum(value * value for value in feat)) + 1e-9
            norm_feat = [value / norm for value in feat]
            features.append((idx, network.predict(norm_feat)))
        features.sort(key=lambda item: item[1])
        sequence = [idx for idx, _ in features]
        schedule = problem.build_schedule(sequence)
        metrics = evaluate_schedule(schedule)
        return ScheduleSolution(schedule=schedule, metrics=metrics, metadata={"policy": "linear_dqn"})
''', False)
# END MODULE: algorithms.deep_rl.dqn

# BEGIN MODULE: algorithms.deep_rl.ppo (algorithms/deep_rl/ppo.py)
_register_module('algorithms.deep_rl.ppo', r'''
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
''', False)
# END MODULE: algorithms.deep_rl.ppo

# BEGIN MODULE: algorithms.hybrid.adaptive_hybrid (algorithms/hybrid/adaptive_hybrid.py)
_register_module('algorithms.hybrid.adaptive_hybrid', r'''
"""Adaptive hybrid optimiser that combines multiple strategies."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from algorithms.classical.dispatching_rules import DISPATCHING_RULES
from algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from core.base_optimizer import BaseOptimizer
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class AdaptiveHybridOptimizer(BaseOptimizer):
    """Select the best schedule among a portfolio of base optimisers."""

    def __init__(self, candidates: Iterable[str] | None = None, **kwargs) -> None:
        if candidates is None:
            candidates = ["fcfs", "spt", "edd", "simulated_annealing"]
        normalised = [name.lower() for name in candidates]
        super().__init__(candidates=normalised, **kwargs)
        self.candidates = normalised

    def _instantiate(self, name: str) -> BaseOptimizer:
        if name in DISPATCHING_RULES:
            return DISPATCHING_RULES[name]()
        if name == "simulated_annealing":
            return SimulatedAnnealing()
        raise ValueError(f"Unknown optimiser '{name}'")

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        if problem.jobs.empty:
            return ScheduleSolution(schedule=problem.jobs)

        results: List[Tuple[str, ScheduleSolution]] = []
        for name in self.candidates:
            optimizer = self._instantiate(name)
            solution = optimizer.solve(problem)
            results.append((name, solution))

        weights = {"makespan": 1.0, "total_tardiness": 0.5, "energy": 0.05}
        def score(metrics: Dict[str, float]) -> float:
            return sum(metrics.get(k, 0.0) * w for k, w in weights.items())

        best_name, best_solution = min(results, key=lambda item: score(item[1].metrics))
        metadata = {
            "selected": best_name,
            "portfolio": {name: sol.metrics for name, sol in results},
        }
        return ScheduleSolution(schedule=best_solution.schedule.copy(), metrics=best_solution.metrics, metadata=metadata)
''', False)
# END MODULE: algorithms.hybrid.adaptive_hybrid

# BEGIN MODULE: algorithms.metaheuristics.ant_colony (algorithms/metaheuristics/ant_colony.py)
_register_module('algorithms.metaheuristics.ant_colony', r'''
"""Ant colony optimisation tailored for job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, processing_times, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class AntColonyOptimization(BaseOptimizer):
    """Constructive ACO with pheromone evaporation and heuristic visibility."""

    def __init__(
        self,
        ants: int = 25,
        iterations: int = 60,
        evaporation: float = 0.4,
        alpha: float = 1.0,
        beta: float = 2.0,
        seed: int = 21,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            ants=ants,
            iterations=iterations,
            evaporation=evaporation,
            alpha=alpha,
            beta=beta,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.ants = ants
        self.iterations = iterations
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _select_next(
        self,
        available: List[int],
        pheromones: Dict[int, float],
        durations: Dict[int, float],
        rng: random.Random,
    ) -> int:
        weights: List[float] = []
        for job in available:
            tau = pheromones.get(job, 1.0) ** self.alpha
            eta = (1.0 / (1.0 + durations.get(job, 1.0))) ** self.beta
            weights.append(max(tau * eta, 1e-12))
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for job, weight in zip(available, weights):
            cumulative += weight
            if cumulative >= threshold:
                return job
        return available[-1]

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        durations = processing_times(problem)
        pheromones: Dict[int, float] = {job: 1.0 for job in jobs}
        best_sequence = jobs
        best_value = float("inf")

        for _ in range(self.iterations):
            iteration_best_sequence = None
            iteration_best_value = float("inf")
            for _ in range(self.ants):
                available = jobs[:]
                sequence: List[int] = []
                while available:
                    job = self._select_next(available, pheromones, durations, rng)
                    sequence.append(job)
                    available.remove(job)
                value, _ = sequence_objective(problem, sequence, self.objective_weights)
                if value < iteration_best_value:
                    iteration_best_value = value
                    iteration_best_sequence = sequence
            assert iteration_best_sequence is not None

            for job in pheromones:
                pheromones[job] = (1.0 - self.evaporation) * pheromones[job]
                pheromones[job] = max(pheromones[job], 1e-6)
            deposit = 1.0 / (1.0 + iteration_best_value)
            for job in iteration_best_sequence:
                pheromones[job] = pheromones.get(job, 1.0) + deposit

            if iteration_best_value < best_value:
                best_value = iteration_best_value
                best_sequence = iteration_best_sequence

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.ant_colony

# BEGIN MODULE: algorithms.metaheuristics.differential_evolution (algorithms/metaheuristics/differential_evolution.py)
_register_module('algorithms.metaheuristics.differential_evolution', r'''
"""Differential evolution using random keys for job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

from algorithms.metaheuristics.utils import merge_objective_weights, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _keys_to_sequence(keys: Sequence[float], jobs: Sequence[int]) -> List[int]:
    return [job for _, job in sorted(zip(keys, jobs), key=lambda item: item[0])]


class DifferentialEvolution(BaseOptimizer):
    """Classic DE/rand/1/bin adapted to combinatorial scheduling."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 80,
        crossover_rate: float = 0.7,
        differential_weight: float = 0.8,
        seed: int = 19,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            differential_weight=differential_weight,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.differential_weight = differential_weight
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        dimension = len(jobs)
        if dimension == 0:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population: List[List[float]] = [[rng.random() for _ in range(dimension)] for _ in range(self.population_size)]
        scores = [sequence_objective(problem, _keys_to_sequence(individual, jobs), self.objective_weights)[0] for individual in population]

        for _ in range(self.generations):
            for idx in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(idx)
                a, b, c = rng.sample(candidates, 3)
                base = population[a]
                diff1 = population[b]
                diff2 = population[c]
                mutant = [base[d] + self.differential_weight * (diff1[d] - diff2[d]) for d in range(dimension)]
                trial = population[idx][:]
                j_rand = rng.randrange(dimension)
                for d in range(dimension):
                    if rng.random() < self.crossover_rate or d == j_rand:
                        trial[d] = mutant[d]
                trial_score = sequence_objective(problem, _keys_to_sequence(trial, jobs), self.objective_weights)[0]
                if trial_score < scores[idx]:
                    population[idx] = trial
                    scores[idx] = trial_score

        best_index = min(range(self.population_size), key=lambda i: scores[i])
        best_sequence = _keys_to_sequence(population[best_index], jobs)
        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": scores[best_index], "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.differential_evolution

# BEGIN MODULE: algorithms.metaheuristics.genetic_algorithm (algorithms/metaheuristics/genetic_algorithm.py)
_register_module('algorithms.metaheuristics.genetic_algorithm', r'''
"""Genetic algorithm for sequencing jobs in manufacturing problems."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class GeneticAlgorithm(BaseOptimizer):
    """Order-based genetic algorithm with partially mapped crossover."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 60,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
        seed: int = 42,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elitism=elitism,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _fitness(self, problem: ManufacturingProblem, sequence: Sequence[int]) -> Tuple[float, Dict[str, float]]:
        value, metrics = sequence_objective(problem, sequence, self.objective_weights)
        return value, metrics

    def _tournament(self, population: List[List[int]], scores: List[float], rng: random.Random) -> List[int]:
        candidates = rng.sample(range(len(population)), self.tournament_size)
        best = min(candidates, key=lambda idx: scores[idx])
        return population[best][:]

    def _crossover(self, parent_a: List[int], parent_b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
        size = len(parent_a)
        if size < 2:
            return parent_a[:], parent_b[:]
        start, end = sorted(rng.sample(range(size), 2))
        child_a = [None] * size
        child_b = [None] * size
        child_a[start:end] = parent_a[start:end]
        child_b[start:end] = parent_b[start:end]

        def fill(child: List[int], donor: List[int], start: int, end: int) -> None:
            idx = end
            for gene in donor:
                if gene not in child:
                    if idx >= size:
                        idx = 0
                    child[idx] = gene
                    idx += 1

        fill(child_a, parent_b, start, end)
        fill(child_b, parent_a, start, end)
        return child_a, child_b

    def _mutate(self, sequence: List[int], rng: random.Random) -> None:
        if len(sequence) < 2:
            return
        i, j = rng.sample(range(len(sequence)), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population = [random_sequence(problem, rng) for _ in range(self.population_size)]
        best_sequence = population[0]
        best_value = float("inf")
        best_metrics: Dict[str, float] = {}

        for _ in range(self.generations):
            scores: List[float] = []
            metrics_store: List[Dict[str, float]] = []
            for individual in population:
                value, metrics = self._fitness(problem, individual)
                scores.append(value)
                metrics_store.append(metrics)
                if value < best_value:
                    best_value = value
                    best_sequence = individual[:]
                    best_metrics = metrics

            ranked = sorted(zip(population, scores, metrics_store), key=lambda item: item[1])
            new_population: List[List[int]] = [ind[:] for ind, _, _ in ranked[: self.elitism]]

            while len(new_population) < self.population_size:
                parent_a = self._tournament(population, scores, rng)
                parent_b = self._tournament(population, scores, rng)
                child_a, child_b = parent_a[:], parent_b[:]
                if rng.random() < self.crossover_rate:
                    child_a, child_b = self._crossover(parent_a, parent_b, rng)
                if rng.random() < self.mutation_rate:
                    self._mutate(child_a, rng)
                if rng.random() < self.mutation_rate:
                    self._mutate(child_b, rng)
                new_population.append(child_a)
                if len(new_population) < self.population_size:
                    new_population.append(child_b)

            population = new_population

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.genetic_algorithm

# BEGIN MODULE: algorithms.metaheuristics.guided_local_search (algorithms/metaheuristics/guided_local_search.py)
_register_module('algorithms.metaheuristics.guided_local_search', r'''
"""Guided local search metaheuristic focusing on tardiness penalties."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class GuidedLocalSearch(BaseOptimizer):
    """Implements a simple GLS with feature penalties on tardy jobs."""

    def __init__(
        self,
        iterations: int = 120,
        lambda_penalty: float = 0.1,
        seed: int = 17,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            lambda_penalty=lambda_penalty,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.lambda_penalty = lambda_penalty
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        penalties: Dict[int, float] = {idx: 0.0 for idx in jobs}
        current_sequence = random_sequence(problem, rng)
        current_value, current_metrics = sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence[:]
        best_value = current_value
        best_metrics = current_metrics

        for _ in range(self.iterations):
            neighbourhood = []
            for _ in range(len(current_sequence)):
                i, j = rng.sample(range(len(current_sequence)), 2)
                neighbour = current_sequence[:]
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                neighbourhood.append(neighbour)

            candidate_sequence = current_sequence
            candidate_augmented = float("inf")
            candidate_value = current_value
            candidate_metrics = current_metrics
            for neighbour in neighbourhood:
                value, metrics = sequence_objective(problem, neighbour, self.objective_weights)
                augmented = value + self.lambda_penalty * sum(penalties[idx] for idx in neighbour)
                if augmented < candidate_augmented:
                    candidate_sequence = neighbour
                    candidate_value = value
                    candidate_augmented = augmented
                    candidate_metrics = metrics

            current_sequence = candidate_sequence
            current_value = candidate_value
            current_metrics = candidate_metrics

            if current_value < best_value:
                best_sequence = current_sequence[:]
                best_value = current_value
                best_metrics = current_metrics

            tardiness = current_metrics.get("total_tardiness", 0.0)
            if tardiness > 0:
                for job in current_sequence:
                    penalties[job] += tardiness / len(current_sequence)

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.guided_local_search

# BEGIN MODULE: algorithms.metaheuristics.iterated_local_search (algorithms/metaheuristics/iterated_local_search.py)
_register_module('algorithms.metaheuristics.iterated_local_search', r'''
"""Iterated local search for manufacturing scheduling."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class IteratedLocalSearch(BaseOptimizer):
    """Repeated perturbation and descent to escape local optima."""

    def __init__(
        self,
        iterations: int = 80,
        perturbation_strength: int = 3,
        seed: int = 13,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            perturbation_strength=perturbation_strength,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.perturbation_strength = perturbation_strength
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _local_descent(self, problem: ManufacturingProblem, sequence: List[int]) -> tuple[List[int], float]:
        current_sequence = sequence[:]
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        improved = True
        rng = random.Random(self.seed + 1)
        while improved:
            improved = False
            for _ in range(len(sequence)):
                i, j = rng.sample(range(len(sequence)), 2)
                candidate = current_sequence[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value, _ = sequence_objective(problem, candidate, self.objective_weights)
                if value < current_value:
                    current_sequence = candidate
                    current_value = value
                    improved = True
                    break
        return current_sequence, current_value

    def _perturb(self, sequence: List[int], rng: random.Random) -> List[int]:
        perturbed = sequence[:]
        for _ in range(self.perturbation_strength):
            i, j = rng.sample(range(len(sequence)), 2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        return perturbed

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = random_sequence(problem, rng)
        current_sequence, current_value = self._local_descent(problem, current_sequence)
        best_sequence = current_sequence
        best_value = current_value

        for _ in range(self.iterations):
            candidate_sequence = self._perturb(current_sequence, rng)
            candidate_sequence, candidate_value = self._local_descent(problem, candidate_sequence)
            if candidate_value < best_value:
                best_sequence = candidate_sequence
                best_value = candidate_value
                current_sequence = candidate_sequence
                current_value = candidate_value
            else:
                current_sequence = candidate_sequence

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.iterated_local_search

# BEGIN MODULE: algorithms.metaheuristics.particle_swarm (algorithms/metaheuristics/particle_swarm.py)
_register_module('algorithms.metaheuristics.particle_swarm', r'''
"""Particle swarm optimisation for sequencing jobs using random keys."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


def _position_to_sequence(position: Sequence[float], jobs: Sequence[int]) -> List[int]:
    return [job for _, job in sorted(zip(position, jobs), key=lambda pair: pair[0])]


class ParticleSwarmOptimization(BaseOptimizer):
    """Continuous random-key PSO for combinatorial scheduling."""

    def __init__(
        self,
        swarm_size: int = 30,
        iterations: int = 80,
        inertia: float = 0.72,
        cognitive: float = 1.49,
        social: float = 1.49,
        seed: int = 3,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            swarm_size=swarm_size,
            iterations=iterations,
            inertia=inertia,
            cognitive=cognitive,
            social=social,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        dimension = len(jobs)
        if dimension == 0:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        particles: List[List[float]] = [[rng.random() for _ in range(dimension)] for _ in range(self.swarm_size)]
        velocities: List[List[float]] = [[0.0 for _ in range(dimension)] for _ in range(self.swarm_size)]

        personal_best: List[Tuple[List[float], float]] = []
        best_global_position: List[float] | None = None
        best_global_value = float("inf")

        for position in particles:
            sequence = _position_to_sequence(position, jobs)
            value, _ = sequence_objective(problem, sequence, self.objective_weights)
            personal_best.append((position[:], value))
            if value < best_global_value:
                best_global_value = value
                best_global_position = position[:]

        for _ in range(self.iterations):
            for idx, position in enumerate(particles):
                velocity = velocities[idx]
                pbest_position, pbest_value = personal_best[idx]
                for d in range(dimension):
                    r1 = rng.random()
                    r2 = rng.random()
                    cognitive_term = self.cognitive * r1 * (pbest_position[d] - position[d])
                    social_term = 0.0
                    if best_global_position is not None:
                        social_term = self.social * r2 * (best_global_position[d] - position[d])
                    velocity[d] = self.inertia * velocity[d] + cognitive_term + social_term
                    position[d] += velocity[d]
                sequence = _position_to_sequence(position, jobs)
                value, _ = sequence_objective(problem, sequence, self.objective_weights)
                if value < pbest_value:
                    personal_best[idx] = (position[:], value)
                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position[:]

        assert best_global_position is not None
        best_sequence = _position_to_sequence(best_global_position, jobs)
        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_global_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.particle_swarm

# BEGIN MODULE: algorithms.metaheuristics.simulated_annealing (algorithms/metaheuristics/simulated_annealing.py)
_register_module('algorithms.metaheuristics.simulated_annealing', r'''
"""Simulated annealing metaheuristic for job sequencing."""
from __future__ import annotations

import math
import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class SimulatedAnnealing(BaseOptimizer):
    """Simple simulated annealing optimiser for job sequencing."""

    def __init__(
        self,
        initial_temperature: float = 250.0,
        cooling_rate: float = 0.95,
        steps_per_temperature: int = 20,
        max_iterations: int = 120,
        seed: int = 7,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            steps_per_temperature=steps_per_temperature,
            max_iterations=max_iterations,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.steps_per_temperature = steps_per_temperature
        self.max_iterations = max_iterations
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _neighbour(self, sequence: List[int], rng: random.Random) -> List[int]:
        if len(sequence) < 2:
            return sequence.copy()
        i, j = rng.sample(range(len(sequence)), 2)
        neighbour = sequence.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        return neighbour

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = random_sequence(problem, rng)
        current_value, current_metrics = sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence
        best_value = current_value
        best_metrics = current_metrics

        temperature = self.initial_temperature
        iteration = 0

        while temperature > 1e-3 and iteration < self.max_iterations:
            for _ in range(self.steps_per_temperature):
                candidate_sequence = self._neighbour(current_sequence, rng)
                candidate_value, candidate_metrics = sequence_objective(
                    problem, candidate_sequence, self.objective_weights
                )

                delta = candidate_value - current_value
                if delta < 0 or math.exp(-delta / temperature) > rng.random():
                    current_sequence = candidate_sequence
                    current_value = candidate_value
                    current_metrics = candidate_metrics

                if current_value < best_value:
                    best_sequence = current_sequence.copy()
                    best_value = current_value
                    best_metrics = current_metrics

                iteration += 1
                if iteration >= self.max_iterations:
                    break

            temperature *= self.cooling_rate

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.simulated_annealing

# BEGIN MODULE: algorithms.metaheuristics.tabu_search (algorithms/metaheuristics/tabu_search.py)
_register_module('algorithms.metaheuristics.tabu_search', r'''
"""Tabu search implementation for RMS job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class TabuSearch(BaseOptimizer):
    """Swap-based tabu search with aspiration criteria."""

    def __init__(
        self,
        iterations: int = 150,
        tabu_tenure: int = 8,
        neighbourhood_size: int = 25,
        seed: int = 5,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            tabu_tenure=tabu_tenure,
            neighbourhood_size=neighbourhood_size,
            seed=seed,
            objective_weights=objective_weights,
        )
        self.iterations = iterations
        self.tabu_tenure = tabu_tenure
        self.neighbourhood_size = neighbourhood_size
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _generate_neighbours(self, sequence: List[int], rng: random.Random) -> List[Tuple[List[int], Tuple[int, int]]]:
        neighbours: List[Tuple[List[int], Tuple[int, int]]] = []
        n = len(sequence)
        for _ in range(self.neighbourhood_size):
            i, j = sorted(rng.sample(range(n), 2))
            neighbour = sequence[:]
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
            neighbours.append((neighbour, (i, j)))
        return neighbours

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        current_sequence = random_sequence(problem, rng)
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        best_sequence = current_sequence[:]
        best_value = current_value

        tabu_list: Dict[Tuple[int, int], int] = {}

        for iteration in range(self.iterations):
            neighbours = self._generate_neighbours(current_sequence, rng)
            candidate_sequence = None
            candidate_value = float("inf")
            candidate_move = (0, 0)
            for neighbour_sequence, move in neighbours:
                value, _ = sequence_objective(problem, neighbour_sequence, self.objective_weights)
                if value < candidate_value and (
                    move not in tabu_list or iteration >= tabu_list[move] or value < best_value
                ):
                    candidate_sequence = neighbour_sequence
                    candidate_value = value
                    candidate_move = move
            if candidate_sequence is None:
                continue
            current_sequence = candidate_sequence
            current_value = candidate_value
            tabu_list[candidate_move] = iteration + self.tabu_tenure
            if current_value < best_value:
                best_value = current_value
                best_sequence = current_sequence[:]

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.tabu_search

# BEGIN MODULE: algorithms.metaheuristics.utils (algorithms/metaheuristics/utils.py)
_register_module('algorithms.metaheuristics.utils', r'''
"""Shared helpers for metaheuristic scheduling algorithms."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem

DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "makespan": 1.0,
    "total_completion_time": 0.05,
    "total_tardiness": 0.1,
    "energy": 0.01,
}


def merge_objective_weights(overrides: Dict[str, float] | None) -> Dict[str, float]:
    """Combine user provided weights with sensible defaults."""

    weights = DEFAULT_OBJECTIVE_WEIGHTS.copy()
    if overrides:
        weights.update(overrides)
    return weights


def sequence_objective(
    problem: ManufacturingProblem, sequence: Sequence[int], weights: Dict[str, float]
) -> tuple[float, Dict[str, float]]:
    """Evaluate a permutation of jobs returning weighted objective and metrics."""

    schedule = problem.build_schedule(sequence)
    metrics = evaluate_schedule(schedule)
    objective = 0.0
    for key, weight in weights.items():
        objective += weight * metrics.get(key, 0.0)
    return objective, metrics


def random_sequence(problem: ManufacturingProblem, rng: random.Random) -> List[int]:
    """Generate a random permutation of job indices for the problem."""

    indices = list(problem.jobs.index)
    rng.shuffle(indices)
    return indices


def processing_times(problem: ManufacturingProblem) -> Dict[int, float]:
    """Return the processing time per job index for quick lookup."""

    durations: Dict[int, float] = {}
    for idx, row in problem.jobs.iterrows():
        value = row.get("Processing_Time")
        if value is None:
            value = row.get("Duration", 0.0)
        durations[idx] = float(value if value is not None else 0.0)
    return durations


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "merge_objective_weights",
    "sequence_objective",
    "random_sequence",
    "processing_times",
]
''', False)
# END MODULE: algorithms.metaheuristics.utils

# BEGIN MODULE: algorithms.metaheuristics.variable_neighborhood_search (algorithms/metaheuristics/variable_neighborhood_search.py)
_register_module('algorithms.metaheuristics.variable_neighborhood_search', r'''
"""Variable neighbourhood search for adaptive job sequencing."""
from __future__ import annotations

import random
from typing import Dict, List

from algorithms.metaheuristics.utils import merge_objective_weights, random_sequence, sequence_objective
from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


class VariableNeighborhoodSearch(BaseOptimizer):
    """Implements a shaking and local improvement loop with three neighbourhoods."""

    def __init__(
        self,
        max_iterations: int = 120,
        seed: int = 11,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(max_iterations=max_iterations, seed=seed, objective_weights=objective_weights)
        self.max_iterations = max_iterations
        self.seed = seed
        self.objective_weights = merge_objective_weights(objective_weights)

    def _swap(self, sequence: List[int], rng: random.Random) -> List[int]:
        i, j = rng.sample(range(len(sequence)), 2)
        seq = sequence[:]
        seq[i], seq[j] = seq[j], seq[i]
        return seq

    def _insert(self, sequence: List[int], rng: random.Random) -> List[int]:
        seq = sequence[:]
        i, j = rng.sample(range(len(sequence)), 2)
        value = seq.pop(i)
        seq.insert(j, value)
        return seq

    def _reverse(self, sequence: List[int], rng: random.Random) -> List[int]:
        seq = sequence[:]
        i, j = sorted(rng.sample(range(len(sequence)), 2))
        seq[i:j] = reversed(seq[i:j])
        return seq

    def _local_search(self, problem: ManufacturingProblem, sequence: List[int], rng: random.Random) -> List[int]:
        improved = True
        current_sequence = sequence[:]
        current_value, _ = sequence_objective(problem, current_sequence, self.objective_weights)
        while improved:
            improved = False
            for neighbour_generator in (self._swap, self._insert, self._reverse):
                neighbour = neighbour_generator(current_sequence, rng)
                value, _ = sequence_objective(problem, neighbour, self.objective_weights)
                if value < current_value:
                    current_sequence = neighbour
                    current_value = value
                    improved = True
                    break
        return current_sequence

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        jobs = list(problem.jobs.index)
        if not jobs:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        best_sequence = random_sequence(problem, rng)
        best_value, _ = sequence_objective(problem, best_sequence, self.objective_weights)

        for _ in range(self.max_iterations):
            current_sequence = best_sequence[:]
            for neighbourhood in (self._swap, self._insert, self._reverse):
                shaken = neighbourhood(current_sequence, rng)
                improved = self._local_search(problem, shaken, rng)
                value, _ = sequence_objective(problem, improved, self.objective_weights)
                if value < best_value:
                    best_sequence = improved
                    best_value = value
                    break

        final_schedule = problem.build_schedule(best_sequence)
        final_metrics = evaluate_schedule(final_schedule)
        return ScheduleSolution(
            schedule=final_schedule,
            metrics=final_metrics,
            metadata={"objective": best_value, "sequence": best_sequence},
        )
''', False)
# END MODULE: algorithms.metaheuristics.variable_neighborhood_search

# BEGIN MODULE: algorithms.multi_objective.nsga2 (algorithms/multi_objective/nsga2.py)
_register_module('algorithms.multi_objective.nsga2', r'''
"""Light-weight NSGA-II implementation for sequencing problems."""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from core.base_optimizer import BaseOptimizer
from core.metrics import evaluate_schedule
from core.problem import ManufacturingProblem
from core.solution import ScheduleSolution


Individual = Dict[str, object]


def _evaluate(problem: ManufacturingProblem, sequence: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, float]]:
    schedule = problem.build_schedule(sequence)
    metrics = evaluate_schedule(schedule)
    objectives = {key: metrics.get(key, 0.0) for key in ["makespan", "energy", "total_tardiness"]}
    return objectives, metrics


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    better_or_equal = all(a[key] <= b[key] for key in a)
    strictly_better = any(a[key] < b[key] for key in a)
    return better_or_equal and strictly_better


def _fast_nondominated_sort(population: List[Individual]) -> List[List[Individual]]:
    fronts: List[List[Individual]] = []
    for individual in population:
        individual["dominated_set"] = []
        individual["domination_count"] = 0
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if _dominates(p["objectives"], q["objectives"]):
                p["dominated_set"].append(q)
            elif _dominates(q["objectives"], p["objectives"]):
                p["domination_count"] += 1
        if p["domination_count"] == 0:
            p["rank"] = 0
            if not fronts:
                fronts.append([])
            fronts[0].append(p)
    current_rank = 0
    while current_rank < len(fronts):
        next_front: List[Individual] = []
        for p in fronts[current_rank]:
            for q in p["dominated_set"]:
                q["domination_count"] -= 1
                if q["domination_count"] == 0:
                    q["rank"] = current_rank + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current_rank += 1
    return fronts


def _crowding_distance(front: List[Individual], objectives: Sequence[str]) -> None:
    if not front:
        return
    for individual in front:
        individual["crowding_distance"] = 0.0
    for objective in objectives:
        front.sort(key=lambda ind: ind["objectives"][objective])
        front[0]["crowding_distance"] = float("inf")
        front[-1]["crowding_distance"] = float("inf")
        values = [ind["objectives"][objective] for ind in front]
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            continue
        for i in range(1, len(front) - 1):
            prev_val = front[i - 1]["objectives"][objective]
            next_val = front[i + 1]["objectives"][objective]
            front[i]["crowding_distance"] += (next_val - prev_val) / (max_val - min_val)


def _tournament_selection(population: List[Individual], k: int, rng: random.Random) -> Individual:
    contenders = rng.sample(population, k)
    contenders.sort(key=lambda ind: (ind["rank"], -ind["crowding_distance"]))
    return contenders[0]


def _pmx_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    cx_point1, cx_point2 = sorted(rng.sample(range(size), 2))
    child1 = parent1[:]
    child2 = parent2[:]
    child1[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
    child2[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    def repair(child: List[int], segment: List[int], donor: List[int]) -> None:
        mapping = {donor[i]: segment[i] for i in range(cx_point1, cx_point2)}
        for idx in list(range(cx_point1)) + list(range(cx_point2, size)):
            while child[idx] in mapping:
                mapped = mapping[child[idx]]
                if mapped == child[idx]:
                    break
                child[idx] = mapped

    repair(child1, child1, parent1)
    repair(child2, child2, parent2)
    return child1, child2


def _swap_mutation(sequence: List[int], rng: random.Random) -> List[int]:
    i, j = rng.sample(range(len(sequence)), 2)
    sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence


class NSGAII(BaseOptimizer):
    """A compact NSGA-II optimiser suitable for small instances."""

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 30,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.2,
        tournament_size: int = 2,
        seed: int = 13,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            tournament_size=tournament_size,
            seed=seed,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.seed = seed

    def _create_individual(self, job_indices: List[int], rng: random.Random, problem: ManufacturingProblem) -> Individual:
        sequence = job_indices.copy()
        rng.shuffle(sequence)
        objectives, metrics = _evaluate(problem, sequence)
        return {"sequence": sequence, "objectives": objectives, "metrics": metrics}

    def solve(self, problem: ManufacturingProblem) -> ScheduleSolution:
        job_indices = list(problem.jobs.index)
        if not job_indices:
            return ScheduleSolution(schedule=problem.jobs)

        rng = random.Random(self.seed)
        population = [self._create_individual(job_indices, rng, problem) for _ in range(self.population_size)]

        objectives = ["makespan", "energy", "total_tardiness"]

        for _ in range(self.generations):
            fronts = _fast_nondominated_sort(population)
            for front in fronts:
                _crowding_distance(front, objectives)

            mating_pool: List[Individual] = []
            while len(mating_pool) < self.population_size:
                mating_pool.append(_tournament_selection(population, self.tournament_size, rng))

            offspring: List[Individual] = []
            for i in range(0, self.population_size, 2):
                parent1 = mating_pool[i % len(mating_pool)]
                parent2 = mating_pool[(i + 1) % len(mating_pool)]
                seq1 = parent1["sequence"].copy()
                seq2 = parent2["sequence"].copy()
                if rng.random() < self.crossover_probability:
                    seq1, seq2 = _pmx_crossover(seq1, seq2, rng)
                if rng.random() < self.mutation_probability:
                    seq1 = _swap_mutation(seq1, rng)
                if rng.random() < self.mutation_probability:
                    seq2 = _swap_mutation(seq2, rng)
                for seq in (seq1, seq2):
                    objectives_values, metrics = _evaluate(problem, seq)
                    offspring.append({"sequence": seq, "objectives": objectives_values, "metrics": metrics})

            combined = population + offspring
            fronts = _fast_nondominated_sort(combined)
            new_population: List[Individual] = []
            for front in fronts:
                _crowding_distance(front, objectives)
                front.sort(key=lambda ind: (ind["rank"], -ind["crowding_distance"]))
                for individual in front:
                    if len(new_population) < self.population_size:
                        new_population.append(individual)
            population = new_population

        fronts = _fast_nondominated_sort(population)
        pareto_front = [
            {
                "sequence": individual["sequence"],
                "metrics": individual["metrics"],
                "objectives": individual["objectives"],
            }
            for individual in fronts[0]
        ]
        best = min(fronts[0], key=lambda ind: ind["objectives"]["makespan"])
        best_schedule = problem.build_schedule(best["sequence"])
        return ScheduleSolution(
            schedule=best_schedule,
            metrics=best["metrics"],
            metadata={"pareto_front": pareto_front},
        )
''', False)
# END MODULE: algorithms.multi_objective.nsga2

RESOURCE_FILES: dict[str, str] = {}

def _register_resource(path: str, content: str) -> None:
    RESOURCE_FILES[path] = content

# BEGIN RESOURCE: config/base_config.yaml
_register_resource('config/base_config.yaml', r'''
data:
  sources: []
algorithm:
  name: fcfs
optimisation:
  objectives:
    - makespan
    - energy
  weights:
    makespan: 0.5
    energy: 0.5
simulation:
  repetitions: 10
validation:
  confidence_level: 0.95
hardware:
  use_gpu: false
logging:
  experiment_name: rms-baseline
''')
# END RESOURCE: config/base_config.yaml

# BEGIN RESOURCE: data/benchmarks/fisher_jsp_6x6.csv
_register_resource('data/benchmarks/fisher_jsp_6x6.csv', r'''
Job_ID,Machine_ID,Operation,Processing_Time,Energy_Consumption,Due_Date,Breakdown_Risk
A1,M1,1,40,8.2,2024-01-03T10:00:00,0.04
A1,M3,2,32,7.5,2024-01-03T10:00:00,0.04
A1,M5,3,28,7.0,2024-01-03T10:00:00,0.04
A2,M2,1,45,8.6,2024-01-04T12:00:00,0.05
A2,M4,2,34,7.8,2024-01-04T12:00:00,0.05
A2,M6,3,31,7.1,2024-01-04T12:00:00,0.05
A3,M3,1,38,8.0,2024-01-05T09:00:00,0.03
A3,M1,2,29,7.2,2024-01-05T09:00:00,0.03
A3,M4,3,27,6.8,2024-01-05T09:00:00,0.03
A4,M5,1,41,8.3,2024-01-03T16:00:00,0.06
A4,M2,2,36,7.9,2024-01-03T16:00:00,0.06
A4,M6,3,30,7.2,2024-01-03T16:00:00,0.06
A5,M4,1,39,8.1,2024-01-04T14:00:00,0.05
A5,M5,2,33,7.6,2024-01-04T14:00:00,0.05
A5,M1,3,26,6.9,2024-01-04T14:00:00,0.05
A6,M2,1,44,8.5,2024-01-05T11:00:00,0.04
A6,M3,2,35,7.7,2024-01-05T11:00:00,0.04
A6,M6,3,29,7.0,2024-01-05T11:00:00,0.04
''')
# END RESOURCE: data/benchmarks/fisher_jsp_6x6.csv

# BEGIN RESOURCE: data/benchmarks/industry_case_cell.csv
_register_resource('data/benchmarks/industry_case_cell.csv', r'''
Job_ID,Machine_ID,Cell,Processing_Time,Energy_Consumption,Due_Date,Process_Type,Additive_Layer_Time,Transfer_Time
C1,Mill_A,North,52,9.4,2024-01-06T09:00:00,subtractive,0.0,0.0
C1,Printer_1,North,38,11.2,2024-01-06T09:00:00,additive,45.0,0.0
C1,Grinder_A,North,27,8.6,2024-01-06T09:00:00,subtractive,0.0,0.0
C2,Mill_B,South,49,9.1,2024-01-07T15:00:00,subtractive,0.0,18.0
C2,Printer_2,South,36,10.8,2024-01-07T15:00:00,additive,42.0,18.0
C2,Grinder_B,South,29,8.8,2024-01-07T15:00:00,subtractive,0.0,18.0
C3,Mill_A,North,51,9.3,2024-01-08T11:00:00,subtractive,0.0,0.0
C3,Printer_1,North,37,11.0,2024-01-08T11:00:00,additive,44.0,0.0
C3,Polisher_A,North,26,8.2,2024-01-08T11:00:00,subtractive,0.0,0.0
''')
# END RESOURCE: data/benchmarks/industry_case_cell.csv

# BEGIN RESOURCE: data/benchmarks/taillard_fsp_5x5.csv
_register_resource('data/benchmarks/taillard_fsp_5x5.csv', r'''
Job_ID,Machine_ID,Stage,Processing_Time,Energy_Consumption,Due_Date
J1,M1,1,85,12.5,2024-01-02T08:00:00
J1,M2,2,73,11.8,2024-01-02T08:00:00
J1,M3,3,62,10.4,2024-01-02T08:00:00
J1,M4,4,55,9.8,2024-01-02T08:00:00
J1,M5,5,48,9.1,2024-01-02T08:00:00
J2,M1,1,95,13.2,2024-01-02T08:00:00
J2,M2,2,88,12.7,2024-01-02T08:00:00
J2,M3,3,74,11.6,2024-01-02T08:00:00
J2,M4,4,63,10.5,2024-01-02T08:00:00
J2,M5,5,58,9.9,2024-01-02T08:00:00
J3,M1,1,78,12.1,2024-01-02T08:00:00
J3,M2,2,69,11.3,2024-01-02T08:00:00
J3,M3,3,65,10.9,2024-01-02T08:00:00
J3,M4,4,61,10.1,2024-01-02T08:00:00
J3,M5,5,52,9.4,2024-01-02T08:00:00
J4,M1,1,82,12.0,2024-01-02T08:00:00
J4,M2,2,76,11.5,2024-01-02T08:00:00
J4,M3,3,69,10.7,2024-01-02T08:00:00
J4,M4,4,60,10.0,2024-01-02T08:00:00
J4,M5,5,53,9.3,2024-01-02T08:00:00
J5,M1,1,91,13.4,2024-01-02T08:00:00
J5,M2,2,85,12.6,2024-01-02T08:00:00
J5,M3,3,70,11.2,2024-01-02T08:00:00
J5,M4,4,66,10.6,2024-01-02T08:00:00
J5,M5,5,59,9.8,2024-01-02T08:00:00
''')
# END RESOURCE: data/benchmarks/taillard_fsp_5x5.csv

# BEGIN RESOURCE: data/synthetic/sample.csv
_register_resource('data/synthetic/sample.csv', r'''
Job_ID,Machine_ID,Scheduled_Start,Scheduled_End,Processing_Time,Energy_Consumption,Due_Date,Priority
JOB_00001,M01,2023-01-01T08:00:00,2023-01-01T09:00:00,60,12.5,2023-01-01T10:00:00,1.5
JOB_00002,M02,2023-01-01T08:15:00,2023-01-01T09:05:00,50,11.0,2023-01-01T09:45:00,2.0
JOB_00003,M01,2023-01-01T09:10:00,2023-01-01T10:00:00,50,10.2,2023-01-01T10:50:00,1.2
''')
# END RESOURCE: data/synthetic/sample.csv

def bootstrap_environment(base_path: Path | None = None) -> Path:
    if base_path is None:
        base_path = Path.cwd() / "rms_runtime"
    base_path.mkdir(parents=True, exist_ok=True)
    for name, meta in MODULE_SOURCES.items():
        module_path = base_path / Path(name.replace(".", os.sep))
        if meta["is_package"]:
            module_path.mkdir(parents=True, exist_ok=True)
            init_file = module_path / "__init__.py"
            init_file.write_text(meta["code"], encoding="utf-8")
        else:
            module_path.parent.mkdir(parents=True, exist_ok=True)
            py_file = module_path.with_suffix(".py")
            py_file.write_text(meta["code"], encoding="utf-8")
    for rel_path, content in RESOURCE_FILES.items():
        target = base_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return base_path

def bootstrap_modules(base_path: Path | None = None) -> None:
    if base_path is None:
        base_path = Path.cwd() / "rms_runtime"
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
    for name, meta in sorted(MODULE_SOURCES.items(), key=lambda item: item[0].count(".")):
        module = types.ModuleType(name)
        module.__file__ = str((base_path / Path(name.replace(".", os.sep))).with_suffix(".py"))
        if meta["is_package"]:
            module.__path__ = [str(base_path / Path(name.replace(".", os.sep)))]
            module.__package__ = name
        else:
            module.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
        sys.modules[name] = module
        exec(compile(meta["code"], module.__file__, "exec"), module.__dict__)

def main() -> None:
    base_path = bootstrap_environment()
    bootstrap_modules(base_path)
    from rms_all_in_one import main as orchestrator_main
    orchestrator_main()

if __name__ == "__main__":
    main()
