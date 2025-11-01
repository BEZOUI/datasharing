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

