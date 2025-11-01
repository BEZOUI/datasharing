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
