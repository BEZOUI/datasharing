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

