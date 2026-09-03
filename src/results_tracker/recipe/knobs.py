"""Declared parameter spaces: the one thing a recipe imposes on a method or a problem.

A `Knob` is a named, typed parameter with a default and (optionally) admissible values. A method
declares its knobs, a problem declares its conditions, and everything the GUI or a study spec can
sweep, ablate or filter on has to be declared here first. Nothing else about a method is constrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional

KINDS = ("float", "int", "bool", "choice", "str")


@dataclass(frozen=True)
class Knob:
    """One parameter: `kind` is float | int | bool | choice | str.

    `bounds` (lo, hi) applies to numeric knobs, `choices` to choice knobs, `log` marks a knob that is
    naturally swept on a logarithmic axis. `None` is accepted as a value only when the default is `None`
    (an optional knob such as a floor that may be switched off)."""

    name: str
    kind: str = "float"
    default: Any = None
    choices: Optional[tuple] = None
    bounds: Optional[tuple[float, float]] = None
    log: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("a knob needs a name")
        if self.kind not in KINDS:
            raise ValueError(f"knob {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"knob {self.name!r}: a choice knob needs `choices`")
        if self.bounds is not None:
            if self.kind not in ("float", "int"):
                raise ValueError(f"knob {self.name!r}: bounds only apply to numeric knobs")
            lo, hi = self.bounds
            if not lo <= hi:
                raise ValueError(f"knob {self.name!r}: bounds must satisfy lo <= hi, got {self.bounds}")
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
        if self.default is not None:
            object.__setattr__(self, "default", self.validate(self.default))

    # ------------------------------------------------------------------ values

    def validate(self, value: Any) -> Any:
        """Coerce `value` to the knob's type and check it is admissible; raise ValueError otherwise."""
        if value is None:
            if self.default is None:
                return None
            raise ValueError(f"knob {self.name!r}: None is not a value")
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise ValueError(f"knob {self.name!r}: expected a bool, got {value!r}")
        if self.kind in ("float", "int"):
            if isinstance(value, bool):
                raise ValueError(f"knob {self.name!r}: expected a number, got a bool")
            if isinstance(value, str):
                try:
                    value = float(value) if self.kind == "float" or "." in value or "e" in value.lower() else int(value)
                except ValueError:
                    raise ValueError(f"knob {self.name!r}: expected a number, got {value!r}") from None
            if self.kind == "int":
                if isinstance(value, float):
                    if not value.is_integer():
                        raise ValueError(f"knob {self.name!r}: expected an int, got {value!r}")
                    value = int(value)
                elif not isinstance(value, int):
                    raise ValueError(f"knob {self.name!r}: expected an int, got {value!r}")
            else:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"knob {self.name!r}: expected a float, got {value!r}")
                value = float(value)
            if self.bounds is not None and not self.bounds[0] <= value <= self.bounds[1]:
                raise ValueError(f"knob {self.name!r}: {value!r} is outside bounds {self.bounds}")
            return value
        if self.kind == "choice":
            for choice in self.choices or ():
                if value == choice or str(value) == str(choice):
                    return choice
            raise ValueError(f"knob {self.name!r}: {value!r} is not one of {list(self.choices or ())}")
        return str(value)

    # ------------------------------------------------------------------ (de)serialisation

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind, "default": self.default}
        if self.choices is not None:
            d["choices"] = list(self.choices)
        if self.bounds is not None:
            d["bounds"] = list(self.bounds)
        if self.log:
            d["log"] = True
        if self.doc:
            d["doc"] = self.doc
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Knob":
        return cls(
            name=d["name"], kind=d.get("kind", "float"), default=d.get("default"),
            choices=tuple(d["choices"]) if d.get("choices") is not None else None,
            bounds=tuple(d["bounds"]) if d.get("bounds") is not None else None,
            log=bool(d.get("log", False)), doc=d.get("doc", ""),
        )


class KnobSpace:
    """An ordered set of knobs with unique names; resolves partial overrides into full configs."""

    def __init__(self, knobs: Iterable[Knob] = ()):
        self._knobs: dict[str, Knob] = {}
        for k in knobs:
            if k.name in self._knobs:
                raise ValueError(f"duplicate knob {k.name!r}")
            self._knobs[k.name] = k

    def __iter__(self) -> Iterator[Knob]:
        return iter(self._knobs.values())

    def __len__(self) -> int:
        return len(self._knobs)

    def __contains__(self, name: object) -> bool:
        return name in self._knobs

    def __getitem__(self, name: str) -> Knob:
        return self._knobs[name]

    @property
    def names(self) -> list[str]:
        return list(self._knobs)

    def defaults(self) -> dict[str, Any]:
        return {k.name: k.default for k in self}

    def resolve(self, overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Full config in declaration order: defaults overridden by `overrides`, every value validated.

        Unknown keys are an error: a parameter that is set but never declared would silently do nothing."""
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(self._knobs))
        if unknown:
            raise KeyError(f"unknown knob(s) {unknown}; declared: {self.names}")
        return {k.name: k.validate(overrides[k.name]) if k.name in overrides else k.default for k in self}

    def diff(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
        """Names of knobs whose values differ between two full configs."""
        return [k.name for k in self if a.get(k.name) != b.get(k.name)]

    def to_list(self) -> list[dict[str, Any]]:
        return [k.to_dict() for k in self]

    @classmethod
    def from_list(cls, items: Iterable[Mapping[str, Any]]) -> "KnobSpace":
        return cls(Knob.from_dict(d) for d in items)
