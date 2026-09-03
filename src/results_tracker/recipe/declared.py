"""Knob declarations as data: dump a registry to JSON, rebuild a planning-only registry from it.

The GUI plans studies from the knobs methods and problems declare, but the GUI's environment need not be
able to *run* them (no torch, no data). `results-tracker recipe export-knobs -i adaptivepnp.recipes -o
studies/knobs.json` writes the declarations next to the specs; `load_declarations` turns them back into
stand-in classes that expand and validate specs but raise if asked to reconstruct anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

from .core import Method, Problem, Registry
from .knobs import KnobSpace

VERSION = 1


class DeclaredMethod(Method):
    """A method known only by its declaration; `reconstruct` cannot run."""

    def reconstruct(self, instance, config, state):
        raise RuntimeError(f"method {self.key!r} is a declaration only (from knobs.json); run studies in the repo's environment")


class DeclaredProblem(Problem):
    """A problem known only by its declaration; it cannot pose instances or score them."""

    def instances(self, condition, split, n, seed):
        raise RuntimeError(f"problem {self.key!r} is a declaration only (from knobs.json); run studies in the repo's environment")

    def metrics(self, estimate, instance):
        raise RuntimeError(f"problem {self.key!r} is a declaration only (from knobs.json)")


def export_declarations(registry: Registry) -> dict[str, Any]:
    """Everything planning needs about the registered methods and problems, as plain data."""
    return {
        "version": VERSION,
        "methods": [
            {"key": cls.key, "label": cls.label, "citation": cls.citation, "is_baseline": bool(cls.is_baseline),
             "knobs": cls.space().to_list()}
            for cls in registry.methods.values()
        ],
        "problems": [
            {"key": cls.key, "label": cls.label, "conditions": cls.condition_space().to_list(), "splits": list(cls.splits),
             "metric_definitions": {k: list(v) for k, v in cls.metric_definitions.items()}}
            for cls in registry.problems.values()
        ],
    }


def declared_registry(decl: Mapping[str, Any]) -> Registry:
    """A registry of stand-in classes rebuilt from `export_declarations` output."""
    reg = Registry()
    for m in decl.get("methods", []):
        cls = type(f"Declared_{m['key']}", (DeclaredMethod,), {
            "key": m["key"], "label": m.get("label", ""), "citation": m.get("citation", ""),
            "is_baseline": bool(m.get("is_baseline", False)), "knobs": tuple(KnobSpace.from_list(m.get("knobs", []))),
        })
        reg.method(cls)
    for p in decl.get("problems", []):
        cls = type(f"Declared_{p['key']}", (DeclaredProblem,), {
            "key": p["key"], "label": p.get("label", ""), "conditions": tuple(KnobSpace.from_list(p.get("conditions", []))),
            "splits": tuple(p.get("splits") or ("test",)),
            "metric_definitions": {k: tuple(v) for k, v in p.get("metric_definitions", {}).items()},
        })
        reg.problem(cls)
    return reg


def save_declarations(registry: Registry, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export_declarations(registry), indent=2, default=str) + "\n")
    return path


def load_declarations(path: Union[str, Path]) -> Registry:
    return declared_registry(json.loads(Path(path).read_text()))
