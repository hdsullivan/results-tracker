"""Value maps: derive a labelled field from a raw one (`config.kernel` 0-3 -> "isotropic").

Rules are plain data (models.ValueMap.rules): `{"label": "isotropic", "values": [0, 1, 2, 3]}` or
`{"label": "motion", "range": [8, 11]}` (inclusive). `apply_value_maps` writes the derived values into each
record's `derived` dict; `aggregate.get_field` reads them as `derived.<name>`, so every table, filter and figure
can group by them. The text grammar (`parse_rules` / `format_rules`) is what the Settings page and the CLI use:

    isotropic = 0, 1, 2, 3
    anisotropic = 4-7
    motion = 8-11
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import aggregate as agg

Record = dict[str, Any]


def _literal(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_rules(text: str) -> list[dict[str, Any]]:
    """`label = v1, v2` or `label = lo-hi` per line; blank lines and `#` comments are skipped."""
    rules: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"rule {line!r}: expected `label = values`")
        label, _, rhs = line.partition("=")
        label, rhs = label.strip(), rhs.strip()
        if not label or not rhs:
            raise ValueError(f"rule {line!r}: needs a label and values")
        lo_hi = rhs.split("-") if rhs.count("-") == 1 and "," not in rhs and not rhs.startswith("-") else None
        if lo_hi and all(part.strip() and isinstance(_literal(part), (int, float)) for part in lo_hi):
            rules.append({"label": label, "range": [_literal(lo_hi[0]), _literal(lo_hi[1])]})
        else:
            rules.append({"label": label, "values": [_literal(v) for v in rhs.split(",") if v.strip()]})
    if not rules:
        raise ValueError("no rules")
    return rules


def format_rules(rules: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for r in rules:
        if "range" in r:
            lo, hi = r["range"]
            lines.append(f"{r['label']} = {agg.fmt_value(lo)}-{agg.fmt_value(hi)}")
        else:
            lines.append(f"{r['label']} = " + ", ".join(agg.fmt_value(v) for v in r.get("values", [])))
    return "\n".join(lines)


def derive(rules: Sequence[Mapping[str, Any]], value: Any) -> Optional[str]:
    """The label of the first rule matching `value`, else None."""
    if value is None:
        return None
    for r in rules:
        if "range" in r:
            lo, hi = r["range"]
            try:
                if lo <= value <= hi:
                    return str(r["label"])
            except TypeError:
                continue
        elif any(agg.same_value(value, v) for v in r.get("values", [])):
            return str(r["label"])
    return None


def apply_value_maps(records: Iterable[Record], maps: Sequence[Mapping[str, Any]]) -> list[Record]:
    """Set `record["derived"][name]` for every map (`{"name", "field", "rules"}`), in place; returns the list."""
    recs = list(records)
    for r in recs:
        d = r.setdefault("derived", {})
        order = r.setdefault("derived_order", {})
        for m in maps:
            d[m["name"]] = derive(m["rules"], agg.get_field(r, m["field"]))
            order[m["name"]] = rule_labels(m["rules"])
    return recs


def rule_labels(rules: Iterable[Mapping[str, Any]]) -> list[str]:
    """Labels in rule order: the natural column order for a derived field."""
    return [str(r["label"]) for r in rules]
