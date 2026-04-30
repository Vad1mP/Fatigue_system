from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def resolve_feature_sets(
    feature_set_names: Iterable[str],
    feature_sets: dict[str, Any],
    available_columns: Iterable[str],
    *,
    mode: str = "strict",
) -> list[str]:
    available = list(available_columns)
    available_set = set(available)
    resolved: list[str] = []
    visiting: set[str] = set()

    def resolve_one(name: str) -> list[str]:
        if name not in feature_sets:
            if mode == "strict":
                raise KeyError(f"Feature set '{name}' is not defined in protocol")
            return [c for c in available if c.startswith(name)]
        if name in visiting:
            raise ValueError(f"Cyclic feature set reference detected: {name}")
        visiting.add(name)
        spec = feature_sets.get(name) or {}
        features: list[str] = []

        for child in spec.get("include_feature_sets", []) or []:
            features.extend(resolve_one(child))

        features.extend(spec.get("features", []) or [])

        for prefix in spec.get("include_prefixes", []) or []:
            features.extend([c for c in available if str(c).startswith(prefix)])

        exclude = set(spec.get("exclude", []) or [])
        exclude_contains = list(spec.get("exclude_contains", []) or [])
        out = []
        for f in features:
            if f in exclude:
                continue
            if any(token in f for token in exclude_contains):
                continue
            out.append(f)
        visiting.remove(name)
        return out

    for name in feature_set_names:
        resolved.extend(resolve_one(name))

    # keep order, drop duplicates, then check availability
    seen: set[str] = set()
    final: list[str] = []
    missing: list[str] = []
    for f in resolved:
        if f in seen:
            continue
        seen.add(f)
        if f in available_set:
            final.append(f)
        else:
            missing.append(f)
    if missing and mode == "strict":
        raise KeyError(
            "Feature set contains columns missing from input table: "
            + ", ".join(missing[:20])
            + ("..." if len(missing) > 20 else "")
        )
    return final


def formula_key(row: pd.Series | dict[str, Any]) -> str:
    formula = str(row.get("formula", "")).strip()
    base = row.get("base_formula", None)
    if pd.isna(base) or str(base).strip() == "":
        return formula
    return f"{str(base).strip()}_to_{formula}"


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "valid"])


def feature_validation_column(feature: str, schema: dict[str, str]) -> str | None:
    if str(feature).startswith("hrv_"):
        return schema.get("hrv_valid")
    if str(feature).startswith("morph_"):
        return schema.get("morphology_valid")
    return None
