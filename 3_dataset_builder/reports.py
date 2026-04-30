from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def numeric_iqr(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.quantile(0.75) - x.quantile(0.25))


def summarize_series(s: pd.Series, total_n: int | None = None) -> dict[str, Any]:
    total_n = len(s) if total_n is None else total_n
    valid = s.notna()
    numeric = pd.to_numeric(s, errors="coerce")
    return {
        "n_total": int(total_n),
        "n_valid": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if total_n else 0.0,
        "n_missing": int((~valid).sum()),
        "missing_ratio": float((~valid).mean()) if total_n else 0.0,
        "n_unique": int(s.dropna().nunique()),
        "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
        "median": float(numeric.median()) if numeric.notna().any() else np.nan,
        "std": float(numeric.std()) if numeric.notna().sum() > 1 else np.nan,
        "iqr": numeric_iqr(s),
        "min": float(numeric.min()) if numeric.notna().any() else np.nan,
        "max": float(numeric.max()) if numeric.notna().any() else np.nan,
    }


def make_ecg_quality_report(raw: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    group_cols = [schema.get("phase", "phase"), schema.get("recording", "record_type"), schema.get("segment", "segment_label")]
    hrv = schema.get("hrv_valid", "hrv_validated")
    morph = schema.get("morphology_valid", "morphology_validated")
    rows = []
    for keys, g in raw.groupby(group_cols, dropna=False):
        row = dict(zip(["phase", "record_type", "segment_label"], keys if isinstance(keys, tuple) else (keys,)))
        row["n_total"] = int(len(g))
        row["n_dates"] = int(g["date"].nunique()) if "date" in g else np.nan
        if hrv in g:
            vals = _as_bool(g[hrv])
            row["hrv_valid_n"] = int(vals.sum())
            row["hrv_valid_ratio"] = float(vals.mean()) if len(vals) else np.nan
        if morph in g:
            vals = _as_bool(g[morph])
            row["morphology_valid_n"] = int(vals.sum())
            row["morphology_valid_ratio"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "record_type", "segment_label"], ignore_index=True)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "valid"])


def make_context_completeness_report(context: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame()
    cols = columns or [c for c in context.columns if c != "date"]
    rows = []
    for c in cols:
        if c not in context.columns:
            rows.append({"column": c, "present": False, "n_total": len(context), "n_valid": 0, "valid_ratio": 0.0, "reason": "missing_column"})
            continue
        row = {"column": c, "present": True, **summarize_series(context[c])}
        row["reason"] = "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def make_derived_operations_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_id in ["ecg_derived", "context_derived"]:
        df = tables.get(table_id, pd.DataFrame())
        if df.empty:
            continue
        gb_cols = ["operation_id", "operation_type", "formula", "base_formula"]
        present = [c for c in gb_cols if c in df.columns]
        for keys, g in df.groupby(present, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"table": table_id, **dict(zip(present, keys))}
            row["n_rows"] = int(len(g))
            row["n_dates"] = int(g["date"].nunique()) if "date" in g.columns else np.nan
            row["n_features"] = int(g["feature"].nunique()) if "feature" in g.columns else np.nan
            if "valid" in g.columns:
                valid = _as_bool(g["valid"])
                row["valid_n"] = int(valid.sum())
                row["valid_ratio"] = float(valid.mean()) if len(valid) else np.nan
            if "reason" in g.columns:
                reason = g.loc[g["reason"].notna(), "reason"].astype(str)
                row["main_invalid_reason"] = reason.value_counts().index[0] if not reason.empty else ""
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["table", "operation_id", "formula"], ignore_index=True)


def make_target_report(dataset: pd.DataFrame, targets: list[dict[str, Any]], target_cols: dict[str, str], rules: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for target in targets:
        tid = target["id"]
        col = target_cols.get(tid)
        if col not in dataset.columns:
            rows.append({"target": tid, "column": col, "present": False, "usable": False, "reason": "missing_target_column"})
            continue
        stats = summarize_series(dataset[col])
        reason = []
        if stats["n_valid"] < rules.get("min_valid_n", 0):
            reason.append("low_valid_n")
        if stats["valid_ratio"] < rules.get("min_valid_ratio", 0):
            reason.append("low_valid_ratio")
        if stats["n_unique"] < rules.get("min_unique_values", 0):
            reason.append("low_unique_values")
        if not np.isnan(stats["iqr"]) and stats["iqr"] <= rules.get("min_iqr", 0):
            reason.append("low_iqr")
        rows.append({
            "target": tid,
            "column": col,
            "label": target.get("label", ""),
            "direction": target.get("direction", ""),
            "horizon": target.get("horizon", ""),
            "data_type": target.get("data_type", ""),
            "scale_type": target.get("scale_type", ""),
            "present": True,
            **stats,
            "usable": not reason,
            "reason": ";".join(reason) if reason else "ok",
        })
    return pd.DataFrame(rows)


def make_feature_availability_report(dataset: pd.DataFrame, catalog: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    rows = []
    meta_cols = set(catalog.loc[catalog["role"].isin(["meta", "qc", "target"]), "column"]) if not catalog.empty else set()
    for _, item in catalog.iterrows():
        col = item["column"]
        if col in meta_cols or col not in dataset.columns:
            continue
        stats = summarize_series(dataset[col])
        reason = []
        if stats["n_valid"] < rules.get("min_valid_n", 0):
            reason.append("low_valid_n")
        if stats["valid_ratio"] < rules.get("min_valid_ratio", 0):
            reason.append("low_valid_ratio")
        if stats["n_unique"] < rules.get("min_unique_values", 0):
            reason.append("low_unique_values")
        if not np.isnan(stats["iqr"]) and stats["iqr"] <= rules.get("min_iqr", 0):
            reason.append("low_iqr")
        rows.append({
            "feature": col,
            "role": item.get("role", ""),
            "source_table": item.get("source_table", ""),
            "feature_family": item.get("feature_family", ""),
            **stats,
            "usable": not reason,
            "reason": ";".join(reason) if reason else "ok",
        })
    return pd.DataFrame(rows)


def make_feature_target_overlap_report(dataset: pd.DataFrame, feature_report: pd.DataFrame, target_report: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    rows = []
    if feature_report.empty or target_report.empty:
        return pd.DataFrame()
    total_n = len(dataset)
    for _, t in target_report.iterrows():
        tcol = t.get("column")
        if tcol not in dataset.columns:
            continue
        tvalid = dataset[tcol].notna()
        for _, f in feature_report.iterrows():
            fcol = f["feature"]
            if fcol not in dataset.columns:
                continue
            overlap = (tvalid & dataset[fcol].notna()).sum()
            ratio = float(overlap / total_n) if total_n else 0.0
            reason = []
            if overlap < rules.get("min_overlap_n", 0):
                reason.append("low_overlap_n")
            if ratio < rules.get("min_overlap_ratio", 0):
                reason.append("low_overlap_ratio")
            rows.append({
                "target": t["target"],
                "target_column": tcol,
                "feature": fcol,
                "n_target_valid": int(tvalid.sum()),
                "n_feature_valid": int(dataset[fcol].notna().sum()),
                "n_overlap": int(overlap),
                "overlap_ratio": ratio,
                "usable": not reason,
                "reason": ";".join(reason) if reason else "ok",
            })
    return pd.DataFrame(rows)


def make_ml_readiness_report(target_report: pd.DataFrame, overlap_report: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    rows = []
    if target_report.empty:
        return pd.DataFrame()
    for _, t in target_report.iterrows():
        target = t["target"]
        subset = overlap_report[(overlap_report["target"] == target) & (overlap_report["usable"])] if not overlap_report.empty else pd.DataFrame()
        n_samples = int(t.get("n_valid", 0) or 0)
        n_features = int(subset["feature"].nunique()) if not subset.empty else 0
        ratio = float(n_features / n_samples) if n_samples else np.nan
        warnings = []
        if n_samples < rules.get("min_samples_regression", 0):
            warnings.append("too_few_samples_for_regression")
        if n_samples and ratio > rules.get("max_features_to_samples_ratio", 999):
            warnings.append("too_many_features_for_samples")
        if not t.get("usable", False):
            warnings.append("target_not_usable")
        rows.append({
            "target": target,
            "n_samples": n_samples,
            "n_usable_features_by_overlap": n_features,
            "feature_sample_ratio": ratio,
            "recommended_use": "descriptive_or_feature_analysis" if not warnings else "diagnostic_only",
            "warning": ";".join(warnings) if warnings else "ok",
        })
    return pd.DataFrame(rows)


def make_summary(dataset: pd.DataFrame, reports: dict[str, pd.DataFrame], warnings: list[str]) -> dict[str, Any]:
    date_min = dataset["date"].min() if "date" in dataset.columns and not dataset.empty else None
    date_max = dataset["date"].max() if "date" in dataset.columns and not dataset.empty else None
    target_report = reports.get("targets", pd.DataFrame())
    feature_report = reports.get("feature_availability", pd.DataFrame())
    return {
        "n_rows": int(len(dataset)),
        "n_columns": int(dataset.shape[1]),
        "date_min": str(date_min) if date_min is not None else None,
        "date_max": str(date_max) if date_max is not None else None,
        "n_targets": int(len(target_report)) if not target_report.empty else 0,
        "n_usable_targets": int(target_report.get("usable", pd.Series(dtype=bool)).sum()) if not target_report.empty else 0,
        "n_features_total": int(len(feature_report)) if not feature_report.empty else 0,
        "n_usable_features": int(feature_report.get("usable", pd.Series(dtype=bool)).sum()) if not feature_report.empty else 0,
        "warnings": warnings,
    }


def write_markdown_report(summary: dict[str, Any], reports: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Analysis dataset report", ""]
    lines.append("## Summary")
    for key, value in summary.items():
        if key == "warnings":
            continue
        lines.append(f"- **{key}**: {value}")
    if summary.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
    for name, df in reports.items():
        lines.append("")
        lines.append(f"## {name.replace('_', ' ').title()}")
        if df.empty:
            lines.append("No rows.")
        else:
            lines.append(df.head(20).to_markdown(index=False))
            if len(df) > 20:
                lines.append(f"\nShowing first 20 of {len(df)} rows.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
