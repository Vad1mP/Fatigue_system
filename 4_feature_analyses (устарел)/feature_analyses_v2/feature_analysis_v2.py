#!/usr/bin/env python3
"""
feature_analysis_v3.py

Simple, defensible feature-analysis module for an individual ECG/fatigue project.

Purpose
-------
1) Target analysis:
   - inspect candidate target variables;
   - normalize target direction so that "higher = worse / more fatigue";
   - export target time series;
   - compute target-target Spearman correlation matrix;
   - produce a simple HTML report for visual inspection.

2) Feature validation:
   - select input features automatically and/or manually;
   - optionally restrict inputs to ECG-only features;
   - optionally restrict inputs by access mode: all / before / after / delta;
   - remove target/meta/QC/technical columns;
   - remove target leakage;
   - filter features by data completeness and variability;
   - compute Spearman association between each feature and each normalized target;
   - compute leave-one-out direction stability;
   - rank features using a simple weighted score;
   - remove redundancy by grouping highly correlated features and keeping the best representative;
   - output validated individual features.

Dependencies
------------
Only Python standard library is required. PyYAML is optional for reading protocol.yaml.
If PyYAML is unavailable, the module runs with defaults and uses all target__ columns.

Example
-------
python feature_analysis_v3.py \
  --dataset analysis_dataset.csv \
  --catalog analysis_feature_catalog.csv \
  --protocol protocol.yaml \
  --out feature_analysis_v3
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

MISSING_STRINGS = {"", "na", "nan", "none", "null", "n/a", "#n/a", "inf", "-inf"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text_head(path: str, n: int = 4096) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read(n)


def sniff_dialect(path: str) -> csv.Dialect:
    sample = read_text_head(path)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        class Default(csv.excel):
            delimiter = ","
        return Default


def read_csv_dicts(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    dialect = sniff_dialect(path)
    with open(path, "r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv_dicts(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_value(row.get(k, "")) for k in fieldnames})


def serialize_value(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        if math.isinf(x):
            return ""
        return f"{x:.12g}"
    if isinstance(x, (list, tuple)):
        return ";".join(str(v) for v in x)
    return str(x)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if s.lower() in MISSING_STRINGS:
        return None
    # Support decimal comma if there is no decimal point.
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def default_config() -> Dict[str, Any]:
    """Safe defaults matching the minimal, defensible v3 protocol.

    These defaults are intentionally conservative and fast:
    - validate ECG features only;
    - use acute after-before targets only;
    - require a feature to pass all selected targets;
    - use Spearman + leave-one-out direction stability.
    """
    return {
        "feature_analysis": {
            "outputs": {"output_dir": "feature_analysis_v3"},
            "target_analysis": {
                "enabled": True,
                "targets": [
                    {"column": "target__subjective_strain_delta", "direction": "higher_worse", "weight": 1.0},
                    {"column": "target__fatigue_delta", "direction": "higher_worse", "weight": 1.0},
                    {"column": "target__reaction_time_delta", "direction": "higher_worse", "weight": 1.0},
                ],
            },
            "feature_scope": {
                "selection_mode": "auto",
                "ecg_only": True,
                "access_mode": "all",
                "manual_features": [],
                "exclude_roles": ["target", "meta", "qc", "technical"],
            },
            "leakage_control": {
                "enabled": True,
                "exclude_exact_target_duplicates": True,
                "exclude_same_source_operation": True,
                "forbidden_features_by_target": {},
            },
            "quality_rules": {
                "min_overlap_n": 10,
                "max_missing_ratio": 0.5,
                "min_unique_values": 4,
            },
            "association": {
                "method": "spearman",
                "min_abs_spearman": 0.3,
                "leave_one_out_direction_stability": True,
                "min_direction_stability": 0.7,
            },
            "multi_target_validation": {
                "require_pass_all_targets": True,
                "require_consistent_direction": True,
                "min_cross_target_direction_consistency": 1.0,
            },
            "redundancy": {
                "enabled": True,
                "method": "spearman",
                "threshold_abs_corr": 0.85,
                "choose_representative_by": "final_rank_score",
                "keep_redundant_alternatives": True,
            },
            "ranking": {
                "rank_score": {
                    "formula": "weighted_sum",
                    "weights": {"abs_spearman": 0.7, "direction_stability": 0.3},
                }
            },
        }
    }


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_optional(path: Optional[str]) -> Dict[str, Any]:
    # Always start from safe defaults, even if PyYAML is unavailable.
    cfg = default_config()
    if not path or not os.path.exists(path):
        return cfg
    try:
        import yaml  # type: ignore
    except Exception:
        # PyYAML is optional. Without it, the module still runs with safe defaults.
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return deep_update(cfg, data)
        return cfg
    except Exception:
        return cfg


def get_nested(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def bool_like(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "yes", "1", "on", "y"}:
        return True
    if s in {"false", "no", "0", "off", "n"}:
        return False
    return default


# -----------------------------------------------------------------------------
# Statistics: ranks, Spearman, leave-one-out stability
# -----------------------------------------------------------------------------

def rankdata(values: Sequence[float]) -> List[float]:
    """Average ranks for ties. Ranks start at 1."""
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def sign_of(v: Optional[float], eps: float = 1e-12) -> int:
    if v is None or not math.isfinite(v):
        return 0
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def leave_one_out_direction_stability(x: Sequence[float], y: Sequence[float], base_r: Optional[float]) -> Optional[float]:
    base_sign = sign_of(base_r)
    n = len(x)
    if base_sign == 0 or n < 4:
        return None
    usable = 0
    same = 0
    for i in range(n):
        xx = list(x[:i]) + list(x[i + 1:])
        yy = list(y[:i]) + list(y[i + 1:])
        r = spearman(xx, yy)
        s = sign_of(r)
        if s == 0:
            continue
        usable += 1
        if s == base_sign:
            same += 1
    if usable == 0:
        return None
    return same / usable


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


def iqr(values: Sequence[float]) -> Optional[float]:
    if len(values) < 4:
        return None
    s = sorted(values)
    # Simple percentile approximation with linear interpolation.
    return percentile(s, 75) - percentile(s, 25)


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


# -----------------------------------------------------------------------------
# Config and metadata
# -----------------------------------------------------------------------------

@dataclass
class TargetSpec:
    column: str
    direction: str = "higher_worse"
    label: str = ""
    horizon: str = ""
    weight: float = 1.0


def load_catalog(catalog_path: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    if not catalog_path or not os.path.exists(catalog_path):
        return {}, []
    rows, _ = read_csv_dicts(catalog_path)
    meta: Dict[str, Dict[str, str]] = {}
    targets: List[str] = []
    for row in rows:
        col = (row.get("column") or "").strip()
        if not col:
            continue
        meta[col] = row
        role = (row.get("role") or "").strip().lower()
        is_target = (row.get("is_target") or "").strip().lower() == "true"
        if role == "target" or is_target or col.startswith("target__"):
            targets.append(col)
    return meta, targets


def extract_target_specs(config: Dict[str, Any], catalog_meta: Dict[str, Dict[str, str]], catalog_targets: List[str], dataset_cols: List[str]) -> List[TargetSpec]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    ta = fa.get("target_analysis", {}) if isinstance(fa.get("target_analysis", {}), dict) else {}

    raw_targets = None
    # Priority 1: feature_analysis.target_analysis.targets
    if "targets" in ta:
        raw_targets = ta.get("targets")
    # Priority 2: feature_analysis.targets
    elif "targets" in fa:
        raw_targets = fa.get("targets")

    target_columns: List[Any] = []
    if raw_targets is None:
        # Default: use catalog targets if available, else all dataset target__ columns.
        target_columns = list(catalog_targets) if catalog_targets else [c for c in dataset_cols if c.startswith("target__")]
    elif isinstance(raw_targets, dict):
        # Supports:
        # targets:
        #   primary: [...]
        #   secondary: [...]
        # or:
        #   target__x:
        #     direction: higher_worse
        if "primary" in raw_targets or "secondary" in raw_targets:
            target_columns.extend(as_list(raw_targets.get("primary")))
            target_columns.extend(as_list(raw_targets.get("secondary")))
        else:
            for col, spec in raw_targets.items():
                if isinstance(spec, dict):
                    d = dict(spec)
                    d["column"] = col
                    target_columns.append(d)
                else:
                    target_columns.append(col)
    else:
        target_columns = as_list(raw_targets)

    specs: List[TargetSpec] = []
    seen = set()
    for item in target_columns:
        if isinstance(item, dict):
            col = str(item.get("column") or item.get("name") or item.get("id") or "").strip()
            if not col:
                continue
            direction = str(item.get("direction") or catalog_meta.get(col, {}).get("direction") or "higher_worse")
            label = str(item.get("label") or catalog_meta.get(col, {}).get("label") or col)
            horizon = str(item.get("horizon") or catalog_meta.get(col, {}).get("horizon") or "")
            weight = parse_float(item.get("weight")) or 1.0
        else:
            col = str(item).strip()
            direction = str(catalog_meta.get(col, {}).get("direction") or "higher_worse")
            label = str(catalog_meta.get(col, {}).get("label") or col)
            horizon = str(catalog_meta.get(col, {}).get("horizon") or "")
            weight = 1.0
        if col in seen:
            continue
        seen.add(col)
        if col in dataset_cols:
            specs.append(TargetSpec(col, direction, label, horizon, weight))
    return specs


def target_multiplier(direction: str) -> float:
    d = (direction or "").strip().lower()
    if d in {"higher_worse", "high_worse", "bigger_worse", "increases_worse", "worse_high"}:
        return 1.0
    if d in {"higher_better", "high_better", "bigger_better", "increases_better", "better_high"}:
        return -1.0
    if d in {"lower_worse", "low_worse", "smaller_worse", "decreases_worse", "worse_low"}:
        return -1.0
    if d in {"lower_better", "low_better", "smaller_better", "decreases_better", "better_low"}:
        return 1.0
    return 1.0


def normalized_target_values(rows: List[Dict[str, str]], spec: TargetSpec) -> List[Optional[float]]:
    m = target_multiplier(spec.direction)
    out: List[Optional[float]] = []
    for row in rows:
        v = parse_float(row.get(spec.column))
        out.append(None if v is None else m * v)
    return out


# -----------------------------------------------------------------------------
# Target analysis
# -----------------------------------------------------------------------------

def analyze_targets(rows: List[Dict[str, str]], date_col: str, targets: List[TargetSpec], out_dir: str) -> Dict[str, Any]:
    target_values = {t.column: normalized_target_values(rows, t) for t in targets}

    quality_rows: List[Dict[str, Any]] = []
    for t in targets:
        vals = [v for v in target_values[t.column] if v is not None]
        quality_rows.append({
            "target": t.column,
            "label": t.label,
            "direction": t.direction,
            "normalized_as": "higher_worse",
            "horizon": t.horizon,
            "weight": t.weight,
            "n_total_days": len(rows),
            "n_valid": len(vals),
            "missing_ratio": 1 - len(vals) / len(rows) if rows else "",
            "n_unique": len(set(vals)),
            "min": min(vals) if vals else "",
            "median": median(vals) if vals else "",
            "max": max(vals) if vals else "",
            "iqr": iqr(vals) if vals else "",
        })

    write_csv_dicts(
        os.path.join(out_dir, "target_quality_report.csv"),
        quality_rows,
        ["target", "label", "direction", "normalized_as", "horizon", "weight", "n_total_days", "n_valid", "missing_ratio", "n_unique", "min", "median", "max", "iqr"],
    )

    # Timeseries table: original and normalized targets.
    ts_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        out = {"date": row.get(date_col, "")}
        for t in targets:
            orig = parse_float(row.get(t.column))
            norm = target_values[t.column][i]
            out[f"{t.column}__original"] = orig
            out[f"{t.column}__normalized"] = norm
        ts_rows.append(out)
    ts_fields = ["date"] + [f"{t.column}__original" for t in targets] + [f"{t.column}__normalized" for t in targets]
    write_csv_dicts(os.path.join(out_dir, "target_timeseries.csv"), ts_rows, ts_fields)

    # Target correlation matrix, using pairwise non-missing rows.
    corr_rows: List[Dict[str, Any]] = []
    for t1 in targets:
        out: Dict[str, Any] = {"target": t1.column}
        for t2 in targets:
            xs: List[float] = []
            ys: List[float] = []
            for a, b in zip(target_values[t1.column], target_values[t2.column]):
                if a is not None and b is not None:
                    xs.append(a)
                    ys.append(b)
            r = spearman(xs, ys)
            out[t2.column] = r if r is not None else ""
        corr_rows.append(out)
    corr_fields = ["target"] + [t.column for t in targets]
    write_csv_dicts(os.path.join(out_dir, "target_correlation_matrix.csv"), corr_rows, corr_fields)

    # Pairwise target agreement report.
    pair_rows: List[Dict[str, Any]] = []
    for i, t1 in enumerate(targets):
        for t2 in targets[i + 1:]:
            xs, ys = [], []
            for a, b in zip(target_values[t1.column], target_values[t2.column]):
                if a is not None and b is not None:
                    xs.append(a)
                    ys.append(b)
            r = spearman(xs, ys)
            pair_rows.append({
                "target_a": t1.column,
                "target_b": t2.column,
                "n_overlap": len(xs),
                "spearman_r_after_direction_normalization": r if r is not None else "",
                "agreement_status": classify_target_agreement(r, len(xs)),
            })
    write_csv_dicts(
        os.path.join(out_dir, "target_agreement_report.csv"),
        pair_rows,
        ["target_a", "target_b", "n_overlap", "spearman_r_after_direction_normalization", "agreement_status"],
    )

    write_target_html(os.path.join(out_dir, "target_timeseries.html"), rows, date_col, targets, target_values, quality_rows, corr_rows)

    return {
        "n_targets": len(targets),
        "targets": [t.column for t in targets],
        "quality_rows": quality_rows,
        "pairwise_agreement_rows": pair_rows,
    }


def classify_target_agreement(r: Optional[float], n: int) -> str:
    if r is None or n < 5:
        return "not_enough_data"
    if r >= 0.5:
        return "strong_same_direction"
    if r >= 0.25:
        return "moderate_same_direction"
    if r > -0.25:
        return "weak_or_unclear"
    return "conflicting_direction"


def write_target_html(path: str, rows: List[Dict[str, str]], date_col: str, targets: List[TargetSpec], target_values: Dict[str, List[Optional[float]]], quality_rows: List[Dict[str, Any]], corr_rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    dates = [row.get(date_col, "") for row in rows]
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Target analysis</title>")
    parts.append("<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35} table{border-collapse:collapse;margin:16px 0} th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px} th{background:#f4f4f4} .chart{margin:20px 0;padding:12px;border:1px solid #ddd;border-radius:8px}.note{color:#555;max-width:900px}</style></head><body>")
    parts.append("<h1>Target analysis</h1>")
    parts.append("<p class='note'>All target values are normalized so that higher values mean worse state / stronger fatigue. Targets with direction <code>higher_better</code> are multiplied by -1 before correlations are computed.</p>")

    parts.append("<h2>Target quality</h2><table><tr>")
    q_fields = ["target", "direction", "horizon", "n_valid", "n_unique", "missing_ratio", "median", "iqr"]
    for f in q_fields:
        parts.append(f"<th>{html.escape(f)}</th>")
    parts.append("</tr>")
    for row in quality_rows:
        parts.append("<tr>")
        for f in q_fields:
            parts.append(f"<td>{html.escape(serialize_value(row.get(f, '')))}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    parts.append("<h2>Normalized target time series</h2>")
    for t in targets:
        vals = target_values[t.column]
        parts.append("<div class='chart'>")
        parts.append(f"<h3>{html.escape(t.column)}</h3>")
        parts.append(svg_line_chart(dates, vals, width=900, height=260))
        parts.append("</div>")

    parts.append("<h2>Target correlation matrix, Spearman after direction normalization</h2><table><tr><th>target</th>")
    for t in targets:
        parts.append(f"<th>{html.escape(t.column)}</th>")
    parts.append("</tr>")
    for row in corr_rows:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(str(row.get('target', '')))}</td>")
        for t in targets:
            parts.append(f"<td>{html.escape(serialize_value(row.get(t.column, '')))}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def svg_line_chart(labels: List[str], vals: List[Optional[float]], width: int = 900, height: int = 260) -> str:
    margin_l, margin_r, margin_t, margin_b = 55, 15, 20, 45
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(valid) < 2:
        return "<p>Not enough data to plot.</p>"
    ys = [v for _, v in valid]
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    n = max(len(vals), 2)

    def xy(i: int, v: float) -> Tuple[float, float]:
        x = margin_l + plot_w * (i / (n - 1))
        y = margin_t + plot_h * (1 - (v - ymin) / (ymax - ymin))
        return x, y

    points = []
    circles = []
    for i, v in valid:
        x, y = xy(i, v)
        points.append(f"{x:.2f},{y:.2f}")
        label = html.escape(labels[i] if i < len(labels) else str(i))
        circles.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3'><title>{label}: {v:.4g}</title></circle>")

    x0, y0 = margin_l, margin_t + plot_h
    x1, y1 = margin_l + plot_w, margin_t
    mid_y = margin_t + plot_h / 2
    y_mid_val = (ymin + ymax) / 2
    label_start = html.escape(labels[0]) if labels else ""
    label_end = html.escape(labels[-1]) if labels else ""
    return f"""
<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='white'/>
  <line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#999'/>
  <line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#999'/>
  <line x1='{x0}' y1='{mid_y}' x2='{x1}' y2='{mid_y}' stroke='#eee'/>
  <text x='5' y='{y1+5:.1f}' font-size='12'>{ymax:.4g}</text>
  <text x='5' y='{mid_y+5:.1f}' font-size='12'>{y_mid_val:.4g}</text>
  <text x='5' y='{y0+5:.1f}' font-size='12'>{ymin:.4g}</text>
  <text x='{x0}' y='{height-12}' font-size='12'>{label_start}</text>
  <text x='{max(x0, x1-110):.1f}' y='{height-12}' font-size='12'>{label_end}</text>
  <polyline fill='none' stroke='black' stroke-width='2' points='{' '.join(points)}'/>
  {''.join(circles)}
</svg>
"""


# -----------------------------------------------------------------------------
# Feature selection and validation
# -----------------------------------------------------------------------------

def is_ecg_feature(column: str, meta: Dict[str, str]) -> bool:
    st = (meta.get("source_table") or "").lower()
    fam = (meta.get("feature_family") or "").lower()
    role = (meta.get("role") or "").lower()
    if column.startswith("raw_ecg__") or column.startswith("ecg_der__"):
        return True
    if st in {"raw_ecg_features", "ecg_derived", "features_protocol", "features_derived"}:
        return True
    if fam.startswith("hrv") or fam.startswith("morph") or fam in {"morphology", "qrs", "p", "t"}:
        return role not in {"target", "meta", "qc", "technical"}
    return False


def infer_access_mode(column: str, meta: Dict[str, str]) -> str:
    phase = (meta.get("phase") or "").lower()
    op_type = (meta.get("operation_type") or "").lower()
    formula = (meta.get("formula") or "").lower()
    op_id = (meta.get("operation_id") or "").lower()

    text = " ".join([column.lower(), phase, op_type, formula, op_id])

    # Delta / response features first.
    if any(token in text for token in ["delta", "response", "baseline", "dev", "z", "orthostatic", "training_response"]):
        return "delta"
    if "__before__" in column or phase == "before" or "before" in op_id:
        return "before"
    if "__after__" in column or phase == "after" or "after" in op_id:
        return "after"
    return "unknown"


def select_candidate_feature_columns(dataset_cols: List[str], catalog_meta: Dict[str, Dict[str, str]], targets: List[TargetSpec], config: Dict[str, Any]) -> List[str]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    scope = fa.get("feature_scope", {}) if isinstance(fa.get("feature_scope", {}), dict) else {}

    selection_mode = str(scope.get("selection_mode") or "auto").lower()
    manual_features = [str(x) for x in as_list(scope.get("manual_features"))]
    ecg_only = bool_like(scope.get("ecg_only"), default=False)
    exclude_roles = {str(x).lower() for x in as_list(scope.get("exclude_roles"))} or {"target", "meta", "qc", "technical"}
    include_source_tables = {str(x).lower() for x in as_list(scope.get("include_source_tables"))}
    exclude_source_tables = {str(x).lower() for x in as_list(scope.get("exclude_source_tables"))}
    include_feature_families = {str(x).lower() for x in as_list(scope.get("include_feature_families"))}
    exclude_feature_families = {str(x).lower() for x in as_list(scope.get("exclude_feature_families"))}

    access_modes = [str(x).lower() for x in as_list(scope.get("access_modes"))]
    if not access_modes:
        access_modes = [str(scope.get("access_mode") or "all").lower()]
    access_modes = set(access_modes)

    target_cols = {t.column for t in targets}

    def allowed(col: str) -> bool:
        if col in target_cols:
            return False
        meta = catalog_meta.get(col, {})
        role = (meta.get("role") or "").lower()
        st = (meta.get("source_table") or "").lower()
        fam = (meta.get("feature_family") or "").lower()

        if col.lower() == "date" or role in exclude_roles:
            return False
        if col.startswith("target__"):
            return False
        if ecg_only and not is_ecg_feature(col, meta):
            return False
        if include_source_tables and st not in include_source_tables:
            return False
        if exclude_source_tables and st in exclude_source_tables:
            return False
        if include_feature_families and fam not in include_feature_families:
            return False
        if exclude_feature_families and fam in exclude_feature_families:
            return False
        mode = infer_access_mode(col, meta)
        if "all" not in access_modes and mode not in access_modes:
            return False
        return True

    auto_cols = [c for c in dataset_cols if allowed(c)]
    if selection_mode == "manual":
        cols = [c for c in manual_features if c in dataset_cols]
    elif selection_mode == "auto_plus_manual":
        cols = auto_cols + [c for c in manual_features if c in dataset_cols]
    else:
        cols = auto_cols

    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def has_target_leakage(feature: str, target: TargetSpec, catalog_meta: Dict[str, Dict[str, str]], config: Dict[str, Any]) -> Tuple[bool, str]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    lc = fa.get("leakage_control", {}) if isinstance(fa.get("leakage_control", {}), dict) else {}
    if not bool_like(lc.get("enabled"), default=True):
        return False, ""

    if feature == target.column:
        return True, "same_as_target"

    fmeta = catalog_meta.get(feature, {})
    tmeta = catalog_meta.get(target.column, {})

    # Manual forbidden list: preferred for simple and defensible leakage control.
    forbidden_by_target = lc.get("forbidden_features_by_target", {}) if isinstance(lc.get("forbidden_features_by_target", {}), dict) else {}
    forbidden = set(str(x) for x in as_list(forbidden_by_target.get(target.column)))
    if feature in forbidden:
        return True, "manual_forbidden_for_target"

    # Exact same feature/id/source operation.
    if bool_like(lc.get("exclude_exact_target_duplicates"), default=True):
        for key in ["id", "feature", "label"]:
            fv = (fmeta.get(key) or "").strip().lower()
            tv = (tmeta.get(key) or "").strip().lower()
            if fv and tv and fv == tv:
                return True, f"same_{key}_as_target"

    if bool_like(lc.get("exclude_same_source_operation"), default=True):
        fop = (fmeta.get("operation_id") or "").strip()
        top = (tmeta.get("operation_id") or "").strip()
        if fop and top and fop == top:
            return True, "same_operation_id_as_target"

    # Name-based fallback for obvious duplicates.
    short = target.column.replace("target__", "").lower()
    if short and short in feature.lower():
        # This catches exact computed target duplicates such as ctx_der__...subjective_strain_delta.
        return True, "feature_name_contains_target_name"

    return False, ""


def pair_values(rows: List[Dict[str, str]], feature: str, target_values: List[Optional[float]]) -> Tuple[List[float], List[float], int, int, int]:
    xs: List[float] = []
    ys: List[float] = []
    feature_valid = 0
    target_valid = 0
    for row, tv in zip(rows, target_values):
        fv = parse_float(row.get(feature))
        if fv is not None:
            feature_valid += 1
        if tv is not None:
            target_valid += 1
        if fv is not None and tv is not None:
            xs.append(fv)
            ys.append(tv)
    return xs, ys, feature_valid, target_valid, len(xs)


def analyze_feature_target_associations(rows: List[Dict[str, str]], dataset_cols: List[str], catalog_meta: Dict[str, Dict[str, str]], targets: List[TargetSpec], candidate_cols: List[str], config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    q = fa.get("quality_rules", {}) if isinstance(fa.get("quality_rules", {}), dict) else {}
    assoc = fa.get("association", {}) if isinstance(fa.get("association", {}), dict) else {}
    ranking = fa.get("ranking", {}) if isinstance(fa.get("ranking", {}), dict) else {}
    mtv = fa.get("multi_target_validation", {}) if isinstance(fa.get("multi_target_validation", {}), dict) else {}

    min_overlap_n = int(q.get("min_overlap_n", 10))
    max_missing_ratio = float(q.get("max_missing_ratio", 0.5))
    min_unique_values = int(q.get("min_unique_values", 4))
    min_abs_spearman = float(assoc.get("min_abs_spearman", 0.3))
    min_direction_stability = float(assoc.get("min_direction_stability", 0.7))

    weights = get_nested(ranking, ["rank_score", "weights"], {}) or {}
    w_r = float(weights.get("abs_spearman", 0.7))
    w_st = float(weights.get("direction_stability", 0.3))

    target_norm = {t.column: normalized_target_values(rows, t) for t in targets}

    assoc_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    passed_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for t in targets:
        for feature in candidate_cols:
            meta = catalog_meta.get(feature, {})
            leaked, leak_reason = has_target_leakage(feature, t, catalog_meta, config)
            if leaked:
                rejected_rows.append(base_reject_row(t, feature, meta, "rejected_leakage", leak_reason))
                continue

            xs, ys, feature_valid, target_valid, n_overlap = pair_values(rows, feature, target_norm[t.column])
            missing_ratio = 1 - feature_valid / len(rows) if rows else 1.0
            n_unique = len(set(xs))

            if n_overlap < min_overlap_n:
                rejected_rows.append(base_reject_row(t, feature, meta, "rejected_low_overlap", f"n_overlap={n_overlap} < {min_overlap_n}", n_overlap, n_unique, missing_ratio))
                continue
            if missing_ratio > max_missing_ratio:
                rejected_rows.append(base_reject_row(t, feature, meta, "rejected_many_missing", f"missing_ratio={missing_ratio:.3g} > {max_missing_ratio}", n_overlap, n_unique, missing_ratio))
                continue
            if n_unique < min_unique_values:
                rejected_rows.append(base_reject_row(t, feature, meta, "rejected_low_variability", f"n_unique={n_unique} < {min_unique_values}", n_overlap, n_unique, missing_ratio))
                continue

            r = spearman(xs, ys)
            if r is None:
                rejected_rows.append(base_reject_row(t, feature, meta, "rejected_correlation_not_defined", "Spearman is undefined", n_overlap, n_unique, missing_ratio))
                continue
            abs_r = abs(r)
            direction = "positive" if r > 0 else "negative" if r < 0 else "none"
            stability = leave_one_out_direction_stability(xs, ys, r)
            stability_for_score = stability if stability is not None else 0.0
            rank_score = w_r * abs_r + w_st * stability_for_score

            row = {
                "target": t.column,
                "target_label": t.label,
                "target_direction": t.direction,
                "target_normalized_as": "higher_worse",
                "feature": feature,
                "source_table": meta.get("source_table", ""),
                "feature_family": meta.get("feature_family", ""),
                "operation_id": meta.get("operation_id", ""),
                "access_mode": infer_access_mode(feature, meta),
                "n_overlap": n_overlap,
                "n_feature_valid": feature_valid,
                "n_target_valid": target_valid,
                "n_unique": n_unique,
                "missing_ratio": missing_ratio,
                "spearman_r": r,
                "abs_spearman": abs_r,
                "effect_direction": direction,
                "direction_stability": stability,
                "rank_score": rank_score,
                "status": "passed_single_target",
                "reason": "",
            }

            if abs_r < min_abs_spearman:
                rej = dict(row)
                rej["status"] = "rejected_weak_association"
                rej["reason"] = f"abs_spearman={abs_r:.3g} < {min_abs_spearman}"
                rejected_rows.append(rej)
                assoc_rows.append(row)
                continue
            if stability is None or stability < min_direction_stability:
                rej = dict(row)
                rej["status"] = "rejected_unstable_direction"
                rej["reason"] = f"direction_stability={serialize_value(stability)} < {min_direction_stability}"
                rejected_rows.append(rej)
                assoc_rows.append(row)
                continue

            assoc_rows.append(row)
            passed_by_target[t.column].append(row)

    # Multi-target consolidation.
    validated_rows = consolidate_multi_target(rows, catalog_meta, targets, passed_by_target, config)

    # Redundancy analysis on consolidated features.
    validated_rows, redundancy_rows = apply_redundancy(rows, catalog_meta, validated_rows, config)

    assoc_fields = [
        "target", "target_label", "target_direction", "target_normalized_as", "feature", "source_table", "feature_family", "operation_id", "access_mode",
        "n_overlap", "n_feature_valid", "n_target_valid", "n_unique", "missing_ratio", "spearman_r", "abs_spearman", "effect_direction", "direction_stability", "rank_score", "status", "reason",
    ]
    write_csv_dicts(os.path.join(out_dir, "feature_target_associations.csv"), assoc_rows, assoc_fields)
    write_csv_dicts(os.path.join(out_dir, "rejected_features.csv"), rejected_rows, assoc_fields)

    validated_fields = [
        "feature", "source_table", "feature_family", "operation_id", "access_mode",
        "n_targets_total", "n_targets_passed", "targets_passed", "targets_failed", "cross_target_direction", "cross_target_direction_consistency",
        "mean_abs_spearman", "min_abs_spearman", "mean_direction_stability", "min_direction_stability", "mean_rank_score", "final_rank_score",
        "redundancy_cluster_id", "is_cluster_representative", "candidate_status", "reason", "interpretation",
    ]
    write_csv_dicts(os.path.join(out_dir, "validated_individual_features.csv"), validated_rows, validated_fields)
    # Compatibility with v1 naming.
    write_csv_dicts(os.path.join(out_dir, "candidate_individual_features.csv"), validated_rows, validated_fields)
    write_csv_dicts(os.path.join(out_dir, "redundancy_clusters.csv"), redundancy_rows, ["redundancy_cluster_id", "feature", "final_rank_score", "is_cluster_representative"])

    return {
        "n_candidate_input_columns": len(candidate_cols),
        "n_association_rows": len(assoc_rows),
        "n_rejected_rows": len(rejected_rows),
        "n_validated_rows": len(validated_rows),
        "n_final_candidates": sum(1 for r in validated_rows if r.get("candidate_status") == "validated_feature"),
    }


def base_reject_row(target: TargetSpec, feature: str, meta: Dict[str, str], status: str, reason: str, n_overlap: Any = "", n_unique: Any = "", missing_ratio: Any = "") -> Dict[str, Any]:
    return {
        "target": target.column,
        "target_label": target.label,
        "target_direction": target.direction,
        "target_normalized_as": "higher_worse",
        "feature": feature,
        "source_table": meta.get("source_table", ""),
        "feature_family": meta.get("feature_family", ""),
        "operation_id": meta.get("operation_id", ""),
        "access_mode": infer_access_mode(feature, meta),
        "n_overlap": n_overlap,
        "n_feature_valid": "",
        "n_target_valid": "",
        "n_unique": n_unique,
        "missing_ratio": missing_ratio,
        "spearman_r": "",
        "abs_spearman": "",
        "effect_direction": "",
        "direction_stability": "",
        "rank_score": "",
        "status": status,
        "reason": reason,
    }


def consolidate_multi_target(rows: List[Dict[str, str]], catalog_meta: Dict[str, Dict[str, str]], targets: List[TargetSpec], passed_by_target: Dict[str, List[Dict[str, Any]]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    mtv = fa.get("multi_target_validation", {}) if isinstance(fa.get("multi_target_validation", {}), dict) else {}

    require_pass_all = bool_like(mtv.get("require_pass_all_targets"), default=True)
    require_consistent_direction = bool_like(mtv.get("require_consistent_direction"), default=True)
    min_consistency = float(mtv.get("min_cross_target_direction_consistency", 1.0))
    min_targets_pass = int(mtv.get("min_targets_pass", len(targets) if require_pass_all else 1))

    by_feature: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for target_col, target_rows in passed_by_target.items():
        for row in target_rows:
            by_feature[row["feature"]][target_col] = row

    out: List[Dict[str, Any]] = []
    target_cols = [t.column for t in targets]

    for feature, rows_by_target in by_feature.items():
        passed_targets = [t for t in target_cols if t in rows_by_target]
        failed_targets = [t for t in target_cols if t not in rows_by_target]
        if len(passed_targets) < min_targets_pass:
            continue
        if require_pass_all and failed_targets:
            continue

        signs = [sign_of(rows_by_target[t]["spearman_r"]) for t in passed_targets]
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        if pos >= neg:
            main_sign = 1
            direction = "positive"
            consistency = pos / len(signs) if signs else 0.0
        else:
            main_sign = -1
            direction = "negative"
            consistency = neg / len(signs) if signs else 0.0

        if require_consistent_direction and consistency < min_consistency:
            continue

        vals_abs = [float(rows_by_target[t]["abs_spearman"]) for t in passed_targets]
        vals_st = [float(rows_by_target[t]["direction_stability"] or 0.0) for t in passed_targets]
        vals_rank = [float(rows_by_target[t]["rank_score"] or 0.0) for t in passed_targets]
        meta = catalog_meta.get(feature, {})

        mean_abs = sum(vals_abs) / len(vals_abs)
        mean_st = sum(vals_st) / len(vals_st)
        mean_rank = sum(vals_rank) / len(vals_rank)
        # Simple conservative final score: average score times direction consistency.
        final_score = mean_rank * consistency

        interpretation = make_interpretation(feature, direction, passed_targets)
        out.append({
            "feature": feature,
            "source_table": meta.get("source_table", ""),
            "feature_family": meta.get("feature_family", ""),
            "operation_id": meta.get("operation_id", ""),
            "access_mode": infer_access_mode(feature, meta),
            "n_targets_total": len(targets),
            "n_targets_passed": len(passed_targets),
            "targets_passed": ";".join(passed_targets),
            "targets_failed": ";".join(failed_targets),
            "cross_target_direction": direction,
            "cross_target_direction_consistency": consistency,
            "mean_abs_spearman": mean_abs,
            "min_abs_spearman": min(vals_abs),
            "mean_direction_stability": mean_st,
            "min_direction_stability": min(vals_st),
            "mean_rank_score": mean_rank,
            "final_rank_score": final_score,
            "redundancy_cluster_id": "",
            "is_cluster_representative": "",
            "candidate_status": "passed_multi_target",
            "reason": "",
            "interpretation": interpretation,
        })

    out.sort(key=lambda r: float(r.get("final_rank_score") or 0.0), reverse=True)
    return out


def make_interpretation(feature: str, direction: str, targets: List[str]) -> str:
    if direction == "positive":
        return f"Higher feature values are associated with worse normalized target values across: {', '.join(targets)}."
    if direction == "negative":
        return f"Lower feature values are associated with worse normalized target values across: {', '.join(targets)}."
    return f"No stable direction across: {', '.join(targets)}."


# -----------------------------------------------------------------------------
# Redundancy grouping
# -----------------------------------------------------------------------------

def apply_redundancy(rows: List[Dict[str, str]], catalog_meta: Dict[str, Dict[str, str]], validated_rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fa = config.get("feature_analysis", {}) if isinstance(config.get("feature_analysis", {}), dict) else {}
    red = fa.get("redundancy", {}) if isinstance(fa.get("redundancy", {}), dict) else {}
    if not bool_like(red.get("enabled"), default=True) or len(validated_rows) <= 1:
        for r in validated_rows:
            r["redundancy_cluster_id"] = "cluster_001"
            r["is_cluster_representative"] = True
            r["candidate_status"] = "validated_feature"
        return validated_rows, [{"redundancy_cluster_id": "cluster_001", "feature": r["feature"], "final_rank_score": r["final_rank_score"], "is_cluster_representative": True} for r in validated_rows]

    threshold = float(red.get("threshold_abs_corr", 0.85))
    features = [r["feature"] for r in validated_rows]
    idx = {f: i for i, f in enumerate(features)}
    adj: Dict[str, List[str]] = {f: [] for f in features}

    # Pairwise feature-feature Spearman with pairwise nonmissing rows.
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            xs, ys = [], []
            for row in rows:
                a = parse_float(row.get(f1))
                b = parse_float(row.get(f2))
                if a is not None and b is not None:
                    xs.append(a)
                    ys.append(b)
            r = spearman(xs, ys)
            if r is not None and abs(r) >= threshold:
                adj[f1].append(f2)
                adj[f2].append(f1)

    # Connected components.
    components: List[List[str]] = []
    seen = set()
    for f in features:
        if f in seen:
            continue
        q = deque([f])
        seen.add(f)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        components.append(comp)

    by_feature = {r["feature"]: r for r in validated_rows}
    redundancy_rows: List[Dict[str, Any]] = []
    for ci, comp in enumerate(components, start=1):
        cluster_id = f"cluster_{ci:03d}"
        representative = max(comp, key=lambda f: float(by_feature[f].get("final_rank_score") or 0.0))
        for f in comp:
            rr = by_feature[f]
            is_rep = f == representative
            rr["redundancy_cluster_id"] = cluster_id
            rr["is_cluster_representative"] = is_rep
            rr["candidate_status"] = "validated_feature" if is_rep else "redundant_alternative"
            rr["reason"] = "" if is_rep else f"Redundant with representative {representative}"
            redundancy_rows.append({
                "redundancy_cluster_id": cluster_id,
                "feature": f,
                "final_rank_score": rr.get("final_rank_score", ""),
                "is_cluster_representative": is_rep,
            })

    validated_rows.sort(key=lambda r: (0 if r.get("candidate_status") == "validated_feature" else 1, -float(r.get("final_rank_score") or 0.0)))
    return validated_rows, redundancy_rows


# -----------------------------------------------------------------------------
# Default protocol block generation
# -----------------------------------------------------------------------------

DEFAULT_PROTOCOL_BLOCK = """feature_analysis:
  enabled: true

  inputs:
    dataset: "analysis_dataset.csv"
    feature_catalog: "analysis_feature_catalog.csv"

  outputs:
    output_dir: "feature_analysis_v3"

  # Target diagnostics are run before feature validation.
  target_analysis:
    enabled: true
    # If omitted, all target__ columns from analysis_feature_catalog.csv are used.
    # You can also define targets manually here.
    targets:
      - column: "target__subjective_strain_delta"
        direction: "higher_worse"
        weight: 1.0
      - column: "target__fatigue_delta"
        direction: "higher_worse"
        weight: 1.0
      - column: "target__reaction_time_delta"
        direction: "higher_worse"
        weight: 1.0
      # Use a separate target group if readiness_before is conceptually pre-training.
      # - column: "target__readiness_before"
      #   direction: "higher_better"
      #   weight: 1.0

  feature_scope:
    selection_mode: "auto"      # auto | manual | auto_plus_manual
    ecg_only: true              # true = validate ECG features only
    access_mode: "all"          # all | before | after | delta
    # access_modes: ["before", "delta"]
    manual_features: []
    exclude_roles:
      - "target"
      - "meta"
      - "qc"
      - "technical"

  leakage_control:
    enabled: true
    exclude_exact_target_duplicates: true
    exclude_same_source_operation: true
    forbidden_features_by_target: {}

  quality_rules:
    min_overlap_n: 10
    max_missing_ratio: 0.5
    min_unique_values: 4

  association:
    method: "spearman"
    min_abs_spearman: 0.3
    leave_one_out_direction_stability: true
    min_direction_stability: 0.7

  multi_target_validation:
    # Strict mode: feature must pass all selected targets and keep one direction.
    require_pass_all_targets: true
    require_consistent_direction: true
    min_cross_target_direction_consistency: 1.0
    # Softer alternative:
    # require_pass_all_targets: false
    # min_targets_pass: 2
    # min_cross_target_direction_consistency: 0.67

  redundancy:
    enabled: true
    method: "spearman"
    threshold_abs_corr: 0.85
    choose_representative_by: "final_rank_score"
    keep_redundant_alternatives: true

  ranking:
    rank_score:
      formula: "weighted_sum"
      weights:
        abs_spearman: 0.7
        direction_stability: 0.3
"""


def write_default_protocol_block(path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_PROTOCOL_BLOCK)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple target and ECG feature validation module.")
    parser.add_argument("--dataset", required=True, help="Path to analysis_dataset.csv")
    parser.add_argument("--catalog", required=True, help="Path to analysis_feature_catalog.csv")
    parser.add_argument("--protocol", default=None, help="Path to protocol.yaml")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--write-default-protocol-block", default=None, help="Write example feature_analysis protocol block and exit")
    args = parser.parse_args()

    if args.write_default_protocol_block:
        write_default_protocol_block(args.write_default_protocol_block)
        print(f"Wrote default protocol block to: {args.write_default_protocol_block}")
        return

    config = load_yaml_optional(args.protocol)
    rows, dataset_cols = read_csv_dicts(args.dataset)
    catalog_meta, catalog_targets = load_catalog(args.catalog)

    out_dir = args.out or get_nested(config, ["feature_analysis", "outputs", "output_dir"], "feature_analysis_v3")
    ensure_dir(out_dir)

    date_col = "date" if "date" in dataset_cols else dataset_cols[0]
    targets = extract_target_specs(config, catalog_meta, catalog_targets, dataset_cols)
    if not targets:
        raise RuntimeError("No target columns found. Add target__ columns or specify feature_analysis.target_analysis.targets in protocol.yaml.")

    target_summary = analyze_targets(rows, date_col, targets, out_dir)
    candidate_cols = select_candidate_feature_columns(dataset_cols, catalog_meta, targets, config)
    feature_summary = analyze_feature_target_associations(rows, dataset_cols, catalog_meta, targets, candidate_cols, config, out_dir)

    summary = {
        "dataset": args.dataset,
        "catalog": args.catalog,
        "protocol": args.protocol,
        "output_dir": out_dir,
        "n_days": len(rows),
        "n_dataset_columns": len(dataset_cols),
        "target_analysis": {
            "n_targets": target_summary["n_targets"],
            "targets": target_summary["targets"],
        },
        "feature_analysis": feature_summary,
        "notes": [
            "Targets are normalized so that higher target values mean worse state / stronger fatigue.",
            "Spearman correlation is used as a rank-based monotonic association measure.",
            "Direction stability is computed by leave-one-out resampling.",
            "Redundancy clusters are based on absolute Spearman correlation between validated features.",
        ],
    }
    with open(os.path.join(out_dir, "feature_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
