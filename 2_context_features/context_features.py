from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import math

import numpy as np
import pandas as pd
import yaml


# -----------------------------------------------------------------------------
# Formula layer
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulaResult:
    value: float
    valid: bool
    reason: str = ""


def _is_missing(x: Any) -> bool:
    return pd.isna(x)


def safe_delta(left: float, right: float) -> FormulaResult:
    if _is_missing(left) or _is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")
    return FormulaResult(float(left) - float(right), True)


def safe_percent_delta(left: float, right: float) -> FormulaResult:
    if _is_missing(left) or _is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")
    if float(right) == 0.0:
        return FormulaResult(np.nan, False, "percent_delta_division_by_zero")
    return FormulaResult(100.0 * (float(left) - float(right)) / abs(float(right)), True)


def safe_ratio(left: float, right: float) -> FormulaResult:
    if _is_missing(left) or _is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")
    if float(right) == 0.0:
        return FormulaResult(np.nan, False, "ratio_division_by_zero")
    return FormulaResult(float(left) / float(right), True)


def safe_log_ratio(left: float, right: float) -> FormulaResult:
    if _is_missing(left) or _is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")
    if float(left) <= 0.0 or float(right) <= 0.0:
        return FormulaResult(np.nan, False, "log_ratio_requires_positive_values")
    return FormulaResult(math.log(float(left) / float(right)), True)


def safe_robust_z(value: float, center: float, scale: float) -> FormulaResult:
    if _is_missing(value) or _is_missing(center) or _is_missing(scale):
        return FormulaResult(np.nan, False, "missing_value")
    if float(scale) == 0.0:
        return FormulaResult(np.nan, False, "robust_z_zero_scale")
    return FormulaResult((float(value) - float(center)) / float(scale), True)


PAIRWISE_FORMULAS: dict[str, Callable[[float, float], FormulaResult]] = {
    "delta": safe_delta,
    "percent_delta": safe_percent_delta,
    "ratio": safe_ratio,
    "log_ratio": safe_log_ratio,
}


BASELINE_FORMULAS = {"delta", "robust_z"}
ROLLING_AGGREGATIONS = {"sum", "mean", "median", "std", "min", "max"}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_context_features_with_protocol(
    root_dir: str | Path,
    protocol_config_path: str | Path,
    *,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Build derived context/manual features from a daily CSV table.

    This block is intended for non-ECG parameters: subjective scores, blood
    pressure, reaction time, sleep, training load and watch/manual training data.

    Expected YAML block:

    context_features:
      input: "daily_data_extended.csv"
      output: "context_derived.csv"
      parsing:
        sep: ";"
        decimal: ","
      date_column: "Date"
      feature_sets: {...}
      computed_columns: [...]
      operations: [...]
      baseline_profiles: {...}
    """

    root_dir = Path(root_dir)
    protocol_config_path = Path(protocol_config_path)

    with open(protocol_config_path, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    if not isinstance(protocol, dict) or "context_features" not in protocol:
        raise ValueError("protocol.yaml does not contain required block: context_features")

    cfg = protocol["context_features"]

    input_csv = root_dir / cfg["input"]
    output_csv = root_dir / cfg["output"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Input context CSV not found: {input_csv}")

    df = read_context_csv(input_csv, cfg)
    df = normalize_context_df(df, protocol, cfg)
    df = apply_computed_columns(df, cfg)

    computed_output = cfg.get("computed_output")

    if computed_output:
        computed_output_csv = root_dir / computed_output
        computed_output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(computed_output_csv, index=False)

        if debug:
            print(f"\n[INFO] Saved context table with computed columns: {computed_output_csv}")
            print(f"[INFO] Computed/context table shape: {df.shape}")

    if debug:
        print_debug_context_table(df, cfg)
        print_debug_context_feature_sets(df, cfg)

    result_parts: list[pd.DataFrame] = []

    for op in cfg.get("operations", []):
        op_type = op.get("type")

        if op_type == "pairwise_columns":
            op_df = compute_pairwise_columns(df, cfg, op)
        elif op_type == "baseline_deviation":
            op_df = compute_baseline_deviation(df, cfg, op)
        elif op_type == "lagged_pairwise":
            op_df = compute_lagged_pairwise(df, cfg, op)
        elif op_type == "rolling_aggregate":
            op_df = compute_rolling_aggregate(df, cfg, op)
        else:
            raise ValueError(f"Unsupported context operation type: {op_type!r}")

        result_parts.append(op_df)

    if not result_parts:
        raise RuntimeError(
            "No context operations were configured. Check context_features.operations in protocol.yaml."
        )

    context_derived = pd.concat(result_parts, ignore_index=True)

    if context_derived.empty:
        raise RuntimeError(
            "Context-derived features table is empty. Check configured feature_sets, "
            "column names and operations."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    context_derived.to_csv(output_csv, index=False)

    if debug:
        print(f"\n[INFO] Saved context-derived features: {output_csv}")
        print(f"[INFO] Output shape: {context_derived.shape}")
        print("[INFO] Valid values:")
        print(context_derived["valid"].value_counts(dropna=False).to_string())

    return context_derived


# -----------------------------------------------------------------------------
# Reading / normalization
# -----------------------------------------------------------------------------


def read_context_csv(input_csv: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    parsing = cfg.get("parsing", {})
    sep = parsing.get("sep", ";")
    encoding = parsing.get("encoding", "utf-8-sig")

    df = pd.read_csv(input_csv, sep=sep, encoding=encoding)

    # Remove unnamed/trailing empty columns produced by spreadsheet exports.
    df = df.loc[:, [not is_empty_header(col) for col in df.columns]]

    return df


def is_empty_header(col: Any) -> bool:
    text = str(col).strip()
    return text == "" or text.lower().startswith("unnamed:")


def normalize_context_df(
    df: pd.DataFrame,
    protocol: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_column = cfg.get("date_column", "Date")
    if date_column not in df.columns:
        raise ValueError(f"Missing date column in context table: {date_column}")

    date_format = cfg.get("date_format") or protocol.get("storage", {}).get("date_format")

    if date_format:
        df["date"] = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=True)

    if df["date"].isna().all():
        raise ValueError(
            "Date column exists, but all values failed to parse. "
            "Check context_features.date_column and storage.date_format/context_features.date_format."
        )

    if df["date"].isna().any():
        print(f"[WARNING] {int(df['date'].isna().sum())} context rows have unparsed dates")

    # Convert numeric-looking columns. Decimal comma is supported.
    parsing = cfg.get("parsing", {})
    decimal = parsing.get("decimal", ",")
    non_numeric_columns = set(cfg.get("non_numeric_columns", [])) | {date_column, "date"}

    for col in df.columns:
        if col in non_numeric_columns:
            continue
        df[col] = maybe_convert_numeric(df[col], decimal=decimal)

    return df.sort_values("date").reset_index(drop=True)


def maybe_convert_numeric(series: pd.Series, *, decimal: str = ",") -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    original = series.copy()
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})

    # Do not try to parse clock-like values as decimals.
    nonempty = s.dropna()
    if len(nonempty) > 0 and nonempty.str.contains(":", regex=False).mean() > 0.3:
        return original

    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)

    converted = pd.to_numeric(s, errors="coerce")

    # Convert only if the majority of non-empty values are numeric.
    nonempty_count = int(nonempty.shape[0])
    if nonempty_count == 0:
        return converted

    parsed_count = int(converted.notna().sum())
    if parsed_count / nonempty_count >= 0.7:
        return converted

    return original


# -----------------------------------------------------------------------------
# Computed columns
# -----------------------------------------------------------------------------


def apply_computed_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    for spec in cfg.get("computed_columns", []):
        col_id = spec["id"]
        kind = spec["type"]

        if kind == "difference":
            df[col_id] = to_num(df, spec["left"]) - to_num(df, spec["right"])

        elif kind == "sum":
            cols = spec["columns"]
            df[col_id] = sum(to_num(df, col) for col in cols)

        elif kind == "product":
            cols = spec["columns"]
            value = to_num(df, cols[0])
            for col in cols[1:]:
                value = value * to_num(df, col)
            df[col_id] = value

        elif kind == "ratio":
            numerator = to_num(df, spec["numerator"])
            denominator = to_num(df, spec["denominator"])
            df[col_id] = numerator / denominator.replace(0, np.nan)

        elif kind == "weighted_sum":
            total = pd.Series(0.0, index=df.index)
            for col, weight in spec["weights"].items():
                total = total + to_num(df, col) * float(weight)
            df[col_id] = total

        elif kind == "composite_mean":
            components = []
            for component in spec["components"]:
                col = component["column"]
                values = to_num(df, col)

                if component.get("invert", False):
                    max_value = float(component.get("max_value", 10))
                    min_value = float(component.get("min_value", 0))
                    values = max_value + min_value - values

                weight = float(component.get("weight", 1.0))
                components.append(values * weight)

            if not components:
                df[col_id] = np.nan
            else:
                matrix = pd.concat(components, axis=1)
                df[col_id] = matrix.mean(axis=1, skipna=True)

        else:
            raise ValueError(f"Unsupported computed column type: {kind!r}")

    return df


def to_num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Missing column required for context feature computation: {column}")
    return pd.to_numeric(df[column], errors="coerce")


# -----------------------------------------------------------------------------
# Feature sets
# -----------------------------------------------------------------------------


def resolve_feature_set(df: pd.DataFrame, cfg: dict[str, Any], feature_set_name: str) -> list[str]:
    feature_sets = cfg.get("feature_sets", {})

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown context feature_set: {feature_set_name}")

    spec = feature_sets[feature_set_name]

    if "features" in spec:
        features = list(spec["features"])
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            print(f"[WARNING] context feature_set '{feature_set_name}' has missing columns: {missing}")
        return features

    if "include_feature_sets" in spec:
        result: list[str] = []
        for nested in spec["include_feature_sets"]:
            result.extend(resolve_feature_set(df, cfg, nested))
        return sorted(set(result))

    raise ValueError(f"Feature set '{feature_set_name}' must contain 'features' or 'include_feature_sets'")


# -----------------------------------------------------------------------------
# Operations
# -----------------------------------------------------------------------------


def compute_pairwise_columns(df: pd.DataFrame, cfg: dict[str, Any], op: dict[str, Any]) -> pd.DataFrame:
    formulas = list(op["formulas"])
    validate_pairwise_formulas(formulas)

    pairs = op.get("pairs", [])
    if not pairs:
        raise ValueError(f"pairwise_columns operation '{op['id']}' has no pairs")

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        date = row["date"]

        for pair in pairs:
            feature = pair["feature"]
            left_col = pair["left"]
            right_col = pair["right"]

            left_value = row[left_col] if left_col in row.index else np.nan
            right_value = row[right_col] if right_col in row.index else np.nan

            for formula in formulas:
                result = PAIRWISE_FORMULAS[formula](left_value, right_value)
                rows.append(
                    make_result_row(
                        date=date,
                        operation_id=op["id"],
                        operation_type="pairwise_columns",
                        feature=feature,
                        formula=formula,
                        base_formula=None,
                        left_source=left_col,
                        right_source=right_col,
                        left_value=left_value,
                        right_value=right_value,
                        value=result.value,
                        valid=result.valid,
                        reason="" if result.valid else result.reason,
                    )
                )

    return pd.DataFrame(rows)


def compute_baseline_deviation(df: pd.DataFrame, cfg: dict[str, Any], op: dict[str, Any]) -> pd.DataFrame:
    formulas = list(op["formulas"])
    validate_baseline_formulas(formulas)

    feature_set = op["feature_set"]
    features = resolve_feature_set(df, cfg, feature_set)

    profile_name = op["baseline_profile"]
    profile = cfg["baseline_profiles"][profile_name]

    window_days = int(profile["window_days"])
    min_periods = int(profile["min_periods"])
    statistic = profile.get("statistic", "median")
    scale = profile.get("scale", "iqr")
    include_current = bool(op.get("include_current_day", cfg.get("defaults", {}).get("baseline_include_current_day", False)))

    rows: list[dict[str, Any]] = []
    df_sorted = df.sort_values("date")

    for _, current_row in df_sorted.iterrows():
        current_date = current_row["date"]
        start_date = pd.Timestamp(current_date) - pd.Timedelta(days=window_days)

        if include_current:
            baseline_df = df_sorted[(df_sorted["date"] >= start_date) & (df_sorted["date"] <= current_date)]
        else:
            baseline_df = df_sorted[(df_sorted["date"] >= start_date) & (df_sorted["date"] < current_date)]

        for feature in features:
            if feature not in df_sorted.columns:
                for formula in formulas:
                    rows.append(missing_row(current_date, op, "baseline_deviation", feature, formula, "feature_missing"))
                continue

            values = pd.to_numeric(baseline_df[feature], errors="coerce").dropna()
            current_value = current_row[feature]

            if len(values) < min_periods:
                center = np.nan
                spread = np.nan
                baseline_valid = False
                baseline_reason = "insufficient_baseline"
            else:
                center = compute_center(values, statistic)
                spread = compute_spread(values, scale)
                baseline_valid = not pd.isna(center)
                baseline_reason = "" if baseline_valid else "invalid_baseline"

            for formula in formulas:
                if not baseline_valid:
                    result = FormulaResult(np.nan, False, baseline_reason)
                elif formula == "delta":
                    result = safe_delta(current_value, center)
                elif formula == "robust_z":
                    result = safe_robust_z(current_value, center, spread)
                else:
                    raise ValueError(f"Unsupported baseline formula: {formula}")

                rows.append(
                    make_result_row(
                        date=current_date,
                        operation_id=op["id"],
                        operation_type="baseline_deviation",
                        feature=feature,
                        formula=formula,
                        base_formula=None,
                        left_source=feature,
                        right_source=f"baseline:{profile_name}",
                        left_value=current_value,
                        right_value=center,
                        value=result.value,
                        valid=result.valid,
                        reason="" if result.valid else result.reason,
                    )
                )

    return pd.DataFrame(rows)


def compute_lagged_pairwise(df: pd.DataFrame, cfg: dict[str, Any], op: dict[str, Any]) -> pd.DataFrame:
    formulas = list(op["formulas"])
    validate_pairwise_formulas(formulas)

    feature_set = op["feature_set"]
    features = resolve_feature_set(df, cfg, feature_set)
    lag_days = int(op.get("lag_days", 1))

    rows: list[dict[str, Any]] = []
    df_sorted = df.sort_values("date").copy()

    for _, current_row in df_sorted.iterrows():
        current_date = pd.Timestamp(current_row["date"])
        previous_date = current_date - pd.Timedelta(days=lag_days)
        previous_rows = df_sorted[df_sorted["date"] == previous_date]

        for feature in features:
            if feature not in df_sorted.columns:
                for formula in formulas:
                    rows.append(missing_row(current_date, op, "lagged_pairwise", feature, formula, "feature_missing"))
                continue

            left_value = current_row[feature]

            if previous_rows.empty:
                right_value = np.nan
                for formula in formulas:
                    rows.append(
                        make_result_row(
                            date=current_date,
                            operation_id=op["id"],
                            operation_type="lagged_pairwise",
                            feature=feature,
                            formula=formula,
                            base_formula=None,
                            left_source=feature,
                            right_source=f"{feature}:lag_{lag_days}d",
                            left_value=left_value,
                            right_value=np.nan,
                            value=np.nan,
                            valid=False,
                            reason="missing_lagged_day",
                        )
                    )
                continue

            right_value = previous_rows.iloc[0][feature]

            for formula in formulas:
                result = PAIRWISE_FORMULAS[formula](left_value, right_value)
                rows.append(
                    make_result_row(
                        date=current_date,
                        operation_id=op["id"],
                        operation_type="lagged_pairwise",
                        feature=feature,
                        formula=formula,
                        base_formula=None,
                        left_source=feature,
                        right_source=f"{feature}:lag_{lag_days}d",
                        left_value=left_value,
                        right_value=right_value,
                        value=result.value,
                        valid=result.valid,
                        reason="" if result.valid else result.reason,
                    )
                )

    return pd.DataFrame(rows)


def compute_rolling_aggregate(df: pd.DataFrame, cfg: dict[str, Any], op: dict[str, Any]) -> pd.DataFrame:
    feature_set = op["feature_set"]
    features = resolve_feature_set(df, cfg, feature_set)

    window_days = int(op["window_days"])
    min_periods = int(op.get("min_periods", 1))
    aggregations = list(op["aggregations"])
    include_current = bool(op.get("include_current_day", False))

    unknown = [agg for agg in aggregations if agg not in ROLLING_AGGREGATIONS]
    if unknown:
        raise ValueError(f"Unsupported rolling aggregations: {unknown}")

    rows: list[dict[str, Any]] = []
    df_sorted = df.sort_values("date")

    for _, current_row in df_sorted.iterrows():
        current_date = pd.Timestamp(current_row["date"])
        start_date = current_date - pd.Timedelta(days=window_days)

        if include_current:
            window_df = df_sorted[(df_sorted["date"] >= start_date) & (df_sorted["date"] <= current_date)]
        else:
            window_df = df_sorted[(df_sorted["date"] >= start_date) & (df_sorted["date"] < current_date)]

        for feature in features:
            if feature not in df_sorted.columns:
                for agg in aggregations:
                    rows.append(missing_row(current_date, op, "rolling_aggregate", feature, agg, "feature_missing"))
                continue

            values = pd.to_numeric(window_df[feature], errors="coerce").dropna()

            for agg in aggregations:
                if len(values) < min_periods:
                    value = np.nan
                    valid = False
                    reason = "insufficient_window"
                else:
                    value = aggregate_values(values, agg)
                    valid = not pd.isna(value)
                    reason = "" if valid else "invalid_aggregate"

                rows.append(
                    make_result_row(
                        date=current_date,
                        operation_id=op["id"],
                        operation_type="rolling_aggregate",
                        feature=feature,
                        formula=f"rolling_{window_days}d_{agg}",
                        base_formula=None,
                        left_source=feature,
                        right_source=f"previous_{window_days}d",
                        left_value=current_row[feature] if feature in current_row.index else np.nan,
                        right_value=np.nan,
                        value=value,
                        valid=valid,
                        reason=reason,
                    )
                )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def validate_pairwise_formulas(formulas: list[str]) -> None:
    unknown = [formula for formula in formulas if formula not in PAIRWISE_FORMULAS]
    if unknown:
        raise ValueError(f"Unsupported pairwise formulas: {unknown}")


def validate_baseline_formulas(formulas: list[str]) -> None:
    unknown = [formula for formula in formulas if formula not in BASELINE_FORMULAS]
    if unknown:
        raise ValueError(f"Unsupported baseline formulas: {unknown}. Allowed: {sorted(BASELINE_FORMULAS)}")


def compute_center(values: pd.Series, statistic: str) -> float:
    if statistic == "median":
        return float(values.median())
    if statistic == "mean":
        return float(values.mean())
    raise ValueError(f"Unsupported baseline statistic: {statistic}")


def compute_spread(values: pd.Series, scale: str) -> float:
    if scale == "iqr":
        return float(values.quantile(0.75) - values.quantile(0.25))
    if scale == "std":
        return float(values.std())
    raise ValueError(f"Unsupported baseline scale: {scale}")


def aggregate_values(values: pd.Series, agg: str) -> float:
    if agg == "sum":
        return float(values.sum())
    if agg == "mean":
        return float(values.mean())
    if agg == "median":
        return float(values.median())
    if agg == "std":
        return float(values.std())
    if agg == "min":
        return float(values.min())
    if agg == "max":
        return float(values.max())
    raise ValueError(f"Unsupported aggregation: {agg}")


def missing_row(date: Any, op: dict[str, Any], operation_type: str, feature: str, formula: str, reason: str) -> dict[str, Any]:
    return make_result_row(
        date=date,
        operation_id=op["id"],
        operation_type=operation_type,
        feature=feature,
        formula=formula,
        base_formula=None,
        left_source=feature,
        right_source="",
        left_value=np.nan,
        right_value=np.nan,
        value=np.nan,
        valid=False,
        reason=reason,
    )


def make_result_row(
    *,
    date: Any,
    operation_id: str,
    operation_type: str,
    feature: str,
    formula: str,
    base_formula: str | None,
    left_source: str,
    right_source: str,
    left_value: Any,
    right_value: Any,
    value: Any,
    valid: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date).date().isoformat() if not pd.isna(date) else None,
        "operation_id": operation_id,
        "operation_type": operation_type,
        "feature": feature,
        "formula": formula,
        "base_formula": base_formula,
        "left_source": left_source,
        "right_source": right_source,
        "left_value": left_value,
        "right_value": right_value,
        "value": value,
        "valid": bool(valid),
        "reason": reason,
    }


def print_debug_context_table(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print("\n[DEBUG] Context table shape:", df.shape)
    print("\n[DEBUG] Date range:")
    print(df["date"].min(), "->", df["date"].max())
    print("\n[DEBUG] Columns:")
    print(list(df.columns))


def print_debug_context_feature_sets(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print("\n[DEBUG] Context feature sets:")
    for name in cfg.get("feature_sets", {}):
        try:
            features = resolve_feature_set(df, cfg, name)
            print(f"  {name}: {len(features)} features")
            if len(features) <= 20:
                print("    ", features)
            else:
                print("    ", features[:20], "...")
        except Exception as exc:
            print(f"  {name}: ERROR -> {exc}")
