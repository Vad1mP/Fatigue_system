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


PAIRWISE_FORMULAS: dict[str, Callable[[float, float], FormulaResult]] = {
    "delta": safe_delta,
    "percent_delta": safe_percent_delta,
    "ratio": safe_ratio,
    "log_ratio": safe_log_ratio,
}


# -----------------------------------------------------------------------------
# Feature set rules
# -----------------------------------------------------------------------------


DEFAULT_FULL_FEATURE_SET = {
    "include_prefixes": [
        "hrv_time__",
        "hrv_freq__",
        "hrv_nonlinear__",
        "morph_p__",
        "morph_qrs__",
        "morph_t__",
    ],
    "exclude_prefixes": [
        "meta__",
        "qc__",
    ],
    "exclude_contains": [
        "_idx",
        "threshold",
        "Threshold",
        "noise",
        "Noise",
        "snr",
        "SNR",
        "error",
        "Error",
    ],
    "exclude": [
        # HRV service-like counters
        "hrv_time__n_rr",
        "hrv_time__n_rpeaks",
        # generic recording / segmentation metadata
        "sampling_rate",
        "sampling_rate_hz",
        "fs",
        "start_sec",
        "end_sec",
        "duration_sec",
        "window_idx",
        "segment_idx",
        "n_samples",
        # common QC / review columns
        "n_rpeaks",
        "n_beats_extracted",
        "n_beats_good",
        "good_beats_ratio",
        "clipping_ratio",
        "rr_phys_bad_ratio",
        "suspicious_ratio",
        "rpeak_amp_bad_ratio",
        "corr_min",
        "corr_max",
        "corr_median",
        "morph_corr_threshold",
    ],
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_derived_features_with_protocol(
    root_dir: str | Path,
    protocol_config_path: str | Path,
    *,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Build protocol-defined derived ECG features from features_protocol.csv.

    Expected protocol block:

    derived_features:
      input: "features_protocol.csv"
      output: "features_derived.csv"
      defaults:
        missing_policy: "nan"
        quality_policy: "valid_only"
        baseline_include_current_day: false
        feature_set: "full"
      sources:
        sit_before:
          phase: "before"
          recording: "sit"
          segment: "full"
      operations:
        - id: "training_response"
          type: "pairwise"
          left: "long_after"
          right: "long_before"
          feature_set: "full"
          formulas: ["delta", "percent_delta"]
      baseline_profiles: {}
      baseline_deviations: []

    The input table may use either:
      - record_type / recording / recording_id
      - segment / segment_label / segment_id

    Internally everything is normalized to:
      - date
      - phase
      - record_type
      - segment
    """

    root_dir = Path(root_dir)
    protocol_config_path = Path(protocol_config_path)

    with open(protocol_config_path, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    if not isinstance(protocol, dict) or "derived_features" not in protocol:
        raise ValueError("protocol.yaml does not contain required block: derived_features")

    cfg = protocol["derived_features"]

    input_csv = root_dir / cfg["input"]
    output_csv = root_dir / cfg["output"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Input raw features CSV not found: {input_csv}")

    raw_df = pd.read_csv(input_csv)
    raw_df = normalize_raw_df(raw_df, protocol)
    print(
        raw_df[["date", "phase", "record_type", "segment"]]
        .drop_duplicates()
        .sort_values("date")
        .to_string(index=False)
    )
    validate_required_columns(raw_df)

    if debug:
        print_debug_raw_table(raw_df, cfg)
        print_debug_feature_sets(raw_df, cfg)

    operation_results: dict[str, pd.DataFrame] = {}
    result_parts: list[pd.DataFrame] = []

    for op in cfg.get("operations", []):
        op_type = op.get("type")

        if op_type == "pairwise":
            op_df = compute_pairwise_operation(raw_df, cfg, op)
        elif op_type == "pairwise_derived":
            op_df = compute_pairwise_derived_operation(operation_results, cfg, op)
        else:
            raise ValueError(f"Unsupported derived operation type: {op_type!r}")

        operation_results[op["id"]] = op_df
        result_parts.append(op_df)

    for baseline_op in cfg.get("baseline_deviations", []):
        baseline_df = compute_baseline_deviation(raw_df, cfg, baseline_op)
        operation_results[baseline_op["id"]] = baseline_df
        result_parts.append(baseline_df)

    if not result_parts:
        raise RuntimeError(
            "No derived operations were configured. Check derived_features.operations "
            "and derived_features.baseline_deviations in protocol.yaml."
        )

    derived_df = pd.concat(result_parts, ignore_index=True)

    if derived_df.empty:
        print_debug_raw_table(raw_df, cfg)
        raise RuntimeError(
            "Derived features table is empty. Most likely configured sources did not "
            "match rows in features_protocol.csv. Check phase / recording / segment names."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_csv, index=False)

    if debug:
        print(f"\n[INFO] Saved derived features: {output_csv}")
        print(f"[INFO] Output shape: {derived_df.shape}")
        print("[INFO] Valid values:")
        print(derived_df["valid"].value_counts(dropna=False).to_string())

    return derived_df


# -----------------------------------------------------------------------------
# Normalization / validation / diagnostics
# -----------------------------------------------------------------------------


def normalize_raw_df(df: pd.DataFrame, protocol: dict[str, Any] | None = None) -> pd.DataFrame:
    df = df.copy()

    # Strip column names just in case.
    df.columns = [str(c).strip() for c in df.columns]

    # date
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")

    date_format = None
    if protocol is not None:
        date_format = (
            protocol.get("storage", {})
            .get("date_format")
        )

    if date_format:
        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    else:
        # Fallback for the current project: dates are usually like 01.04.2026 = day.month.year.
        # dayfirst=True prevents pandas from interpreting this as month.day.year.
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if df["date"].isna().all():
        raise ValueError(
            "Column 'date' exists, but all values failed to parse as datetime. "
            "Check storage.date_format in protocol.yaml."
        )

    if df["date"].isna().any():
        bad_count = int(df["date"].isna().sum())
        print(f"[WARNING] {bad_count} rows have unparsed date values and will be ignored by date-based operations")

    # phase
    if "phase" not in df.columns:
        raise ValueError("Missing required column: phase")
    df["phase"] = df["phase"].astype(str).str.strip()

    # record_type compatibility
    if "record_type" not in df.columns:
        if "recording" in df.columns:
            df["record_type"] = df["recording"]
        elif "recording_id" in df.columns:
            df["record_type"] = df["recording_id"]
        elif "record" in df.columns:
            df["record_type"] = df["record"]
        else:
            raise ValueError(
                "Missing required record column. Expected one of: "
                "record_type, recording, recording_id, record"
            )

    df["record_type"] = (
        df["record_type"]
        .astype(str)
        .str.strip()
        .str.replace(".csv", "", regex=False)
    )

    # segment compatibility.
    # Your current extractor writes segment_label with values full/window/start/end.
    if "segment" not in df.columns:
        if "segment_label" in df.columns:
            df["segment"] = df["segment_label"]
        elif "segment_id" in df.columns:
            df["segment"] = df["segment_id"]
        elif "segment_name" in df.columns:
            df["segment"] = df["segment_name"]
        else:
            df["segment"] = "full"

    df["segment"] = df["segment"].astype(str).str.strip()

    empty_like = {"", "nan", "NaN", "None", "none", "null", "Null"}
    df.loc[df["segment"].isin(empty_like), "segment"] = "full"

    # Normalize common aliases without touching meaningful custom labels.
    segment_aliases = {
        "record": "full",
        "full_record": "full",
        "whole": "full",
        "all": "full",
    }
    df["segment"] = df["segment"].replace(segment_aliases)

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {"date", "phase", "record_type", "segment"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in raw features table: {sorted(missing)}")


def print_debug_raw_table(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print("\n[DEBUG] Raw features shape:", raw_df.shape)

    print("\n[DEBUG] Required source columns sample:")
    print(raw_df[["date", "phase", "record_type", "segment"]].head(20).to_string(index=False))

    print("\n[DEBUG] Available source combinations:")
    summary = (
        raw_df[["date", "phase", "record_type", "segment"]]
        .drop_duplicates()
        .sort_values(["date", "phase", "record_type", "segment"])
    )
    print(summary.to_string(index=False))

    print("\n[DEBUG] Checking configured sources:")
    for source_id, source_cfg in cfg.get("sources", {}).items():
        selected = select_source_rows(raw_df, source_cfg)
        print(
            f"  {source_id}: "
            f"phase={source_cfg.get('phase')}, "
            f"recording={source_cfg.get('recording')}, "
            f"segment={source_cfg.get('segment', 'full')} "
            f"-> rows={len(selected)}"
        )


def print_debug_feature_sets(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print("\n[DEBUG] Checking configured feature sets:")

    names = set(cfg.get("feature_sets", {}).keys())
    names.add("full")

    defaults = cfg.get("defaults", {})
    if defaults.get("feature_set"):
        names.add(defaults["feature_set"])

    for op in cfg.get("operations", []):
        if op.get("feature_set"):
            names.add(op["feature_set"])

    for op in cfg.get("baseline_deviations", []):
        if op.get("feature_set"):
            names.add(op["feature_set"])

    for name in sorted(names):
        try:
            features = resolve_feature_set(raw_df, cfg, name)
            print(f"  {name}: {len(features)} features")
            if len(features) <= 20:
                print("    ", features)
            else:
                print("    ", features[:20], "...")
        except Exception as exc:
            print(f"  {name}: ERROR -> {exc}")


# -----------------------------------------------------------------------------
# Source selection
# -----------------------------------------------------------------------------


def select_source_rows(df: pd.DataFrame, source_cfg: dict[str, Any]) -> pd.DataFrame:
    phase = str(source_cfg["phase"]).strip()
    record_type = str(source_cfg["recording"]).strip().replace(".csv", "")
    segment = str(source_cfg.get("segment", "full")).strip()

    result = df[
        (df["phase"] == phase)
        & (df["record_type"] == record_type)
        & (df["segment"] == segment)
    ].copy()

    return result


# -----------------------------------------------------------------------------
# Feature set resolution
# -----------------------------------------------------------------------------


def resolve_feature_set(df: pd.DataFrame, cfg: dict[str, Any], feature_set_name: str) -> list[str]:
    if feature_set_name == "full":
        return resolve_feature_set_by_rules(df, DEFAULT_FULL_FEATURE_SET)

    feature_sets = cfg.get("feature_sets", {})

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature_set: {feature_set_name}")

    spec = feature_sets[feature_set_name]

    if spec == "full":
        return resolve_feature_set_by_rules(df, DEFAULT_FULL_FEATURE_SET)

    if not isinstance(spec, dict):
        raise ValueError(f"Feature set '{feature_set_name}' must be a mapping, got: {type(spec)}")

    if "features" in spec:
        return validate_explicit_features(df, spec["features"], feature_set_name)

    if "include_feature_sets" in spec:
        result: list[str] = []
        for nested_name in spec["include_feature_sets"]:
            result.extend(resolve_feature_set(df, cfg, nested_name))
        return sorted(set(result))

    return resolve_feature_set_by_rules(df, spec)


def validate_explicit_features(df: pd.DataFrame, features: list[str], feature_set_name: str) -> list[str]:
    features = list(features)
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        print(
            f"[WARNING] feature_set '{feature_set_name}' contains missing features: "
            f"{missing}"
        )

    return features


def resolve_feature_set_by_rules(df: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    include_prefixes = list(spec.get("include_prefixes", []))
    exclude_prefixes = list(spec.get("exclude_prefixes", []))
    exclude_contains = list(spec.get("exclude_contains", []))
    exclude = set(spec.get("exclude", []))

    result = []

    for col in numeric_cols:
        if include_prefixes and not any(col.startswith(prefix) for prefix in include_prefixes):
            continue

        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if any(token in col for token in exclude_contains):
            continue

        if col in exclude:
            continue

        result.append(col)

    return sorted(result)


# -----------------------------------------------------------------------------
# Quality handling
# -----------------------------------------------------------------------------


def get_quality_flag(row: pd.Series, feature: str) -> tuple[bool, str]:
    """
    Checks whether a raw feature value should be allowed for derived computation.

    The extractor may use different QC column names over time. This function is
    intentionally tolerant: if the relevant QC column is absent, it does not fail.
    """

    if feature.startswith("morph_"):
        candidate_cols = [
            "morphology_validated",
            "morphology_qc_passed",
            "morphology_valid",
            "morph_valid",
            "valid_morphology",
            "is_morphology_valid",
        ]
    elif feature.startswith("hrv_"):
        candidate_cols = [
            "hrv_validated",
            "hrv_qc_passed",
            "hrv_valid",
            "signal_valid",
            "valid_hrv",
            "is_hrv_valid",
            "is_signal_valid",
        ]
    else:
        candidate_cols = [
            "valid",
            "signal_valid",
            "hrv_valid",
            "morphology_valid",
        ]

    for col in candidate_cols:
        if col not in row.index:
            continue

        value = row[col]

        # Accept common textual/bool/numeric encodings.
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "0", "no", "bad", "invalid"}:
                return False, f"{col}=false"
            if normalized in {"true", "1", "yes", "ok", "valid"}:
                continue

        if pd.isna(value):
            continue

        if value is False:
            return False, f"{col}=false"

        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) == 0.0:
            return False, f"{col}=false"

    return True, ""


# -----------------------------------------------------------------------------
# Pairwise operations
# -----------------------------------------------------------------------------


def compute_pairwise_operation(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    op: dict[str, Any],
) -> pd.DataFrame:
    defaults = cfg.get("defaults", {})
    sources = cfg.get("sources", {})

    left_source_id = op["left"]
    right_source_id = op["right"]

    if left_source_id not in sources:
        raise ValueError(f"Unknown left source '{left_source_id}' in operation '{op['id']}'")
    if right_source_id not in sources:
        raise ValueError(f"Unknown right source '{right_source_id}' in operation '{op['id']}'")

    left_df = select_source_rows(raw_df, sources[left_source_id])
    right_df = select_source_rows(raw_df, sources[right_source_id])

    if left_df.empty:
        print(f"[WARNING] Source '{left_source_id}' matched 0 rows")
    if right_df.empty:
        print(f"[WARNING] Source '{right_source_id}' matched 0 rows")

    feature_set_name = op.get("feature_set", defaults.get("feature_set", "full"))
    features = resolve_feature_set(raw_df, cfg, feature_set_name)
    formulas = list(op["formulas"])

    validate_formulas_exist(formulas)

    # Important: use all dates from raw_df, not only dates from selected sources.
    # This prevents silent empty output when source selectors are wrong.
    dates = sorted(raw_df["date"].dropna().unique())

    rows: list[dict[str, Any]] = []

    for date in dates:
        left_day = left_df[left_df["date"] == date]
        right_day = right_df[right_df["date"] == date]

        if left_day.empty or right_day.empty:
            rows.extend(
                build_missing_pairwise_rows(
                    date=date,
                    op=op,
                    left_source_id=left_source_id,
                    right_source_id=right_source_id,
                    features=features,
                    formulas=formulas,
                    reason=missing_reason(left_day.empty, right_day.empty),
                )
            )
            continue

        if len(left_day) > 1:
            print(
                f"[WARNING] Source '{left_source_id}' has {len(left_day)} rows "
                f"for date={pd.Timestamp(date).date()}; using first row"
            )
        if len(right_day) > 1:
            print(
                f"[WARNING] Source '{right_source_id}' has {len(right_day)} rows "
                f"for date={pd.Timestamp(date).date()}; using first row"
            )

        left_row = left_day.iloc[0]
        right_row = right_day.iloc[0]

        for feature in features:
            if feature not in left_row.index or feature not in right_row.index:
                rows.extend(
                    build_feature_missing_rows(
                        date=date,
                        op=op,
                        left_source_id=left_source_id,
                        right_source_id=right_source_id,
                        feature=feature,
                        formulas=formulas,
                    )
                )
                continue

            left_value = left_row[feature]
            right_value = right_row[feature]

            left_valid, left_reason = get_quality_flag(left_row, feature)
            right_valid, right_reason = get_quality_flag(right_row, feature)
            quality_ok = left_valid and right_valid
            quality_reason = ";".join(x for x in [left_reason, right_reason] if x)

            for formula in formulas:
                formula_fn = PAIRWISE_FORMULAS[formula]

                if defaults.get("quality_policy", "valid_only") == "valid_only" and not quality_ok:
                    formula_result = FormulaResult(np.nan, False, quality_reason or "invalid_quality")
                else:
                    formula_result = formula_fn(left_value, right_value)

                valid = bool(formula_result.valid and quality_ok)
                reason = "" if valid else (formula_result.reason or quality_reason or "invalid_value_or_quality")

                rows.append(
                    make_result_row(
                        date=date,
                        operation_id=op["id"],
                        operation_type=op["type"],
                        feature=feature,
                        formula=formula,
                        base_formula=None,
                        left_source=left_source_id,
                        right_source=right_source_id,
                        left_value=left_value,
                        right_value=right_value,
                        value=formula_result.value,
                        valid=valid,
                        reason=reason,
                    )
                )

    return pd.DataFrame(rows)


def validate_formulas_exist(formulas: list[str]) -> None:
    unknown = [formula for formula in formulas if formula not in PAIRWISE_FORMULAS]
    if unknown:
        raise ValueError(f"Unsupported pairwise formulas: {unknown}")


def missing_reason(left_empty: bool, right_empty: bool) -> str:
    if left_empty and right_empty:
        return "missing_left_and_right"
    if left_empty:
        return "missing_left"
    return "missing_right"


def build_missing_pairwise_rows(
    date: Any,
    op: dict[str, Any],
    left_source_id: str,
    right_source_id: str,
    features: list[str],
    formulas: list[str],
    reason: str,
) -> list[dict[str, Any]]:
    rows = []

    for feature in features:
        for formula in formulas:
            rows.append(
                make_result_row(
                    date=date,
                    operation_id=op["id"],
                    operation_type=op["type"],
                    feature=feature,
                    formula=formula,
                    base_formula=None,
                    left_source=left_source_id,
                    right_source=right_source_id,
                    left_value=np.nan,
                    right_value=np.nan,
                    value=np.nan,
                    valid=False,
                    reason=reason,
                )
            )

    return rows


def build_feature_missing_rows(
    date: Any,
    op: dict[str, Any],
    left_source_id: str,
    right_source_id: str,
    feature: str,
    formulas: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for formula in formulas:
        rows.append(
            make_result_row(
                date=date,
                operation_id=op["id"],
                operation_type=op["type"],
                feature=feature,
                formula=formula,
                base_formula=None,
                left_source=left_source_id,
                right_source=right_source_id,
                left_value=np.nan,
                right_value=np.nan,
                value=np.nan,
                valid=False,
                reason="feature_missing",
            )
        )

    return rows


# -----------------------------------------------------------------------------
# Pairwise-derived operations
# -----------------------------------------------------------------------------


def compute_pairwise_derived_operation(
    operation_results: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    op: dict[str, Any],
) -> pd.DataFrame:
    left_id = op["left_operation"]
    right_id = op["right_operation"]

    if left_id not in operation_results:
        raise ValueError(
            f"pairwise_derived operation '{op['id']}' references unknown left_operation: {left_id}"
        )
    if right_id not in operation_results:
        raise ValueError(
            f"pairwise_derived operation '{op['id']}' references unknown right_operation: {right_id}"
        )

    left_df = operation_results[left_id].copy()
    right_df = operation_results[right_id].copy()

    base_formulas = op.get("base_formulas")
    formulas = list(op["formulas"])

    validate_formulas_exist(formulas)

    if base_formulas is not None:
        left_df = left_df[left_df["formula"].isin(base_formulas)]
        right_df = right_df[right_df["formula"].isin(base_formulas)]

    merge_cols = ["date", "feature", "formula"]

    merged = left_df.merge(
        right_df,
        on=merge_cols,
        suffixes=("_left", "_right"),
    )

    rows: list[dict[str, Any]] = []

    if merged.empty:
        print(
            f"[WARNING] pairwise_derived operation '{op['id']}' produced 0 merged rows. "
            f"Check base_formulas and that both operations contain same feature/formula pairs."
        )

    for _, row in merged.iterrows():
        for formula in formulas:
            formula_fn = PAIRWISE_FORMULAS[formula]

            left_value = row["value_left"]
            right_value = row["value_right"]

            left_valid = bool(row["valid_left"])
            right_valid = bool(row["valid_right"])

            if not left_valid or not right_valid:
                formula_result = FormulaResult(np.nan, False, "invalid_left_or_right_derived")
            else:
                formula_result = formula_fn(left_value, right_value)

            rows.append(
                make_result_row(
                    date=row["date"],
                    operation_id=op["id"],
                    operation_type=op["type"],
                    feature=row["feature"],
                    formula=formula,
                    base_formula=row["formula"],
                    left_source=left_id,
                    right_source=right_id,
                    left_value=left_value,
                    right_value=right_value,
                    value=formula_result.value,
                    valid=formula_result.valid,
                    reason="" if formula_result.valid else formula_result.reason,
                )
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Baseline deviations
# -----------------------------------------------------------------------------


def compute_baseline_deviation(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    op: dict[str, Any],
) -> pd.DataFrame:
    defaults = cfg.get("defaults", {})
    sources = cfg.get("sources", {})

    source_id = op["source"]
    if source_id not in sources:
        raise ValueError(f"Unknown source '{source_id}' in baseline deviation '{op['id']}'")

    source_cfg = sources[source_id]
    source_df = select_source_rows(raw_df, source_cfg).sort_values("date").copy()

    if source_df.empty:
        print(f"[WARNING] Baseline source '{source_id}' matched 0 rows")

    feature_set_name = op.get("feature_set", defaults.get("feature_set", "full"))
    features = resolve_feature_set(raw_df, cfg, feature_set_name)

    profile_name = op["baseline_profile"]
    profile = cfg["baseline_profiles"][profile_name]

    window_days = int(profile["window_days"])
    min_periods = int(profile["min_periods"])
    statistic = profile.get("statistic", "median")
    scale = profile.get("scale", "iqr")
    include_current = bool(defaults.get("baseline_include_current_day", False))

    formulas = list(op["formulas"])
    validate_baseline_formulas(formulas)

    rows: list[dict[str, Any]] = []

    # Again: use all dates from raw_df so wrong source selectors produce diagnostic rows.
    dates = sorted(raw_df["date"].dropna().unique())

    for current_date in dates:
        current_rows = source_df[source_df["date"] == current_date]

        if current_rows.empty:
            for feature in features:
                for formula in formulas:
                    rows.append(
                        make_result_row(
                            date=current_date,
                            operation_id=op["id"],
                            operation_type="baseline_deviation",
                            feature=feature,
                            formula=formula,
                            base_formula=None,
                            left_source=source_id,
                            right_source="baseline",
                            left_value=np.nan,
                            right_value=np.nan,
                            value=np.nan,
                            valid=False,
                            reason="missing_source",
                        )
                    )
            continue

        current_row = current_rows.iloc[0]
        start_date = pd.Timestamp(current_date) - pd.Timedelta(days=window_days)

        if include_current:
            baseline_df = source_df[
                (source_df["date"] >= start_date)
                & (source_df["date"] <= current_date)
            ]
        else:
            baseline_df = source_df[
                (source_df["date"] >= start_date)
                & (source_df["date"] < current_date)
            ]

        for feature in features:
            if feature not in source_df.columns:
                for formula in formulas:
                    rows.append(
                        make_result_row(
                            date=current_date,
                            operation_id=op["id"],
                            operation_type="baseline_deviation",
                            feature=feature,
                            formula=formula,
                            base_formula=None,
                            left_source=source_id,
                            right_source="baseline",
                            left_value=np.nan,
                            right_value=np.nan,
                            value=np.nan,
                            valid=False,
                            reason="feature_missing",
                        )
                    )
                continue

            current_value = current_row[feature]
            quality_ok, quality_reason = get_quality_flag(current_row, feature)

            values = baseline_df[feature].dropna()

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
                if defaults.get("quality_policy", "valid_only") == "valid_only" and not quality_ok:
                    value = np.nan
                    valid = False
                    reason = quality_reason or "invalid_quality"
                elif not baseline_valid:
                    value = np.nan
                    valid = False
                    reason = baseline_reason
                elif formula == "delta":
                    formula_result = safe_delta(current_value, center)
                    value = formula_result.value
                    valid = formula_result.valid
                    reason = "" if valid else formula_result.reason
                elif formula == "robust_z":
                    formula_result = safe_robust_z(current_value, center, spread)
                    value = formula_result.value
                    valid = formula_result.valid
                    reason = "" if valid else formula_result.reason
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
                        left_source=source_id,
                        right_source=f"baseline:{profile_name}",
                        left_value=current_value,
                        right_value=center,
                        value=value,
                        valid=valid,
                        reason=reason,
                    )
                )

    return pd.DataFrame(rows)


def validate_baseline_formulas(formulas: list[str]) -> None:
    allowed = {"delta", "robust_z"}
    unknown = [formula for formula in formulas if formula not in allowed]
    if unknown:
        raise ValueError(f"Unsupported baseline formulas: {unknown}. Allowed: {sorted(allowed)}")


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


def safe_robust_z(value: float, center: float, spread: float) -> FormulaResult:
    if _is_missing(value) or _is_missing(center) or _is_missing(spread):
        return FormulaResult(np.nan, False, "missing_value")
    if float(spread) == 0.0:
        return FormulaResult(np.nan, False, "robust_z_zero_scale")
    return FormulaResult((float(value) - float(center)) / float(spread), True)


# -----------------------------------------------------------------------------
# Row factory
# -----------------------------------------------------------------------------


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
