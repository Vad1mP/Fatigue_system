from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import warnings

import numpy as np
import pandas as pd
import yaml

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
except Exception:  # pragma: no cover
    mutual_info_regression = None
    mutual_info_classif = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_feature_analysis_with_protocol(
    root_dir: str | Path,
    protocol_config_path: str | Path,
    *,
    screening_ids: list[str] | None = None,
    debug: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run exploratory feature analysis defined in protocol.yaml.

    This module is intentionally placed before ml_dataset_builder:
      - it does not train final ML models;
      - it ranks candidate features against a target;
      - it produces missingness/correlation/stability reports and diagnostic plots;
      - later, ml_dataset_builder can use these rankings to build top-k datasets.

    Expected protocol block, simplified:

    feature_analysis:
      output_dir: "feature_analysis"

      inputs:
        ecg_raw:
          type: "ecg_raw"
          path: "features_protocol.csv"
        ecg_derived:
          type: "derived_long"
          path: "features_derived.csv"
        context_raw:
          type: "context_raw"
          path: "daily_data_extended.csv"
          date_column: "Date"
          parsing:
            sep: ";"
            decimal: ","
            encoding: "utf-8-sig"
        context_derived:
          type: "derived_long"
          path: "context_derived.csv"

      defaults:
        min_valid_fraction: 0.5
        min_pairwise_n: 5
        min_unique_values: 4
        bootstrap_iterations: 500
        random_state: 42
        top_n_plots: 12

      targets:
        readiness_before:
          source: "context_computed"
          column: "readiness_before"
          task_type: "regression"

        subjective_strain_delta:
          source: "context_derived"
          operation_id: "subjective_before_after_response"
          feature: "subjective_strain"
          formula: "delta"
          task_type: "regression"

      feature_sets:
        ecg_orthostatic_before:
          source: "ecg_derived"
          operation_id:
            - "orthostatic_before"
          formulas:
            - "delta"
            - "percent_delta"
          valid_only: true

        context_before_objective:
          source: "context_raw"
          features:
            - "Morning_RHR"
            - "Reaction_time_before_ms"

        ecg_before_all:
          include_feature_sets:
            - "ecg_orthostatic_before"
            - "ecg_baseline_before"

      screening_sets:
        - id: "S1_readiness_ecg_before"
          target: "readiness_before"
          candidate_feature_sets:
            - "ecg_before_all"
          analyses:
            - "missingness"
            - "correlation"
            - "bootstrap_correlation"
            - "mutual_information"
            - "plots"
          ranking:
            primary_metric: "abs_spearman"
            min_valid_fraction: 0.5
            min_unique_values: 4
            min_pairwise_n: 5
          outputs:
            ranking_csv: "S1_feature_ranking.csv"
            summary_csv: "S1_summary.csv"
            plots_dir: "S1_plots"
    """

    root_dir = Path(root_dir)
    protocol_config_path = Path(protocol_config_path)

    with open(protocol_config_path, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    if not isinstance(protocol, dict) or "feature_analysis" not in protocol:
        raise ValueError("protocol.yaml does not contain required block: feature_analysis")

    cfg = protocol["feature_analysis"]
    output_dir = root_dir / cfg.get("output_dir", "feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = cfg.get("defaults", {})
    random_state = int(defaults.get("random_state", 42))

    data_sources = load_analysis_inputs(root_dir, protocol, cfg, debug=debug)

    screenings = cfg.get("screening_sets", [])
    if screening_ids is not None:
        screening_ids_set = set(screening_ids)
        screenings = [s for s in screenings if s.get("id") in screening_ids_set]

    if not screenings:
        raise RuntimeError("No feature_analysis.screening_sets configured or selected")

    results: dict[str, pd.DataFrame] = {}

    for screening in screenings:
        screening_id = screening["id"]
        if debug:
            print(f"\n[INFO] Running feature screening: {screening_id}")

        target_name_or_spec = screening["target"]
        y = resolve_target(target_name_or_spec, data_sources, protocol, cfg)

        X = resolve_candidate_feature_sets(
            screening.get("candidate_feature_sets", []),
            data_sources=data_sources,
            protocol=protocol,
            cfg=cfg,
        )

        X = apply_forbidden_rules(X, screening)
        X, y = align_X_y_by_date(X, y)

        X, y, outlier_report = apply_outlier_policy(
            X=X,
            y=y,
            screening=screening,
            defaults=defaults,
        )

        if debug:
            print(f"[DEBUG] X shape after alignment/outlier handling: {X.shape}")
            print(f"[DEBUG] y valid count: {int(y.notna().sum())}")
            if not outlier_report.empty:
                print(f"[DEBUG] Outliers handled: {len(outlier_report)}")

        ranking = analyze_feature_matrix(
            X=X,
            y=y,
            task_type=get_target_task_type(target_name_or_spec, cfg),
            screening=screening,
            defaults=defaults,
            random_state=random_state,
        )

        ranking = add_readable_feature_columns(ranking)
        ranking = add_formula_family_columns(ranking)
        ranking = add_recommendation_flags(ranking, screening, defaults)

        outputs = screening.get("outputs", {})
        ranking_path = output_dir / outputs.get("ranking_csv", f"{screening_id}_feature_ranking_technical.csv")
        report_path = output_dir / outputs.get("report_csv", f"{screening_id}_feature_report.csv")
        summary_path = output_dir / outputs.get("summary_csv", f"{screening_id}_summary.csv")
        outlier_path = output_dir / outputs.get("outlier_csv", f"{screening_id}_outliers.csv")
        plots_dir = output_dir / outputs.get("plots_dir", f"{screening_id}_plots")

        # Technical file keeps all numeric diagnostics for reproducibility.
        ranking.to_csv(ranking_path, index=False)

        # Human-readable report is the main file to inspect manually.
        make_human_ranking_report(ranking, cfg=cfg, screening=screening).to_csv(report_path, index=False)

        make_summary_csv(ranking, X, y).to_csv(summary_path, index=False)

        if not outlier_report.empty:
            outlier_report.to_csv(outlier_path, index=False)

        if "plots" in screening.get("analyses", []) or screening.get("plots", {}).get("enabled", False):
            make_screening_plots(
                X=X,
                y=y,
                ranking=ranking,
                plots_dir=plots_dir,
                screening=screening,
                defaults=defaults,
            )

        results[screening_id] = ranking

        if debug:
            print(f"[INFO] Saved technical ranking: {ranking_path}")
            print(f"[INFO] Saved human report: {report_path}")
            print(f"[INFO] Saved summary: {summary_path}")
            if not outlier_report.empty:
                print(f"[INFO] Saved outlier report: {outlier_path}")
            print_top_features(ranking)

    return results


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_analysis_inputs(
    root_dir: Path,
    protocol: dict[str, Any],
    cfg: dict[str, Any],
    *,
    debug: bool = True,
) -> dict[str, pd.DataFrame]:
    data_sources: dict[str, pd.DataFrame] = {}

    for source_id, spec in cfg.get("inputs", {}).items():
        source_type = spec["type"]
        path = root_dir / spec["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input source '{source_id}' not found: {path}")

        if source_type == "derived_long":
            df = read_derived_long(path)

        elif source_type == "context_raw":
            df = read_context_raw(path, protocol, spec)
            # Make computed context columns available as source=context_computed.
            df = apply_context_computed_columns(df, protocol)

        elif source_type == "ecg_raw":
            df = read_ecg_raw(path, protocol, spec)

        else:
            raise ValueError(f"Unsupported feature_analysis input type: {source_type!r}")

        data_sources[source_id] = df

        # context_computed is currently the same table as context_raw after applying
        # computed_columns from context_features. The alias makes target specs clearer.
        if source_id == "context_raw":
            data_sources["context_computed"] = df

        if debug:
            print(f"[DEBUG] Loaded source '{source_id}' ({source_type}): {df.shape}")

    return data_sources


def read_derived_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Derived long table must contain 'date': {path}")
    df = normalize_subject_column(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def read_ecg_raw(path: Path, protocol: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_subject_column(df, spec=spec)

    if "date" not in df.columns:
        raise ValueError(f"ECG raw table must contain 'date': {path}")

    date_format = spec.get("date_format") or protocol.get("storage", {}).get("date_format")
    if date_format:
        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if "segment" not in df.columns:
        if "segment_label" in df.columns:
            df["segment"] = df["segment_label"]
        elif "segment_id" in df.columns:
            df["segment"] = df["segment_id"]
        else:
            df["segment"] = "full"

    if "record_type" not in df.columns:
        if "recording" in df.columns:
            df["record_type"] = df["recording"]
        elif "recording_id" in df.columns:
            df["record_type"] = df["recording_id"]
        else:
            raise ValueError("ECG raw table must contain record_type/recording/recording_id")

    df["phase"] = df["phase"].astype(str).str.strip()
    df["record_type"] = df["record_type"].astype(str).str.strip().str.replace(".csv", "", regex=False)
    df["segment"] = df["segment"].astype(str).str.strip().replace(
        {"record": "full", "full_record": "full", "whole": "full", "all": "full"}
    )

    return df


def read_context_raw(path: Path, protocol: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
    parsing = spec.get("parsing", {})
    sep = parsing.get("sep", ";")
    encoding = parsing.get("encoding", "utf-8-sig")
    decimal = parsing.get("decimal", ",")

    df = pd.read_csv(path, sep=sep, encoding=encoding)
    df = df.loc[:, [not is_empty_header(c) for c in df.columns]]
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_subject_column(df, spec=spec)

    date_column = spec.get("date_column", "Date")
    if date_column not in df.columns:
        raise ValueError(f"Context raw table missing date column: {date_column}")

    date_format = spec.get("date_format") or protocol.get("storage", {}).get("date_format")
    if date_format:
        df["date"] = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=True)

    non_numeric = set(spec.get("non_numeric_columns", [])) | {date_column, "date"}

    for col in df.columns:
        if col in non_numeric:
            continue
        df[col] = maybe_convert_numeric(df[col], decimal=decimal)

    return df.sort_values(["subject_id", "date"]).reset_index(drop=True)


def normalize_subject_column(
    df: pd.DataFrame,
    *,
    spec: dict[str, Any] | None = None,
    default_subject_id: str = "default",
) -> pd.DataFrame:
    """
    Make the module N>1-ready without requiring subject_id in old N=1 files.

    If a source has no subject column, subject_id='default' is inserted.
    If a protocol source specifies subject_column, it is renamed to subject_id.
    """

    df = df.copy()
    spec = spec or {}

    subject_column = spec.get("subject_column") or spec.get("subject_id_column")

    if subject_column and subject_column in df.columns and subject_column != "subject_id":
        df = df.rename(columns={subject_column: "subject_id"})

    if "subject_id" not in df.columns:
        df["subject_id"] = default_subject_id

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df.loc[df["subject_id"].isin(["", "nan", "NaN", "None", "none", "null"]), "subject_id"] = default_subject_id

    return df


def make_subject_date_index(df: pd.DataFrame) -> pd.MultiIndex:
    if "subject_id" not in df.columns:
        df = normalize_subject_column(df)
    return pd.MultiIndex.from_arrays(
        [df["subject_id"].astype(str), pd.to_datetime(df["date"])],
        names=["subject_id", "date"],
    )


def normalize_analysis_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert legacy DatetimeIndex to subject_id/date MultiIndex."""
    result = obj.copy()
    if isinstance(result.index, pd.MultiIndex):
        return result
    result.index = pd.MultiIndex.from_arrays(
        [["default"] * len(result.index), pd.to_datetime(result.index)],
        names=["subject_id", "date"],
    )
    return result


def is_empty_header(col: Any) -> bool:
    text = str(col).strip()
    return text == "" or text.lower().startswith("unnamed:")


def maybe_convert_numeric(series: pd.Series, *, decimal: str = ",") -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    original = series.copy()
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})

    nonempty = s.dropna()
    if len(nonempty) > 0 and nonempty.str.contains(":", regex=False).mean() > 0.3:
        return original

    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)

    converted = pd.to_numeric(s, errors="coerce")
    nonempty_count = int(nonempty.shape[0])
    if nonempty_count == 0:
        return converted

    parsed_count = int(converted.notna().sum())
    if parsed_count / nonempty_count >= 0.7:
        return converted

    return original


# -----------------------------------------------------------------------------
# Context computed columns
# -----------------------------------------------------------------------------


def apply_context_computed_columns(df: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    """
    Recompute context_features.computed_columns so feature_analysis can use
    targets/features like readiness_before, sRPE_load, MAP_before, etc.
    """

    context_cfg = protocol.get("context_features", {})
    computed_columns = context_cfg.get("computed_columns", [])
    if not computed_columns:
        return df

    df = df.copy()

    for spec in computed_columns:
        col_id = spec["id"]
        kind = spec["type"]

        if kind == "difference":
            df[col_id] = to_num(df, spec["left"]) - to_num(df, spec["right"])

        elif kind == "sum":
            df[col_id] = sum(to_num(df, col) for col in spec["columns"])

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
                values = to_num(df, component["column"])
                if component.get("invert", False):
                    max_value = float(component.get("max_value", 10))
                    min_value = float(component.get("min_value", 0))
                    values = max_value + min_value - values
                components.append(values * float(component.get("weight", 1.0)))

            if components:
                df[col_id] = pd.concat(components, axis=1).mean(axis=1, skipna=True)
            else:
                df[col_id] = np.nan

        else:
            raise ValueError(f"Unsupported computed column type in context_features: {kind!r}")

    return df


def to_num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Missing column required for computed context feature: {column}")
    return pd.to_numeric(df[column], errors="coerce")


# -----------------------------------------------------------------------------
# Target resolution
# -----------------------------------------------------------------------------


def resolve_target(
    target_name_or_spec: str | dict[str, Any],
    data_sources: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.Series:
    if isinstance(target_name_or_spec, str):
        targets = cfg.get("targets", {})
        if target_name_or_spec not in targets:
            raise ValueError(f"Unknown target: {target_name_or_spec}")
        target_spec = targets[target_name_or_spec]
    else:
        target_spec = target_name_or_spec

    source_id = target_spec["source"]
    if source_id not in data_sources:
        raise ValueError(f"Target source '{source_id}' is not loaded")

    df = data_sources[source_id]

    if source_id.endswith("derived") or target_spec.get("operation_id"):
        y = select_derived_target(df, target_spec)
    else:
        col = target_spec["column"]
        if col not in df.columns:
            raise ValueError(f"Target column not found: {col}")
        y = pd.Series(
            pd.to_numeric(df[col], errors="coerce").values,
            index=make_subject_date_index(df),
            name=target_spec.get("id", col),
        )

    y = normalize_analysis_index(y)
    y = y.sort_index()
    return y


def select_derived_target(df: pd.DataFrame, target_spec: dict[str, Any]) -> pd.Series:
    required_cols = {"date", "operation_id", "feature", "formula", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Derived target source missing required columns: {sorted(missing)}")

    mask = df["operation_id"] == target_spec["operation_id"]
    mask &= df["feature"] == target_spec["feature"]
    mask &= df["formula"] == target_spec["formula"]

    if target_spec.get("base_formula") is not None and "base_formula" in df.columns:
        mask &= df["base_formula"] == target_spec["base_formula"]

    if target_spec.get("valid_only", True) and "valid" in df.columns:
        mask &= df["valid"].astype(bool)

    selected = df[mask].copy()
    if selected.empty:
        raise ValueError(f"Derived target selection returned 0 rows: {target_spec}")

    selected = normalize_subject_column(selected)
    selected["date"] = pd.to_datetime(selected["date"])
    return pd.Series(
        pd.to_numeric(selected["value"], errors="coerce").values,
        index=make_subject_date_index(selected),
        name=target_spec["feature"],
    )


def get_target_task_type(target_name_or_spec: str | dict[str, Any], cfg: dict[str, Any]) -> str:
    if isinstance(target_name_or_spec, str):
        return cfg.get("targets", {}).get(target_name_or_spec, {}).get("task_type", "regression")
    return target_name_or_spec.get("task_type", "regression")


# -----------------------------------------------------------------------------
# Feature set resolution
# -----------------------------------------------------------------------------


def resolve_candidate_feature_sets(
    feature_set_names: list[str],
    *,
    data_sources: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    if not feature_set_names:
        raise ValueError("No candidate_feature_sets configured")

    frames = []
    for name in feature_set_names:
        frames.append(resolve_feature_set(name, data_sources=data_sources, protocol=protocol, cfg=cfg))

    X = concat_feature_frames(frames)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.sort_index()
    return X


def resolve_feature_set(
    name: str,
    *,
    data_sources: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    feature_sets = cfg.get("feature_sets", {})
    if name not in feature_sets:
        raise ValueError(f"Unknown feature_analysis.feature_set: {name}")

    spec = feature_sets[name]

    if "include_feature_sets" in spec:
        frames = [
            resolve_feature_set(nested, data_sources=data_sources, protocol=protocol, cfg=cfg)
            for nested in spec["include_feature_sets"]
        ]
        return concat_feature_frames(frames)

    source_id = spec["source"]
    if source_id not in data_sources:
        raise ValueError(f"Feature set '{name}' references unloaded source: {source_id}")

    df = data_sources[source_id]
    source_type = cfg["inputs"].get(source_id, {}).get("type")

    if source_type == "derived_long":
        return resolve_derived_long_feature_set(name, df, spec)

    if source_type == "context_raw" or source_id == "context_computed":
        return resolve_wide_context_feature_set(name, df, spec)

    if source_type == "ecg_raw":
        return resolve_ecg_raw_feature_set(name, df, spec, cfg)

    raise ValueError(f"Unsupported source type for feature set '{name}': {source_type}")


def resolve_derived_long_feature_set(name: str, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    required = {"date", "operation_id", "feature", "formula", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Derived feature source missing columns: {sorted(missing)}")

    selected = df.copy()

    if "operation_id" in spec:
        operation_ids = ensure_list(spec["operation_id"])
        selected = selected[selected["operation_id"].isin(operation_ids)]

    if "features" in spec:
        selected = selected[selected["feature"].isin(ensure_list(spec["features"]))]

    if "formulas" in spec:
        selected = selected[selected["formula"].isin(ensure_list(spec["formulas"]))]

    if "base_formulas" in spec and "base_formula" in selected.columns:
        selected = selected[selected["base_formula"].isin(ensure_list(spec["base_formulas"]))]

    if spec.get("valid_only", True) and "valid" in selected.columns:
        selected = selected[selected["valid"].astype(bool)]

    if selected.empty:
        warnings.warn(f"Feature set '{name}' selected 0 derived rows", RuntimeWarning)
        return pd.DataFrame()

    selected = normalize_subject_column(selected)
    selected["date"] = pd.to_datetime(selected["date"])
    selected["feature_column"] = selected.apply(make_derived_feature_name, axis=1)

    wide = selected.pivot_table(
        index=["subject_id", "date"],
        columns="feature_column",
        values="value",
        aggfunc="first",
    )

    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide.columns = [f"{name}::{col}" for col in wide.columns]
    return wide


def make_derived_feature_name(row: pd.Series) -> str:
    parts = [str(row["operation_id"]), str(row["feature"]), str(row["formula"])]
    if "base_formula" in row.index and not pd.isna(row["base_formula"]):
        parts.append(f"base={row['base_formula']}")
    return "__".join(parts)


def resolve_wide_context_feature_set(name: str, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    features = ensure_list(spec.get("features", []))
    if not features:
        raise ValueError(f"Context feature set '{name}' has no features")

    missing = [col for col in features if col not in df.columns]
    if missing:
        warnings.warn(f"Context feature set '{name}' missing columns: {missing}", RuntimeWarning)

    existing = [col for col in features if col in df.columns]
    if not existing:
        return pd.DataFrame(index=make_subject_date_index(df))

    wide = df[["subject_id", "date", *existing]].copy()
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.set_index(["subject_id", "date"])
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide.columns = [f"{name}::{col}" for col in wide.columns]
    return wide


def resolve_ecg_raw_feature_set(name: str, df: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    selector = spec.get("selector", {})
    selected = df.copy()

    if "phase" in selector:
        selected = selected[selected["phase"].isin(ensure_list(selector["phase"]))]

    if "record_type" in selector:
        selected = selected[selected["record_type"].isin(ensure_list(selector["record_type"]))]

    if "segment" in selector:
        selected = selected[selected["segment"].isin(ensure_list(selector["segment"]))]

    if "segment_label" in selector:
        selected = selected[selected["segment"].isin(ensure_list(selector["segment_label"]))]

    if "features" in spec:
        features = ensure_list(spec["features"])
    elif "feature_set" in spec:
        features = resolve_named_raw_feature_list(spec["feature_set"], cfg)
    else:
        raise ValueError(f"ECG raw feature set '{name}' must define features or feature_set")

    existing = [f for f in features if f in selected.columns]
    missing = [f for f in features if f not in selected.columns]
    if missing:
        warnings.warn(f"ECG raw feature set '{name}' missing columns: {missing}", RuntimeWarning)

    if selected.empty or not existing:
        warnings.warn(f"ECG raw feature set '{name}' selected no data", RuntimeWarning)
        return pd.DataFrame()

    id_cols = ["subject_id", "date", "phase", "record_type", "segment"]
    long = selected[id_cols + existing].copy()
    long["date"] = pd.to_datetime(long["date"])

    melted = long.melt(
        id_vars=id_cols,
        value_vars=existing,
        var_name="feature",
        value_name="value",
    )
    melted["feature_column"] = (
        name
        + "::"
        + melted["phase"].astype(str)
        + "__"
        + melted["record_type"].astype(str)
        + "__"
        + melted["segment"].astype(str)
        + "__"
        + melted["feature"].astype(str)
    )

    wide = melted.pivot_table(index=["subject_id", "date"], columns="feature_column", values="value", aggfunc="first")
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide


def resolve_named_raw_feature_list(name: str, cfg: dict[str, Any]) -> list[str]:
    raw_feature_sets = cfg.get("raw_feature_lists", {})
    if name not in raw_feature_sets:
        raise ValueError(f"Unknown raw feature list: {name}")
    return ensure_list(raw_feature_sets[name])


def concat_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, axis=1)


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# -----------------------------------------------------------------------------
# Anti-leakage / filtering helpers
# -----------------------------------------------------------------------------


def apply_forbidden_rules(X: pd.DataFrame, screening: dict[str, Any]) -> pd.DataFrame:
    if X.empty:
        return X

    forbidden_exact = set(screening.get("forbidden_columns", []))
    forbidden_feature_sets = set(screening.get("forbidden_feature_sets", []))
    forbidden_operations = set(screening.get("forbidden_operations", []))
    forbidden_contains = ensure_list(screening.get("forbidden_contains", []))

    keep_cols = []

    for col in X.columns:
        raw_tail = col.split("::", 1)[-1]
        feature_set_prefix = col.split("::", 1)[0] if "::" in col else ""

        if col in forbidden_exact or raw_tail in forbidden_exact:
            continue

        if feature_set_prefix in forbidden_feature_sets:
            continue

        if any(token in col for token in forbidden_contains):
            continue

        # Derived column format after prefix: operation_id__feature__formula...
        operation_id = raw_tail.split("__", 1)[0]
        if operation_id in forbidden_operations:
            continue

        keep_cols.append(col)

    return X[keep_cols]


def align_X_y_by_date(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align feature matrix and target by subject_id + date.

    The function name is kept for backwards compatibility, but internally the
    index is now entity-time based. For old N=1 files without subject_id, the
    subject is automatically set to 'default'.
    """

    X = normalize_analysis_index(X)
    y = normalize_analysis_index(y)

    common_index = X.index.intersection(y.index)
    X_aligned = X.loc[common_index].sort_index()
    y_aligned = y.loc[common_index].sort_index()

    if y_aligned.index.duplicated().any():
        y_aligned = y_aligned[~y_aligned.index.duplicated(keep="first")]
        X_aligned = X_aligned.loc[y_aligned.index]

    return X_aligned, y_aligned


# -----------------------------------------------------------------------------
# Outlier handling
# -----------------------------------------------------------------------------


def apply_outlier_policy(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    screening: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Mask or winsorize outliers before statistics/plots.

    Raw source files are never modified. The cleaned values exist only in the
    analysis matrix; all changed cells are written to an outlier report.
    """

    cfg = build_outlier_config(screening, defaults)
    if not cfg.get("enabled", False):
        return X, y, pd.DataFrame()

    action = cfg.get("action", "set_nan")
    if action not in {"set_nan", "winsorize"}:
        raise ValueError(f"Unsupported outlier action: {action}")

    X_clean = X.copy()
    y_clean = y.copy()
    report_parts = []

    X_clean, x_report = handle_outliers_in_frame(X_clean, cfg=cfg, value_role="feature")
    if not x_report.empty:
        report_parts.append(x_report)

    if cfg.get("include_target", False):
        y_frame = y_clean.to_frame("__target__")
        y_frame, y_report = handle_outliers_in_frame(y_frame, cfg=cfg, value_role="target")
        y_clean = y_frame["__target__"]
        if not y_report.empty:
            report_parts.append(y_report)

    if report_parts:
        report = pd.concat(report_parts, ignore_index=True)
    else:
        report = pd.DataFrame(
            columns=[
                "value_role",
                "subject_id",
                "date",
                "feature_name",
                "original_value",
                "new_value",
                "outlier_score",
                "method",
                "threshold",
                "action",
            ]
        )

    return X_clean, y_clean, report


def build_outlier_config(screening: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    default_cfg = {
        "enabled": False,
        "method": "zscore",          # zscore | robust_mad | iqr
        "threshold": 3.0,
        "action": "set_nan",         # set_nan | winsorize
        "scope": "global",           # global | within_subject
        "include_target": False,
        "min_n": 6,
    }

    if isinstance(defaults.get("outliers"), dict):
        default_cfg.update(defaults["outliers"])

    if isinstance(screening.get("outliers"), dict):
        default_cfg.update(screening["outliers"])

    return default_cfg


def handle_outliers_in_frame(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    value_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = frame.copy()
    report_rows: list[dict[str, Any]] = []

    method = str(cfg.get("method", "zscore"))
    threshold = float(cfg.get("threshold", 3.0))
    action = str(cfg.get("action", "set_nan"))
    scope = str(cfg.get("scope", "global"))
    min_n = int(cfg.get("min_n", 6))

    for col in cleaned.columns:
        series = pd.to_numeric(cleaned[col], errors="coerce")

        if scope == "within_subject" and isinstance(series.index, pd.MultiIndex) and "subject_id" in series.index.names:
            scores = series.groupby(level="subject_id", group_keys=False).apply(
                lambda s: compute_outlier_scores(s, method=method, min_n=min_n)
            )
            bounds = series.groupby(level="subject_id", group_keys=False).apply(
                lambda s: compute_outlier_bounds(s, method=method, threshold=threshold, min_n=min_n)
            )
        else:
            scores = compute_outlier_scores(series, method=method, min_n=min_n)
            bounds = compute_outlier_bounds(series, method=method, threshold=threshold, min_n=min_n)

        outlier_mask = scores.abs() > threshold
        outlier_mask = outlier_mask.fillna(False)

        if not outlier_mask.any():
            continue

        for idx in series.index[outlier_mask]:
            original_value = series.loc[idx]

            if action == "set_nan":
                new_value = np.nan
            elif action == "winsorize":
                lower, upper = get_bounds_for_index(bounds, idx)
                new_value = min(max(original_value, lower), upper)
            else:
                raise ValueError(f"Unsupported outlier action: {action}")

            cleaned.loc[idx, col] = new_value

            subject_id, date = index_to_subject_date(idx)
            report_rows.append(
                {
                    "value_role": value_role,
                    "subject_id": subject_id,
                    "date": pd.Timestamp(date).date().isoformat() if not pd.isna(date) else None,
                    "feature_name": col,
                    "original_value": original_value,
                    "new_value": new_value,
                    "outlier_score": scores.loc[idx],
                    "method": method,
                    "threshold": threshold,
                    "action": action,
                }
            )

    return cleaned, pd.DataFrame(report_rows)


def compute_outlier_scores(series: pd.Series, *, method: str, min_n: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    values = s.dropna()

    if len(values) < min_n:
        return pd.Series(np.nan, index=s.index)

    if method == "zscore":
        center = values.mean()
        scale = values.std()
        if pd.isna(scale) or scale == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - center) / scale

    if method == "robust_mad":
        center = values.median()
        mad = (values - center).abs().median()
        scale = 1.4826 * mad
        if pd.isna(scale) or scale == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - center) / scale

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(np.nan, index=s.index)
        # Score is expressed in IQR units from the nearest quartile.
        score = pd.Series(0.0, index=s.index)
        score[s < q1] = (s[s < q1] - q1) / iqr
        score[s > q3] = (s[s > q3] - q3) / iqr
        return score

    raise ValueError(f"Unsupported outlier method: {method}")


def compute_outlier_bounds(
    series: pd.Series,
    *,
    method: str,
    threshold: float,
    min_n: int,
) -> Any:
    s = pd.to_numeric(series, errors="coerce")
    values = s.dropna()

    if len(values) < min_n:
        return (np.nan, np.nan)

    if method == "zscore":
        center = values.mean()
        scale = values.std()
        return (center - threshold * scale, center + threshold * scale)

    if method == "robust_mad":
        center = values.median()
        mad = (values - center).abs().median()
        scale = 1.4826 * mad
        return (center - threshold * scale, center + threshold * scale)

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        return (q1 - threshold * iqr, q3 + threshold * iqr)

    raise ValueError(f"Unsupported outlier method: {method}")


def get_bounds_for_index(bounds: Any, idx: Any) -> tuple[float, float]:
    # Global bounds are just a tuple.
    if isinstance(bounds, tuple):
        return bounds

    # Grouped apply may return a Series of tuples indexed like the original data.
    try:
        value = bounds.loc[idx]
        if isinstance(value, tuple):
            return value
    except Exception:
        pass

    return (np.nan, np.nan)


def index_to_subject_date(idx: Any) -> tuple[str, Any]:
    if isinstance(idx, tuple) and len(idx) >= 2:
        return str(idx[0]), idx[1]
    return "default", idx


# -----------------------------------------------------------------------------
# Analysis core
# -----------------------------------------------------------------------------


def analyze_feature_matrix(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    screening: dict[str, Any],
    defaults: dict[str, Any],
    random_state: int,
) -> pd.DataFrame:
    if X.empty:
        raise RuntimeError("Candidate feature matrix X is empty after filtering")

    min_pairwise_n = int(screening.get("ranking", {}).get("min_pairwise_n", defaults.get("min_pairwise_n", 5)))
    bootstrap_iterations = int(defaults.get("bootstrap_iterations", 500))

    rows = []

    for feature_name in X.columns:
        x = pd.to_numeric(X[feature_name], errors="coerce")
        pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()

        n_total = int(len(X))
        n_valid_x = int(x.notna().sum())
        valid_fraction = n_valid_x / n_total if n_total else np.nan
        pairwise_n = int(len(pair))
        n_unique = int(pair["x"].nunique(dropna=True)) if pairwise_n else 0

        pearson_r, pearson_p = compute_corr(pair, method="pearson", min_n=min_pairwise_n)
        spearman_r, spearman_p = compute_corr(pair, method="spearman", min_n=min_pairwise_n)

        mi = compute_mutual_information(pair, task_type=task_type, random_state=random_state, min_n=min_pairwise_n)

        boot = bootstrap_spearman(
            pair,
            n_iter=bootstrap_iterations,
            random_state=random_state,
            min_n=min_pairwise_n,
        )

        rows.append(
            {
                "feature_name": feature_name,
                "n_total": n_total,
                "n_valid_x": n_valid_x,
                "valid_fraction": valid_fraction,
                "pairwise_n": pairwise_n,
                "n_unique": n_unique,
                "mean": float(x.mean(skipna=True)) if n_valid_x else np.nan,
                "std": float(x.std(skipna=True)) if n_valid_x else np.nan,
                "min": float(x.min(skipna=True)) if n_valid_x else np.nan,
                "max": float(x.max(skipna=True)) if n_valid_x else np.nan,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "abs_pearson": abs_or_nan(pearson_r),
                "abs_spearman": abs_or_nan(spearman_r),
                "mutual_information": mi,
                **boot,
            }
        )

    ranking = pd.DataFrame(rows)

    primary_metric = screening.get("ranking", {}).get("primary_metric", "abs_spearman")
    if primary_metric not in ranking.columns:
        raise ValueError(f"Ranking primary_metric '{primary_metric}' not found in ranking columns")

    ranking = ranking.sort_values(primary_metric, ascending=False, na_position="last").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    return ranking


def compute_corr(pair: pd.DataFrame, *, method: str, min_n: int) -> tuple[float, float]:
    if len(pair) < min_n:
        return np.nan, np.nan
    if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return np.nan, np.nan

    if stats is not None:
        if method == "pearson":
            r, p = stats.pearsonr(pair["x"], pair["y"])
        elif method == "spearman":
            r, p = stats.spearmanr(pair["x"], pair["y"])
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        return float(r), float(p)

    if method == "pearson":
        r = pair["x"].corr(pair["y"], method="pearson")
    elif method == "spearman":
        r = pair["x"].corr(pair["y"], method="spearman")
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(r), np.nan


def compute_mutual_information(
    pair: pd.DataFrame,
    *,
    task_type: str,
    random_state: int,
    min_n: int,
) -> float:
    if len(pair) < min_n:
        return np.nan
    if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return np.nan

    X_arr = pair[["x"]].to_numpy(dtype=float)
    y_arr = pair["y"].to_numpy()

    try:
        if task_type == "classification":
            if mutual_info_classif is None:
                return np.nan
            return float(mutual_info_classif(X_arr, y_arr, random_state=random_state)[0])

        if mutual_info_regression is None:
            return np.nan
        return float(mutual_info_regression(X_arr, y_arr.astype(float), random_state=random_state)[0])
    except Exception:
        return np.nan


def bootstrap_spearman(
    pair: pd.DataFrame,
    *,
    n_iter: int,
    random_state: int,
    min_n: int,
) -> dict[str, float]:
    if len(pair) < min_n or pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return {
            "bootstrap_spearman_median": np.nan,
            "bootstrap_spearman_ci_low": np.nan,
            "bootstrap_spearman_ci_high": np.nan,
            "bootstrap_sign_stability": np.nan,
        }

    rng = np.random.default_rng(random_state)
    n = len(pair)
    values = []

    x = pair["x"].to_numpy(dtype=float)
    y = pair["y"].to_numpy(dtype=float)

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue

        if stats is not None:
            r, _ = stats.spearmanr(xb, yb)
        else:
            r = pd.Series(xb).corr(pd.Series(yb), method="spearman")

        if not pd.isna(r):
            values.append(float(r))

    if not values:
        return {
            "bootstrap_spearman_median": np.nan,
            "bootstrap_spearman_ci_low": np.nan,
            "bootstrap_spearman_ci_high": np.nan,
            "bootstrap_sign_stability": np.nan,
        }

    arr = np.array(values, dtype=float)
    median = float(np.nanmedian(arr))
    ci_low = float(np.nanpercentile(arr, 2.5))
    ci_high = float(np.nanpercentile(arr, 97.5))

    if median > 0:
        stability = float(np.mean(arr > 0))
    elif median < 0:
        stability = float(np.mean(arr < 0))
    else:
        stability = float(max(np.mean(arr > 0), np.mean(arr < 0)))

    return {
        "bootstrap_spearman_median": median,
        "bootstrap_spearman_ci_low": ci_low,
        "bootstrap_spearman_ci_high": ci_high,
        "bootstrap_sign_stability": stability,
    }


def abs_or_nan(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return abs(float(value))


def add_recommendation_flags(
    ranking: pd.DataFrame,
    screening: dict[str, Any],
    defaults: dict[str, Any],
) -> pd.DataFrame:
    """
    Add a human-oriented feature status.

    recommended=True is kept for backward compatibility, but the preferred
    interpretation field is now `status`:
      - recommended: strong candidate under current thresholds;
      - candidate: moderate candidate, useful but needs caution/more data;
      - weak: weak association or unstable signal;
      - reject: insufficient data/variation or no meaningful association.
    """

    ranking = ranking.copy()
    ranking_cfg = screening.get("ranking", {})

    min_valid_fraction = float(ranking_cfg.get("min_valid_fraction", defaults.get("min_valid_fraction", 0.5)))
    min_pairwise_n = int(ranking_cfg.get("min_pairwise_n", defaults.get("min_pairwise_n", 5)))
    min_unique_values = int(ranking_cfg.get("min_unique_values", defaults.get("min_unique_values", 4)))
    min_abs_corr = float(ranking_cfg.get("min_abs_correlation", 0.3))
    min_stability = float(ranking_cfg.get("min_stability", 0.6))

    recommended_abs_corr = float(ranking_cfg.get("recommended_min_abs_correlation", max(0.4, min_abs_corr)))
    recommended_stability = float(ranking_cfg.get("recommended_min_stability", max(0.7, min_stability)))

    statuses = []
    recommended = []
    reasons = []

    for _, row in ranking.iterrows():
        hard_reasons = []
        soft_reasons = []

        if row["valid_fraction"] < min_valid_fraction:
            hard_reasons.append("low_valid_fraction")
        if row["pairwise_n"] < min_pairwise_n:
            hard_reasons.append("low_pairwise_n")
        if row["n_unique"] < min_unique_values:
            hard_reasons.append("low_unique_values")

        abs_spearman = row["abs_spearman"]
        stability = row["bootstrap_sign_stability"]

        if pd.isna(abs_spearman) or abs_spearman < min_abs_corr:
            soft_reasons.append("low_abs_spearman")
        if pd.isna(stability) or stability < min_stability:
            soft_reasons.append("low_bootstrap_stability")

        if hard_reasons:
            status = "reject"
            reason = ";".join(hard_reasons + soft_reasons)
        elif not soft_reasons and abs_spearman >= recommended_abs_corr and stability >= recommended_stability:
            status = "recommended"
            reason = "strong_association;stable_direction;sufficient_coverage"
        elif not soft_reasons:
            status = "candidate"
            reason = "moderate_association;passes_minimum_filters"
        elif abs_spearman >= min_abs_corr or (not pd.isna(stability) and stability >= min_stability):
            status = "weak"
            reason = ";".join(soft_reasons)
        else:
            status = "reject"
            reason = ";".join(soft_reasons)

        statuses.append(status)
        recommended.append(status == "recommended")
        reasons.append(reason)

    ranking["status"] = statuses
    ranking["recommended"] = recommended
    ranking["not_recommended_reason"] = reasons
    return ranking


# -----------------------------------------------------------------------------
# Human-readable feature names / reports
# -----------------------------------------------------------------------------


def add_readable_feature_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()

    parsed = ranking["feature_name"].apply(parse_feature_name_for_display).apply(pd.Series)
    for col in parsed.columns:
        ranking[col] = parsed[col]

    return ranking


def parse_feature_name_for_display(feature_name: str) -> dict[str, str]:
    text = str(feature_name)

    if "::" in text:
        feature_set, tail = text.split("::", 1)
    else:
        feature_set, tail = "", text

    parts = tail.split("__")

    # Raw ECG format:
    # ecg_raw_rest_before::before__long__full__hrv_time__RMSSD_ms
    if len(parts) >= 5 and parts[0] in {"before", "after", "before2", "after2"}:
        phase, record_type, segment = parts[0], parts[1], parts[2]
        raw_feature = "__".join(parts[3:])
        return {
            "feature_set_name": feature_set,
            "feature_family": "ECG raw",
            "operation": "absolute",
            "condition": f"{phase}/{record_type}/{segment}",
            "base_feature": raw_feature,
            "formula_readable": "absolute value",
            "feature_readable": f"{pretty_ecg_feature(raw_feature)} ({phase}, {record_type}, {segment})",
        }

    # Derived feature format:
    # ecg_orthostatic_before::orthostatic_before__hrv_time__RMSSD_ms__delta
    # ecg_orthostatic_before::orthostatic_before__hrv_time__RMSSD_ms__delta__base=ratio
    if len(parts) >= 3:
        operation = parts[0]
        formula = parts[-1]
        base_formula = ""

        if formula.startswith("base=") and len(parts) >= 4:
            base_formula = formula.replace("base=", "")
            formula = parts[-2]
            base_feature = "__".join(parts[1:-2])
        else:
            base_feature = "__".join(parts[1:-1])

        condition = pretty_operation(operation)
        formula_text = pretty_formula(formula)
        if base_formula:
            formula_text += f" of {pretty_formula(base_formula)}"

        return {
            "feature_set_name": feature_set,
            "feature_family": infer_feature_family(base_feature),
            "operation": operation,
            "condition": condition,
            "base_feature": base_feature,
            "formula_readable": formula_text,
            "feature_readable": f"{condition}: {pretty_ecg_feature(base_feature)} — {formula_text}",
        }

    return {
        "feature_set_name": feature_set,
        "feature_family": "unknown",
        "operation": "",
        "condition": "",
        "base_feature": tail,
        "formula_readable": "",
        "feature_readable": text,
    }


def infer_feature_family(feature: str) -> str:
    if feature.startswith("hrv_time__"):
        return "HRV time"
    if feature.startswith("hrv_freq__"):
        return "HRV frequency"
    if feature.startswith("hrv_nonlinear__"):
        return "HRV nonlinear"
    if feature.startswith("morph_qrs__"):
        return "QRS morphology"
    if feature.startswith("morph_p__"):
        return "P-wave morphology"
    if feature.startswith("morph_t__"):
        return "T-wave morphology"
    return "context/other"


def pretty_operation(operation: str) -> str:
    mapping = {
        "orthostatic_before": "Orthostatic response before training",
        "orthostatic_after": "Orthostatic response after training",
        "breath_in_response_before": "Inhale breath-hold response before training",
        "breath_out_response_before": "Exhale breath-hold response before training",
        "training_response": "Training response",
        "training_response_hrv_ratio": "Training response, HRV ratio",
        "orthostatic_training_response": "Change in orthostatic response after training",
        "baseline_dev_sit_before": "Deviation from personal baseline, sit before training",
    }
    return mapping.get(operation, operation.replace("_", " "))


def pretty_formula(formula: str) -> str:
    mapping = {
        "delta": "difference",
        "percent_delta": "percent change",
        "ratio": "ratio",
        "log_ratio": "log-ratio",
        "robust_z": "robust z-score",
    }
    return mapping.get(formula, formula.replace("_", " "))


def pretty_ecg_feature(feature: str) -> str:
    mapping = {
        "hrv_time__MeanHR_bpm": "Mean heart rate, bpm",
        "hrv_time__MeanNN_ms": "Mean NN interval, ms",
        "hrv_time__SDNN_ms": "SDNN, ms",
        "hrv_time__RMSSD_ms": "RMSSD, ms",
        "hrv_time__pNN50_percent": "pNN50, %",
        "hrv_freq__LF_power": "LF power",
        "hrv_freq__HF_power": "HF power",
        "hrv_freq__LF_HF_ratio": "LF/HF ratio",
        "hrv_nonlinear__SD1_ms": "Poincaré SD1, ms",
        "hrv_nonlinear__SD2_ms": "Poincaré SD2, ms",
        "hrv_nonlinear__SD1_SD2_ratio": "SD1/SD2 ratio",
        "morph_qrs__QRS_duration_ms": "QRS duration, ms",
        "morph_qrs__QRS_area": "QRS area",
        "morph_qrs__QRS_main_amp": "QRS main amplitude",
        "morph_qrs__R_width_half_ms": "R half-width, ms",
        "morph_t__T_amp": "T-wave amplitude",
        "morph_t__RT_interval_ms": "RT interval, ms",
        "morph_t__QT_like_ms": "QT-like interval, ms",
    }
    return mapping.get(feature, feature)


def add_formula_family_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()

    if "formula_readable" not in ranking.columns:
        ranking["formula_family"] = "unknown"
        ranking["equivalent_feature_key"] = ranking["feature_name"]
        return ranking

    ranking["formula_family"] = ranking.apply(infer_formula_family_from_row, axis=1)
    ranking["equivalent_feature_key"] = ranking.apply(make_equivalent_feature_key, axis=1)
    return ranking


def infer_formula_family_from_row(row: pd.Series) -> str:
    text = str(row.get("feature_name", ""))
    formula = ""

    if "::" in text:
        tail = text.split("::", 1)[1]
    else:
        tail = text

    parts = tail.split("__")

    if len(parts) >= 3:
        formula = parts[-1]
        if formula.startswith("base=") and len(parts) >= 4:
            formula = parts[-2]

    return get_formula_family(formula)


def get_formula_family(formula: str) -> str:
    if formula in {"percent_delta", "ratio"}:
        return "relative_change"
    if formula == "delta":
        return "absolute_change"
    if formula == "robust_z":
        return "normalized_deviation"
    if formula == "log_ratio":
        return "log_relative_change"
    if formula in {"", "absolute", "absolute value"}:
        return "absolute_value"
    return formula or "unknown"


def make_equivalent_feature_key(row: pd.Series) -> str:
    """
    Key used to collapse near-duplicate feature representations in the human report.

    Example: percent_delta and ratio for the same operation/base_feature are treated
    as the same relative_change family.
    """

    feature_set = str(row.get("feature_set_name", ""))
    operation = str(row.get("operation", ""))
    condition = str(row.get("condition", ""))
    base_feature = str(row.get("base_feature", ""))
    family = str(row.get("formula_family", ""))
    return "||".join([feature_set, operation, condition, base_feature, family])


def make_human_ranking_report(
    ranking: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    screening: dict[str, Any] | None = None,
) -> pd.DataFrame:
    report_source = ranking.copy()

    if should_collapse_equivalent_features(cfg, screening):
        report_source = collapse_equivalent_features_for_report(report_source, cfg=cfg, screening=screening)

    columns = [
        "rank",
        "status",
        "recommended",
        "feature_readable",
        "feature_family",
        "condition",
        "base_feature",
        "formula_readable",
        "formula_family",
        "pairwise_n",
        "valid_fraction",
        "spearman_r",
        "abs_spearman",
        "bootstrap_sign_stability",
        "mutual_information",
        "not_recommended_reason",
        "feature_name",
    ]

    existing = [col for col in columns if col in report_source.columns]
    report = report_source[existing].copy()

    report = report.reset_index(drop=True)
    report["rank"] = np.arange(1, len(report) + 1)

    # Round for readability, not for technical storage.
    for col in ["valid_fraction", "spearman_r", "abs_spearman", "bootstrap_sign_stability", "mutual_information"]:
        if col in report.columns:
            report[col] = report[col].round(3)

    return report


def should_collapse_equivalent_features(
    cfg: dict[str, Any] | None,
    screening: dict[str, Any] | None,
) -> bool:
    screening = screening or {}
    cfg = cfg or {}

    if "collapse_equivalent_features" in screening:
        return bool(screening["collapse_equivalent_features"])

    defaults = cfg.get("defaults", {})
    return bool(defaults.get("collapse_equivalent_features", True))


def collapse_equivalent_features_for_report(
    ranking: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    screening: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Collapse near-duplicate feature transformations only for the human report.

    Technical ranking remains untouched. By default, percent_delta is preferred
    over ratio for relative_change features. If the preferred formula is absent,
    the row with the best ranking metric is kept.
    """

    if ranking.empty or "equivalent_feature_key" not in ranking.columns:
        return ranking

    cfg = cfg or {}
    screening = screening or {}
    ranking_cfg = screening.get("ranking", {})
    primary_metric = ranking_cfg.get("primary_metric", "abs_spearman")

    formula_preferences = get_formula_preferences(cfg, screening)

    rows = []
    for _, group in ranking.groupby("equivalent_feature_key", dropna=False, sort=False):
        rows.append(select_representative_feature_row(group, primary_metric, formula_preferences))

    collapsed = pd.DataFrame(rows)
    if primary_metric in collapsed.columns:
        collapsed = collapsed.sort_values(primary_metric, ascending=False, na_position="last")
    elif "rank" in collapsed.columns:
        collapsed = collapsed.sort_values("rank")

    return collapsed.reset_index(drop=True)


def get_formula_preferences(
    cfg: dict[str, Any],
    screening: dict[str, Any],
) -> dict[str, str]:
    defaults = cfg.get("defaults", {})
    groups = defaults.get("equivalent_formula_groups", {})

    # Screening-level settings override defaults.
    groups = {**groups, **screening.get("equivalent_formula_groups", {})}

    preferences = {
        "relative_change": "percent_delta",
    }

    for family, spec in groups.items():
        if isinstance(spec, dict) and spec.get("preferred_formula"):
            preferences[family] = spec["preferred_formula"]

    return preferences


def select_representative_feature_row(
    group: pd.DataFrame,
    primary_metric: str,
    formula_preferences: dict[str, str],
) -> pd.Series:
    if group.empty:
        raise ValueError("Cannot select representative from empty group")

    family = str(group.iloc[0].get("formula_family", ""))
    preferred_formula = formula_preferences.get(family)

    if preferred_formula:
        preferred_mask = group["feature_name"].astype(str).str.contains(f"__{preferred_formula}", regex=False)
        if preferred_mask.any():
            candidates = group[preferred_mask].copy()
            if primary_metric in candidates.columns:
                return candidates.sort_values(primary_metric, ascending=False, na_position="last").iloc[0]
            return candidates.sort_values("rank").iloc[0]

    if primary_metric in group.columns:
        return group.sort_values(primary_metric, ascending=False, na_position="last").iloc[0]

    return group.sort_values("rank").iloc[0]


# -----------------------------------------------------------------------------
# Summaries / plots
# -----------------------------------------------------------------------------


def make_summary_csv(ranking: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = [
        {"metric": "n_observations", "value": len(X)},
        {"metric": "n_subjects", "value": get_n_subjects_from_index(X.index)},
        {"metric": "n_features", "value": X.shape[1]},
        {"metric": "target_valid_count", "value": int(y.notna().sum())},
        {"metric": "features_recommended", "value": int(ranking["recommended"].sum()) if "recommended" in ranking.columns else np.nan},
        {"metric": "median_valid_fraction", "value": float(ranking["valid_fraction"].median())},
        {"metric": "median_abs_spearman", "value": float(ranking["abs_spearman"].median(skipna=True))},
        {"metric": "max_abs_spearman", "value": float(ranking["abs_spearman"].max(skipna=True))},
    ]
    return pd.DataFrame(rows)


def get_n_subjects_from_index(index: pd.Index) -> int:
    if isinstance(index, pd.MultiIndex) and "subject_id" in index.names:
        return int(index.get_level_values("subject_id").nunique())
    return 1


def make_screening_plots(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    ranking: pd.DataFrame,
    plots_dir: Path,
    screening: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    if plt is None:
        warnings.warn("matplotlib is not available; plots were skipped", RuntimeWarning)
        return

    plot_cfg = build_plot_config(screening, defaults)
    if not plot_cfg.get("enabled", True):
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use the same collapsed representation for human plots as for human report.
    plot_ranking = ranking.copy()
    if plot_cfg.get("collapse_equivalent_features", True):
        plot_ranking = collapse_equivalent_features_for_report(plot_ranking, cfg={"defaults": defaults}, screening=screening)

    top_n = int(plot_cfg.get("top_n", defaults.get("top_n_plots", 12)))
    top_features = plot_ranking.head(top_n)["feature_name"].tolist()

    if plot_cfg.get("target_distribution", True):
        make_target_distribution_plot(y, plots_dir / "target_distribution.png")

    if plot_cfg.get("missingness", True):
        make_missingness_plot(plot_ranking, plots_dir / "missingness_top.png", top_n=top_n)

    if plot_cfg.get("scatter_with_target", True):
        scatter_dir = plots_dir / "scatter_with_target"
        for i, row in plot_ranking.head(top_n).iterrows():
            feature_name = row["feature_name"]
            if feature_name not in X.columns:
                continue
            family = sanitize_filename(str(row.get("feature_family", "other")))
            out_dir = scatter_dir / family
            out_dir.mkdir(parents=True, exist_ok=True)
            title = str(row.get("feature_readable", feature_name))
            safe_name = sanitize_filename(title)[:120]
            make_scatter_plot(
                x=X[feature_name],
                y=y,
                title=title,
                path=out_dir / f"rank_{int(row.get('rank', i + 1)):03d}_{safe_name}.png",
                xlabel=title,
                ylabel="Target",
            )

    if plot_cfg.get("time_series", False):
        ts_dir = plots_dir / "time_series"
        overlay_target = bool(plot_cfg.get("time_series_overlay_target", True))
        overlay_mode = str(plot_cfg.get("time_series_overlay_mode", "zscore"))
        for i, row in plot_ranking.head(top_n).iterrows():
            feature_name = row["feature_name"]
            if feature_name not in X.columns:
                continue
            family = sanitize_filename(str(row.get("feature_family", "other")))
            out_dir = ts_dir / family
            out_dir.mkdir(parents=True, exist_ok=True)
            title = str(row.get("feature_readable", feature_name))
            safe_name = sanitize_filename(title)[:120]
            make_time_series_plot(
                x=X[feature_name],
                y=y if overlay_target else None,
                title=title,
                path=out_dir / f"rank_{int(row.get('rank', i + 1)):03d}_{safe_name}.png",
                overlay_mode=overlay_mode,
            )

    if plot_cfg.get("correlation_matrix", False):
        matrix_dir = plots_dir / "correlation_matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        selected = [f for f in top_features if f in X.columns]
        make_feature_correlation_heatmap(
            X[selected],
            y=y if plot_cfg.get("correlation_matrix_include_target", True) else None,
            ranking=plot_ranking,
            path=matrix_dir / "top_features_with_target.png",
            method=str(plot_cfg.get("correlation_matrix_method", "spearman")),
            annot=bool(plot_cfg.get("correlation_matrix_annot", True)),
            mask_upper_triangle=bool(plot_cfg.get("correlation_matrix_mask_upper_triangle", True)),
        )


def build_plot_config(screening: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    global_plotting = defaults.get("plotting", {}) if isinstance(defaults.get("plotting", {}), dict) else {}
    local_plots = screening.get("plots", {}) if isinstance(screening.get("plots", {}), dict) else {}

    cfg = {
        "enabled": True,
        "top_n": defaults.get("top_n_plots", 12),
        "collapse_equivalent_features": defaults.get("collapse_equivalent_features", True),
        "target_distribution": True,
        "missingness": True,
        "scatter_with_target": True,
        "time_series": False,
        "time_series_overlay_target": True,
        "time_series_overlay_mode": "zscore",
        "correlation_matrix": False,
        "correlation_matrix_include_target": True,
        "correlation_matrix_method": "spearman",
        "correlation_matrix_annot": True,
        "correlation_matrix_mask_upper_triangle": True,
    }

    # Support both old and new config styles.
    cfg.update(global_plotting)
    cfg.update(local_plots)

    kinds = local_plots.get("kinds") or global_plotting.get("kinds")
    if kinds:
        kinds = set(kinds)
        cfg["scatter_with_target"] = "scatter_with_target" in kinds
        cfg["time_series"] = "time_series" in kinds
        cfg["correlation_matrix"] = "correlation_matrix" in kinds

    # Nested style support.
    if isinstance(local_plots.get("time_series"), dict):
        ts = local_plots["time_series"]
        cfg["time_series"] = ts.get("enabled", True)
        cfg["time_series_overlay_target"] = ts.get("overlay_target", cfg["time_series_overlay_target"])
        cfg["time_series_overlay_mode"] = ts.get("overlay_mode", cfg["time_series_overlay_mode"])
        cfg["top_n"] = ts.get("top_n", cfg["top_n"])

    if isinstance(local_plots.get("correlation_matrix"), dict):
        cm = local_plots["correlation_matrix"]
        cfg["correlation_matrix"] = cm.get("enabled", True)
        cfg["correlation_matrix_include_target"] = cm.get("include_target", cfg["correlation_matrix_include_target"])
        cfg["correlation_matrix_method"] = cm.get("method", cfg["correlation_matrix_method"])
        cfg["correlation_matrix_annot"] = cm.get("annot", cfg["correlation_matrix_annot"])
        cfg["correlation_matrix_mask_upper_triangle"] = cm.get("mask_upper_triangle", cfg["correlation_matrix_mask_upper_triangle"])
        cfg["top_n"] = cm.get("top_n", cfg["top_n"])

    return cfg


def make_missingness_plot(ranking: pd.DataFrame, path: Path, *, top_n: int) -> None:
    data = ranking.head(top_n).copy()
    if data.empty:
        return
    missing_fraction = 1.0 - data["valid_fraction"]

    plt.figure(figsize=(10, max(4, 0.35 * len(data))))
    plt.barh(np.arange(len(data)), missing_fraction)
    plt.yticks(np.arange(len(data)), data["feature_name"].astype(str))
    plt.xlabel("Missing fraction")
    plt.title("Missingness of top-ranked features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_target_distribution_plot(y: pd.Series, path: Path) -> None:
    values = pd.to_numeric(y, errors="coerce").dropna()
    if values.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=min(10, max(3, int(math.sqrt(len(values))))))
    plt.xlabel("Target value")
    plt.ylabel("Count")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_scatter_plot(
    x: pd.Series,
    y: pd.Series,
    title: str,
    path: Path,
    *,
    xlabel: str = "Feature",
    ylabel: str = "Target",
) -> None:
    pair = pd.concat([pd.to_numeric(x, errors="coerce").rename("x"), y.rename("y")], axis=1).dropna()
    if pair.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.scatter(pair["x"], pair["y"])

    # Add a simple trend line when possible. This is visual only, not used in ranking.
    if len(pair) >= 3 and pair["x"].nunique() >= 2:
        try:
            coeff = np.polyfit(pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float), deg=1)
            x_line = np.linspace(pair["x"].min(), pair["x"].max(), 100)
            y_line = coeff[0] * x_line + coeff[1]
            plt.plot(x_line, y_line, linewidth=1.5)
        except Exception:
            pass

    plt.xlabel(shorten_label(xlabel, max_len=90))
    plt.ylabel(ylabel)
    plt.title(shorten_label(title, max_len=110))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_time_series_plot(
    x: pd.Series,
    y: pd.Series | None,
    title: str,
    path: Path,
    *,
    overlay_mode: str = "zscore",
) -> None:
    x_num = pd.to_numeric(x, errors="coerce")

    if isinstance(x_num.index, pd.MultiIndex):
        # MVP behavior: plot each subject as a separate line on the same axes.
        x_df = x_num.rename("feature").reset_index()
    else:
        x_df = pd.DataFrame({"subject_id": "default", "date": pd.to_datetime(x_num.index), "feature": x_num.values})

    x_df = x_df.dropna(subset=["feature"]).sort_values(["subject_id", "date"])
    if x_df.empty:
        return

    plt.figure(figsize=(9, 4.8))

    if overlay_mode == "zscore":
        x_df["plot_value"] = x_df.groupby("subject_id")["feature"].transform(zscore_series)
        ylabel = "z-score"
    else:
        x_df["plot_value"] = x_df["feature"]
        ylabel = "feature value"

    for subject_id, subject_df in x_df.groupby("subject_id"):
        plt.plot(subject_df["date"], subject_df["plot_value"], marker="o", linewidth=1.5, label=f"feature: {subject_id}")

    if y is not None:
        y_num = pd.to_numeric(y, errors="coerce")
        if isinstance(y_num.index, pd.MultiIndex):
            y_df = y_num.rename("target").reset_index()
        else:
            y_df = pd.DataFrame({"subject_id": "default", "date": pd.to_datetime(y_num.index), "target": y_num.values})
        y_df = y_df.dropna(subset=["target"]).sort_values(["subject_id", "date"])

        if not y_df.empty:
            if overlay_mode == "zscore":
                y_df["plot_value"] = y_df.groupby("subject_id")["target"].transform(zscore_series)
                for subject_id, subject_df in y_df.groupby("subject_id"):
                    plt.plot(subject_df["date"], subject_df["plot_value"], marker="x", linestyle="--", linewidth=1.2, label=f"target: {subject_id}")
            elif overlay_mode == "raw_secondary_axis":
                ax = plt.gca()
                ax2 = ax.twinx()
                for subject_id, subject_df in y_df.groupby("subject_id"):
                    ax2.plot(subject_df["date"], subject_df["target"], marker="x", linestyle="--", linewidth=1.2, label=f"target: {subject_id}")
                ax2.set_ylabel("target value")

    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(shorten_label(title, max_len=110))
    plt.xticks(rotation=45)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(skipna=True)
    if pd.isna(std) or std == 0:
        return series * np.nan
    return (series - series.mean(skipna=True)) / std


def make_feature_correlation_heatmap(
    X: pd.DataFrame,
    path: Path,
    *,
    y: pd.Series | None = None,
    ranking: pd.DataFrame | None = None,
    method: str = "spearman",
    annot: bool = True,
    mask_upper_triangle: bool = True,
) -> None:
    if X.empty:
        return

    data = X.apply(pd.to_numeric, errors="coerce").copy()

    # Use readable labels where possible.
    rename_map = {}
    if ranking is not None and "feature_name" in ranking.columns and "feature_readable" in ranking.columns:
        rename_map = dict(zip(ranking["feature_name"], ranking["feature_readable"]))
    data = data.rename(columns={col: shorten_label(rename_map.get(col, col), max_len=45) for col in data.columns})

    if y is not None:
        y_aligned = normalize_analysis_index(y).reindex(data.index)
        data["Target"] = pd.to_numeric(y_aligned, errors="coerce")

    if data.shape[1] < 2:
        return

    corr = data.corr(method=method)
    if corr.empty:
        return

    labels = corr.columns.tolist()
    matrix = corr.to_numpy()

    if mask_upper_triangle:
        matrix = matrix.copy()
        mask = np.zeros_like(matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        matrix[mask] = np.nan

    size = max(7, min(18, 0.65 * len(labels)))
    plt.figure(figsize=(size, size))
    im = plt.imshow(matrix, vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, shrink=0.75, label=f"{method.title()} correlation")
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)

    if annot and len(labels) <= 16:
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                if pd.isna(value):
                    continue
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    plt.title("Correlation matrix: top features + target")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def shorten_label(value: str, *, max_len: int = 80) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def sanitize_filename(value: str) -> str:
    bad = r'<>:"/\\|?*' + "\n\r\t"
    result = str(value)
    for ch in bad:
        result = result.replace(ch, "_")
    result = result.replace(" ", "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def print_top_features(ranking: pd.DataFrame, *, n: int = 10) -> None:
    cols = [
        "rank",
        "feature_name",
        "pairwise_n",
        "valid_fraction",
        "spearman_r",
        "abs_spearman",
        "bootstrap_sign_stability",
        "mutual_information",
        "recommended",
    ]
    existing = [c for c in cols if c in ranking.columns]
    print("\n[INFO] Top features:")
    print(ranking[existing].head(n).to_string(index=False))


from pathlib import Path

ROOT_DIR = Path(r"C:\Users\pv190\Desktop\Feature_extractor")
PROTOCOL_CONFIG = Path(r"C:\Users\pv190\Desktop\Feature_extractor\protocol.yaml")

run_feature_analysis_with_protocol(
    root_dir=ROOT_DIR,
    protocol_config_path=PROTOCOL_CONFIG,
    debug=True,
)