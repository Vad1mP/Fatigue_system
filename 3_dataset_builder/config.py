from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_protocol(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Protocol must be a YAML mapping: {path}")
    return data


def get_analysis_config(protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol.get("analysis_dataset")
    if cfg is None:
        cfg = default_analysis_dataset_config()
    return merge_dicts(default_analysis_dataset_config(), cfg)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def default_analysis_dataset_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "tables": {
            "raw_ecg_features": {
                "path": "features_protocol.csv",
                "required": True,
                "date_column": "date",
                "date_format": "%d.%m.%Y",
                "csv": {"sep": ",", "decimal": ".", "encoding": "utf-8-sig"},
            },
            "ecg_derived": {
                "path": "features_derived.csv",
                "required": True,
                "date_column": "date",
                "date_format": "%Y-%m-%d",
                "csv": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
            },
            "context_computed": {
                "path": "context_computed.csv",
                "required": True,
                "date_column": "date",
                "date_format": "%Y-%m-%d",
                "csv": {"sep": ",", "decimal": ".", "encoding": "utf-8-sig"},
            },
            "context_derived": {
                "path": "context_derived.csv",
                "required": True,
                "date_column": "date",
                "date_format": "%Y-%m-%d",
                "csv": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
            },
        },
        "output": {
            "dataset": "analysis_dataset.csv",
            "feature_catalog": "analysis_feature_catalog.csv",
            "summary_json": "analysis_summary.json",
            "report_markdown": "analysis_report.md",
            "report_dir": "analysis_report",
        },
        "unit": {"mode": "date", "date_column": "date"},
        "join": {"how": "outer", "base_table": "context_computed", "sort_by_date": True},
        "row_filter": {"enabled": False, "mode": "none"},
        "schemas": {
            "raw_ecg_features": {
                "date": "date",
                "phase": "phase",
                "recording": "record_type",
                "segment": "segment_label",
                "hrv_valid": "hrv_validated",
                "morphology_valid": "morphology_validated",
            },
            "ecg_derived": {
                "date": "date",
                "operation_id": "operation_id",
                "operation_type": "operation_type",
                "feature": "feature",
                "formula": "formula",
                "base_formula": "base_formula",
                "value": "value",
                "valid": "valid",
                "reason": "reason",
            },
            "context_derived": {
                "date": "date",
                "operation_id": "operation_id",
                "operation_type": "operation_type",
                "feature": "feature",
                "formula": "formula",
                "base_formula": "base_formula",
                "value": "value",
                "valid": "valid",
                "reason": "reason",
            },
        },
        "feature_set_resolution": {"mode": "strict", "prefix_fallback": {"enabled": False}},
        "column_naming": {
            "separator": "__",
            "prefixes": {
                "meta": "meta",
                "qc": "qc",
                "raw_ecg": "raw_ecg",
                "ecg_derived": "ecg_der",
                "context_computed": "ctx",
                "context_derived": "ctx_der",
                "target": "target",
                "covariate": "covar",
            },
            "raw_ecg_template": "{prefix}__{phase}__{recording}__{segment}__{feature}",
            "ecg_derived_template": "{prefix}__{operation_id}__{formula_key}__{feature}",
            "context_computed_template": "{prefix}__{feature}",
            "context_derived_template": "{prefix}__{operation_id}__{formula_key}__{feature}",
            "target_template": "{prefix}__{id}",
            "covariate_template": "{prefix}__{id}",
        },
        "include": {
            "metadata": {"enabled": True, "columns": ["date"]},
            "qc_summary": {
                "enabled": True,
                "columns": [
                    "n_ecg_records_expected",
                    "n_ecg_records_found",
                    "n_ecg_records_valid_hrv",
                    "n_ecg_records_valid_morphology",
                    "complete_core_day",
                ],
            },
            "raw_ecg": {
                "enabled": True,
                "sources": [
                    "sit_before",
                    "stand_before",
                    "long_before",
                    "sit_after",
                    "stand_after",
                    "long_after",
                    "breath_in_start_before",
                    "breath_in_end_before",
                ],
                "feature_sets": ["hrv_core", "morphology_core"],
                "quality_policy": "valid_only",
            },
            "ecg_derived": {
                "enabled": True,
                "operations": [
                    "orthostatic_before",
                    "orthostatic_after",
                    "training_response",
                    "training_response_hrv_ratio",
                    "breath_in_response_before",
                    "orthostatic_training_response",
                    "baseline_dev_sit_before",
                ],
                "quality_policy": "valid_only",
            },
            "context_computed": {
                "enabled": True,
                "feature_sets": [
                    "sleep_core",
                    "morning_state_core",
                    "pressure_reaction_before",
                    "training_load_core",
                ],
            },
            "context_derived": {
                "enabled": True,
                "operations": [
                    "subjective_before_after_response",
                    "physio_before_after_response",
                    "baseline_dev_context_before",
                    "baseline_dev_context_flags",
                    "day_to_day_context_change",
                    "training_load_previous_7d",
                ],
            },
        },
        "targets": [
            {
                "id": "subjective_strain_delta",
                "label": "Acute subjective strain response",
                "source": "context_derived",
                "operation_id": "subjective_before_after_response",
                "feature": "subjective_strain",
                "formula": "delta",
                "direction": "higher_worse",
                "horizon": "acute",
                "data_type": "numeric",
                "scale_type": "ordinal",
            },
            {
                "id": "fatigue_delta",
                "label": "Acute subjective fatigue response",
                "source": "context_derived",
                "operation_id": "subjective_before_after_response",
                "feature": "Fatigue",
                "formula": "delta",
                "direction": "higher_worse",
                "horizon": "acute",
                "data_type": "numeric",
                "scale_type": "ordinal",
            },
            {
                "id": "reaction_time_delta",
                "label": "Acute reaction time response",
                "source": "context_derived",
                "operation_id": "physio_before_after_response",
                "feature": "Reaction_time_ms",
                "formula": "delta",
                "direction": "higher_worse",
                "horizon": "acute",
                "data_type": "numeric",
                "scale_type": "continuous",
            },
            {
                "id": "readiness_before",
                "label": "Pre-training readiness",
                "source": "context_computed",
                "column": "readiness_before",
                "direction": "higher_better",
                "horizon": "pre_training",
                "data_type": "numeric",
                "scale_type": "ordinal",
            },
        ],
        "covariates": [
            {
                "id": "training_load_previous_7d",
                "label": "Previous 7-day accumulated sRPE load",
                "source": "context_derived",
                "operation_id": "training_load_previous_7d",
                "feature": "sRPE_load",
                "formula": "rolling_7d_sum",
                "direction": "higher_load",
                "horizon": "previous_7d",
                "data_type": "numeric",
                "scale_type": "continuous",
            },
            {
                "id": "sleep_hours",
                "label": "Sleep duration",
                "source": "context_computed",
                "column": "Sleep_hours",
                "direction": "higher_better",
                "horizon": "same_day",
                "data_type": "numeric",
                "scale_type": "continuous",
            },
            {
                "id": "sRPE_load",
                "label": "Current day sRPE training load",
                "source": "context_computed",
                "column": "sRPE_load",
                "direction": "higher_load",
                "horizon": "same_day",
                "data_type": "numeric",
                "scale_type": "continuous",
            },
        ],
        "quality_rules": {
            "feature": {"min_valid_n": 10, "min_valid_ratio": 0.6, "min_unique_values": 5, "min_iqr": 1.0e-9},
            "target": {"min_valid_n": 10, "min_valid_ratio": 0.6, "min_unique_values": 4, "min_iqr": 1.0e-9},
            "feature_target_overlap": {"min_overlap_n": 10, "min_overlap_ratio": 0.5},
            "ml_readiness": {
                "min_samples_regression": 15,
                "min_samples_classification": 20,
                "min_class_count": 5,
                "max_features_to_samples_ratio": 0.5,
            },
        },
        "core_day_definition": {
            "required_ecg_sources": [
                "sit_before",
                "stand_before",
                "long_before",
                "sit_after",
                "stand_after",
                "long_after",
            ],
            "required_context_columns": [
                "Fatigue_before",
                "Fatigue_after",
                "subjective_strain_before",
                "subjective_strain_after",
                "readiness_before",
                "train_minutes",
                "RPE",
            ],
            "require_hrv_valid": True,
            "require_morphology_valid": False,
        },
        "reports": {
            "save_csv": True,
            "save_markdown": True,
            "save_json": True,
            "sections": {
                "summary": {"enabled": True},
                "coverage_by_day": {"enabled": True, "output": "report_coverage_by_day.csv"},
                "ecg_quality": {"enabled": True, "output": "report_ecg_quality.csv"},
                "context_completeness": {"enabled": True, "output": "report_context_completeness.csv"},
                "derived_operations": {"enabled": True, "output": "report_derived_operations.csv"},
                "targets": {"enabled": True, "output": "report_targets.csv"},
                "feature_availability": {"enabled": True, "output": "report_feature_availability.csv"},
                "feature_target_overlap": {"enabled": True, "output": "report_feature_target_overlap.csv"},
                "ml_readiness": {"enabled": True, "output": "report_ml_readiness.csv"},
            },
        },
        "warnings": {
            "detect_suspicious_column_names": True,
            "detect_all_missing_columns": True,
            "detect_constant_features": True,
            "detect_duplicate_columns": True,
            "detect_target_leakage_candidates": True,
            "detect_low_overlap_targets": True,
        },
    }
