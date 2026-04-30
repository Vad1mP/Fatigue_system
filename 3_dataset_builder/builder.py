from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_analysis_config, load_protocol
from .features import bool_series, feature_validation_column, formula_key, resolve_feature_sets
from .io import read_configured_tables, write_csv
from .reports import (
    make_context_completeness_report,
    make_derived_operations_report,
    make_ecg_quality_report,
    make_feature_availability_report,
    make_feature_target_overlap_report,
    make_ml_readiness_report,
    make_summary,
    make_target_report,
    write_json,
    write_markdown_report,
)


def build_analysis_dataset(protocol_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    base_dir = protocol_path.parent
    protocol = load_protocol(protocol_path)
    cfg = get_analysis_config(protocol)
    if not cfg.get("enabled", True):
        raise RuntimeError("analysis_dataset.enabled is false")

    output_dir = Path(output_dir) if output_dir is not None else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = read_configured_tables(cfg, base_dir)
    dataset, catalog = _build_dataset(protocol, cfg, tables)
    dataset = _apply_join_and_filter(dataset, cfg)

    reports, warnings = _build_reports(protocol, cfg, tables, dataset, catalog)
    summary = make_summary(dataset, reports, warnings)

    out = cfg["output"]
    dataset_path = output_dir / out.get("dataset", "analysis_dataset.csv")
    catalog_path = output_dir / out.get("feature_catalog", "analysis_feature_catalog.csv")
    summary_path = output_dir / out.get("summary_json", "analysis_summary.json")
    report_md_path = output_dir / out.get("report_markdown", "analysis_report.md")
    report_dir = output_dir / out.get("report_dir", "analysis_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(dataset, dataset_path)
    write_csv(catalog, catalog_path)

    if cfg.get("reports", {}).get("save_csv", True):
        sections = cfg.get("reports", {}).get("sections", {})
        for name, df in reports.items():
            section_cfg = sections.get(name, {})
            output_name = section_cfg.get("output", f"report_{name}.csv")
            write_csv(df, report_dir / output_name)

    if cfg.get("reports", {}).get("save_json", True):
        write_json(summary, summary_path)

    if cfg.get("reports", {}).get("save_markdown", True):
        write_markdown_report(summary, reports, report_md_path)

    return {
        "dataset": dataset,
        "catalog": catalog,
        "reports": reports,
        "summary": summary,
        "paths": {
            "dataset": str(dataset_path),
            "feature_catalog": str(catalog_path),
            "summary_json": str(summary_path),
            "report_markdown": str(report_md_path),
            "report_dir": str(report_dir),
        },
    }


def _build_dataset(protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    catalog_rows: list[dict[str, Any]] = []

    all_dates = _all_dates(tables)
    dataset = pd.DataFrame({"date": sorted(all_dates)})
    catalog_rows.append({"column": "date", "role": "meta", "source_table": "computed", "feature_family": "date"})

    if cfg.get("include", {}).get("qc_summary", {}).get("enabled", True):
        qc = _build_qc_summary(protocol, cfg, tables)
        frames.append(qc)
        for c in qc.columns:
            if c != "date":
                catalog_rows.append({"column": c, "role": "qc", "source_table": "raw_ecg_features", "feature_family": "qc_summary"})

    inc = cfg.get("include", {})
    if inc.get("raw_ecg", {}).get("enabled", False):
        raw_df, raw_cat = _wide_raw_ecg(protocol, cfg, tables)
        frames.append(raw_df)
        catalog_rows.extend(raw_cat)

    if inc.get("ecg_derived", {}).get("enabled", False):
        der_df, der_cat = _wide_derived("ecg_derived", protocol, cfg, tables)
        frames.append(der_df)
        catalog_rows.extend(der_cat)

    if inc.get("context_computed", {}).get("enabled", False):
        ctx_df, ctx_cat = _wide_context_computed(protocol, cfg, tables)
        frames.append(ctx_df)
        catalog_rows.extend(ctx_cat)

    if inc.get("context_derived", {}).get("enabled", False):
        cder_df, cder_cat = _wide_derived("context_derived", protocol, cfg, tables)
        frames.append(cder_df)
        catalog_rows.extend(cder_cat)

    target_cols = {}
    if cfg.get("targets"):
        tgt_df, tgt_cat, target_cols = _wide_named_items("target", cfg.get("targets", []), cfg, tables)
        frames.append(tgt_df)
        catalog_rows.extend(tgt_cat)

    if cfg.get("covariates"):
        cov_df, cov_cat, _ = _wide_named_items("covariate", cfg.get("covariates", []), cfg, tables)
        frames.append(cov_df)
        catalog_rows.extend(cov_cat)

    for frame in frames:
        if frame.empty:
            continue
        dataset = dataset.merge(frame, on="date", how="outer")

    if cfg.get("join", {}).get("sort_by_date", True) and "date" in dataset.columns:
        dataset = dataset.sort_values("date").reset_index(drop=True)

    catalog = pd.DataFrame(catalog_rows).drop_duplicates(subset=["column"], keep="first").reset_index(drop=True)
    if target_cols:
        catalog["is_target"] = catalog["column"].isin(target_cols.values())
    return dataset, catalog


def _all_dates(tables: dict[str, pd.DataFrame]) -> set[str]:
    dates: set[str] = set()
    for df in tables.values():
        if not df.empty and "date" in df.columns:
            dates.update(df["date"].dropna().astype(str).tolist())
    return dates


def _apply_join_and_filter(dataset: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    # For the MVP, tables are outer-merged first. Row filtering is explicit and non-destructive by default.
    row_filter = cfg.get("row_filter", {}) or {}
    if row_filter.get("enabled") and row_filter.get("mode") == "complete_core_day" and "complete_core_day" in dataset.columns:
        return dataset[dataset["complete_core_day"].fillna(False)].reset_index(drop=True)
    return dataset


def _build_qc_summary(protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = tables.get("raw_ecg_features", pd.DataFrame())
    context = tables.get("context_computed", pd.DataFrame())
    if raw.empty:
        return pd.DataFrame(columns=["date"])

    schema = cfg.get("schemas", {}).get("raw_ecg_features", {})
    source_defs = protocol.get("derived_features", {}).get("sources", {}) or {}
    core = cfg.get("core_day_definition", {}) or {}
    required_sources = core.get("required_ecg_sources", []) or []
    required_context = core.get("required_context_columns", []) or []
    require_hrv = bool(core.get("require_hrv_valid", True))
    require_morph = bool(core.get("require_morphology_valid", False))

    rows = []
    for date, g in raw.groupby("date", dropna=False):
        row = {"date": date, "n_ecg_records_expected": len(required_sources)}
        found = 0
        valid_hrv = 0
        valid_morph = 0
        source_status: dict[str, dict[str, bool]] = {}
        for source in required_sources:
            spec = source_defs.get(source)
            if not spec:
                source_status[source] = {"found": False, "hrv": False, "morph": False}
                continue
            mask = _source_mask(g, spec, schema)
            sg = g[mask]
            is_found = len(sg) > 0
            found += int(is_found)
            hrv_ok = False
            morph_ok = False
            hrv_col = schema.get("hrv_valid", "hrv_validated")
            morph_col = schema.get("morphology_valid", "morphology_validated")
            if is_found and hrv_col in sg.columns:
                hrv_ok = bool(bool_series(sg[hrv_col]).any())
            if is_found and morph_col in sg.columns:
                morph_ok = bool(bool_series(sg[morph_col]).any())
            valid_hrv += int(hrv_ok)
            valid_morph += int(morph_ok)
            source_status[source] = {"found": is_found, "hrv": hrv_ok, "morph": morph_ok}
        row["n_ecg_records_found"] = found
        row["n_ecg_records_valid_hrv"] = valid_hrv
        row["n_ecg_records_valid_morphology"] = valid_morph

        ecg_complete = all(v["found"] for v in source_status.values())
        if require_hrv:
            ecg_complete = ecg_complete and all(v["hrv"] for v in source_status.values())
        if require_morph:
            ecg_complete = ecg_complete and all(v["morph"] for v in source_status.values())

        ctx_complete = True
        if required_context and not context.empty:
            crows = context[context["date"] == date]
            if crows.empty:
                ctx_complete = False
            else:
                first = crows.iloc[0]
                for c in required_context:
                    if c not in context.columns or pd.isna(first.get(c)):
                        ctx_complete = False
                        break
        elif required_context:
            ctx_complete = False
        row["complete_core_day"] = bool(ecg_complete and ctx_complete)
        rows.append(row)
    return pd.DataFrame(rows)


def _source_mask(df: pd.DataFrame, spec: dict[str, Any], schema: dict[str, str]) -> pd.Series:
    phase_col = schema.get("phase", "phase")
    rec_col = schema.get("recording", "record_type")
    seg_col = schema.get("segment", "segment_label")
    mask = pd.Series(True, index=df.index)
    if "phase" in spec:
        mask &= df[phase_col].astype(str) == str(spec["phase"])
    if "recording" in spec:
        mask &= df[rec_col].astype(str) == str(spec["recording"])
    if "segment" in spec:
        mask &= df[seg_col].astype(str) == str(spec["segment"])
    return mask


def _wide_raw_ecg(protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw = tables.get("raw_ecg_features", pd.DataFrame())
    if raw.empty:
        return pd.DataFrame(columns=["date"]), []

    schema = cfg.get("schemas", {}).get("raw_ecg_features", {})
    include_cfg = cfg.get("include", {}).get("raw_ecg", {})
    feature_sets = protocol.get("derived_features", {}).get("feature_sets", {}) or {}
    mode = cfg.get("feature_set_resolution", {}).get("mode", "strict")
    features = resolve_feature_sets(include_cfg.get("feature_sets", []), feature_sets, raw.columns, mode=mode)
    source_defs = protocol.get("derived_features", {}).get("sources", {}) or {}
    sources = include_cfg.get("sources", []) or []
    quality_policy = include_cfg.get("quality_policy", "valid_only")
    prefixes = cfg.get("column_naming", {}).get("prefixes", {})
    template = cfg.get("column_naming", {}).get("raw_ecg_template", "{prefix}__{phase}__{recording}__{segment}__{feature}")
    prefix = prefixes.get("raw_ecg", "raw_ecg")

    wide = pd.DataFrame({"date": sorted(raw["date"].dropna().unique())})
    catalog: list[dict[str, Any]] = []

    for source in sources:
        if source not in source_defs:
            raise KeyError(f"raw_ecg source '{source}' is not defined in derived_features.sources")
        spec = source_defs[source]
        sg = raw[_source_mask(raw, spec, schema)].copy()
        if sg.empty:
            continue
        per_source = pd.DataFrame({"date": sorted(sg["date"].dropna().unique())})
        for feature in features:
            if feature not in sg.columns:
                continue
            values = pd.to_numeric(sg[feature], errors="coerce")
            if quality_policy == "valid_only":
                valid_col = feature_validation_column(feature, schema)
                if valid_col and valid_col in sg.columns:
                    values = values.where(bool_series(sg[valid_col]), np.nan)
            tmp = pd.DataFrame({"date": sg["date"], "value": values})
            tmp = tmp.groupby("date", as_index=False)["value"].first()
            col = template.format(prefix=prefix, phase=spec.get("phase"), recording=spec.get("recording"), segment=spec.get("segment"), feature=feature)
            tmp = tmp.rename(columns={"value": col})
            per_source = per_source.merge(tmp, on="date", how="outer")
            catalog.append({
                "column": col,
                "role": "input",
                "source_table": "raw_ecg_features",
                "feature_family": _family(feature),
                "source_id": source,
                "phase": spec.get("phase"),
                "recording": spec.get("recording"),
                "segment": spec.get("segment"),
                "feature": feature,
            })
        wide = wide.merge(per_source, on="date", how="outer")
    return wide, catalog


def _wide_derived(table_id: str, protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = tables.get(table_id, pd.DataFrame())
    if df.empty:
        return pd.DataFrame(columns=["date"]), []
    include_cfg = cfg.get("include", {}).get(table_id, {})
    operations = include_cfg.get("operations", []) or []
    quality_policy = include_cfg.get("quality_policy", "valid_only")
    prefixes = cfg.get("column_naming", {}).get("prefixes", {})
    prefix = prefixes.get("ecg_derived" if table_id == "ecg_derived" else "context_derived", "ecg_der" if table_id == "ecg_derived" else "ctx_der")
    template_key = "ecg_derived_template" if table_id == "ecg_derived" else "context_derived_template"
    template = cfg.get("column_naming", {}).get(template_key, "{prefix}__{operation_id}__{formula_key}__{feature}")

    sub = df[df["operation_id"].isin(operations)].copy() if operations else df.copy()
    if sub.empty:
        return pd.DataFrame({"date": sorted(df["date"].dropna().unique())}), []
    if quality_policy == "valid_only" and "valid" in sub.columns:
        sub.loc[~bool_series(sub["valid"]), "value"] = np.nan
    sub["formula_key"] = sub.apply(formula_key, axis=1)
    sub["column"] = sub.apply(lambda r: template.format(prefix=prefix, operation_id=r["operation_id"], formula_key=r["formula_key"], feature=r["feature"]), axis=1)

    wide = sub.pivot_table(index="date", columns="column", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    catalog = []
    for _, r in sub.drop_duplicates("column").iterrows():
        catalog.append({
            "column": r["column"],
            "role": "input",
            "source_table": table_id,
            "feature_family": _family(str(r["feature"])),
            "operation_id": r.get("operation_id"),
            "operation_type": r.get("operation_type"),
            "formula": r.get("formula"),
            "base_formula": r.get("base_formula"),
            "formula_key": r.get("formula_key"),
            "feature": r.get("feature"),
        })
    return wide, catalog


def _wide_context_computed(protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = tables.get("context_computed", pd.DataFrame())
    if df.empty:
        return pd.DataFrame(columns=["date"]), []
    include_cfg = cfg.get("include", {}).get("context_computed", {})
    feature_sets = protocol.get("context_features", {}).get("feature_sets", {}) or {}
    mode = cfg.get("feature_set_resolution", {}).get("mode", "strict")
    features = resolve_feature_sets(include_cfg.get("feature_sets", []), feature_sets, df.columns, mode=mode)
    prefixes = cfg.get("column_naming", {}).get("prefixes", {})
    prefix = prefixes.get("context_computed", "ctx")
    template = cfg.get("column_naming", {}).get("context_computed_template", "{prefix}__{feature}")

    cols = ["date"] + features
    wide = df[cols].copy()
    rename = {f: template.format(prefix=prefix, feature=f) for f in features}
    wide = wide.rename(columns=rename)
    # If repeated dates exist, keep first non-null by column.
    wide = wide.groupby("date", as_index=False).first()
    catalog = [{
        "column": rename[f],
        "role": "input",
        "source_table": "context_computed",
        "feature_family": "context",
        "feature": f,
    } for f in features]
    return wide, catalog


def _wide_named_items(kind: str, items: list[dict[str, Any]], cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    assert kind in {"target", "covariate"}
    prefixes = cfg.get("column_naming", {}).get("prefixes", {})
    prefix = prefixes.get(kind, "target" if kind == "target" else "covar")
    template = cfg.get("column_naming", {}).get(f"{kind}_template", "{prefix}__{id}")
    all_dates = _all_dates(tables)
    wide = pd.DataFrame({"date": sorted(all_dates)})
    catalog = []
    item_cols = {}

    for item in items:
        item_id = item["id"]
        col = template.format(prefix=prefix, id=item_id)
        item_cols[item_id] = col
        source = item.get("source")
        values = _extract_named_item_values(item, source, tables)
        if values.empty:
            tmp = pd.DataFrame({"date": sorted(all_dates), col: np.nan})
        else:
            tmp = values.rename(columns={"value": col})
        wide = wide.merge(tmp[["date", col]], on="date", how="left")
        catalog.append({
            "column": col,
            "role": kind,
            "source_table": source,
            "feature_family": kind,
            "id": item_id,
            "label": item.get("label", ""),
            "direction": item.get("direction", ""),
            "horizon": item.get("horizon", ""),
            "data_type": item.get("data_type", ""),
            "scale_type": item.get("scale_type", ""),
            "operation_id": item.get("operation_id", ""),
            "feature": item.get("feature", item.get("column", "")),
            "formula": item.get("formula", ""),
        })
    return wide, catalog, item_cols


def _extract_named_item_values(item: dict[str, Any], source: str | None, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if source == "context_computed":
        df = tables.get("context_computed", pd.DataFrame())
        column = item.get("column")
        if df.empty or column not in df.columns:
            return pd.DataFrame(columns=["date", "value"])
        return df[["date", column]].rename(columns={column: "value"}).groupby("date", as_index=False).first()

    if source in {"context_derived", "ecg_derived"}:
        df = tables.get(source, pd.DataFrame())
        if df.empty:
            return pd.DataFrame(columns=["date", "value"])
        mask = pd.Series(True, index=df.index)
        for key in ["operation_id", "feature", "formula"]:
            if item.get(key) is not None:
                mask &= df[key].astype(str) == str(item[key])
        if item.get("base_formula") is not None and "base_formula" in df.columns:
            mask &= df["base_formula"].astype(str) == str(item["base_formula"])
        sub = df[mask].copy()
        if sub.empty:
            return pd.DataFrame(columns=["date", "value"])
        if "valid" in sub.columns:
            sub.loc[~bool_series(sub["valid"]), "value"] = np.nan
        return sub[["date", "value"]].groupby("date", as_index=False).first()

    if source == "raw_ecg":
        # Not used in the MVP. Prefer context targets to avoid leakage.
        return pd.DataFrame(columns=["date", "value"])

    return pd.DataFrame(columns=["date", "value"])


def _build_reports(protocol: dict[str, Any], cfg: dict[str, Any], tables: dict[str, pd.DataFrame], dataset: pd.DataFrame, catalog: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
    reports: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    raw_schema = cfg.get("schemas", {}).get("raw_ecg_features", {})
    reports["coverage_by_day"] = dataset[[c for c in ["date", "n_ecg_records_expected", "n_ecg_records_found", "n_ecg_records_valid_hrv", "n_ecg_records_valid_morphology", "complete_core_day"] if c in dataset.columns]].copy()
    reports["ecg_quality"] = make_ecg_quality_report(tables.get("raw_ecg_features", pd.DataFrame()), raw_schema)

    context_features = _context_columns_for_report(protocol, cfg, tables.get("context_computed", pd.DataFrame()))
    reports["context_completeness"] = make_context_completeness_report(tables.get("context_computed", pd.DataFrame()), context_features)
    reports["derived_operations"] = make_derived_operations_report(tables)

    target_cols = {row.get("id"): row.get("column") for _, row in catalog[catalog["role"] == "target"].iterrows()} if not catalog.empty else {}
    reports["targets"] = make_target_report(dataset, cfg.get("targets", []), target_cols, cfg.get("quality_rules", {}).get("target", {}))
    reports["feature_availability"] = make_feature_availability_report(dataset, catalog, cfg.get("quality_rules", {}).get("feature", {}))
    reports["feature_target_overlap"] = make_feature_target_overlap_report(dataset, reports["feature_availability"], reports["targets"], cfg.get("quality_rules", {}).get("feature_target_overlap", {}))
    reports["ml_readiness"] = make_ml_readiness_report(reports["targets"], reports["feature_target_overlap"], cfg.get("quality_rules", {}).get("ml_readiness", {}))

    warnings.extend(_detect_warnings(cfg, tables, dataset, catalog, reports))
    return reports, warnings


def _context_columns_for_report(protocol: dict[str, Any], cfg: dict[str, Any], context: pd.DataFrame) -> list[str]:
    if context.empty:
        return []
    include_cfg = cfg.get("include", {}).get("context_computed", {})
    feature_sets = protocol.get("context_features", {}).get("feature_sets", {}) or {}
    try:
        cols = resolve_feature_sets(include_cfg.get("feature_sets", []), feature_sets, context.columns, mode=cfg.get("feature_set_resolution", {}).get("mode", "strict"))
    except Exception:
        cols = [c for c in context.columns if c != "date"]
    for item in cfg.get("targets", []) + cfg.get("covariates", []):
        if item.get("source") == "context_computed" and item.get("column") not in cols:
            cols.append(item.get("column"))
    return [c for c in cols if c]


def _detect_warnings(cfg: dict[str, Any], tables: dict[str, pd.DataFrame], dataset: pd.DataFrame, catalog: pd.DataFrame, reports: dict[str, pd.DataFrame]) -> list[str]:
    warnings: list[str] = []
    wcfg = cfg.get("warnings", {}) or {}
    if wcfg.get("detect_duplicate_columns", True):
        for table_id, df in tables.items():
            dups = pd.Series(df.columns)[pd.Series(df.columns).duplicated()].tolist()
            if dups:
                warnings.append(f"{table_id}: duplicate columns: {dups[:10]}")
    if wcfg.get("detect_suspicious_column_names", True):
        tokens = ["qality", "speep", "sikness", "slakness", "sleepness"]
        for table_id, df in tables.items():
            suspicious = [c for c in df.columns if any(t in c.lower() for t in tokens)]
            if suspicious:
                warnings.append(f"{table_id}: suspicious column names: {suspicious[:15]}")
    if wcfg.get("detect_constant_features", True):
        fr = reports.get("feature_availability", pd.DataFrame())
        if not fr.empty:
            const = fr[fr["reason"].astype(str).str.contains("low_unique_values|low_iqr", regex=True, na=False)]
            if not const.empty:
                warnings.append(f"{len(const)} features have low uniqueness or low IQR")
    if wcfg.get("detect_low_overlap_targets", True):
        ml = reports.get("ml_readiness", pd.DataFrame())
        if not ml.empty:
            low = ml[ml["warning"].astype(str) != "ok"]
            if not low.empty:
                warnings.append(f"{len(low)} targets have ML/data readiness warnings")
    return warnings


def _family(feature: str) -> str:
    if "__" in feature:
        return feature.split("__", 1)[0]
    if feature.startswith("hrv_"):
        return "hrv"
    if feature.startswith("morph_"):
        return "morphology"
    return "context"
