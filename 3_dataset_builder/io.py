from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def read_table(table_id: str, table_cfg: dict[str, Any], base_dir: str | Path) -> pd.DataFrame:
    path = resolve_path(table_cfg["path"], base_dir)
    required = bool(table_cfg.get("required", True))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required table '{table_id}' was not found: {path}")
        return pd.DataFrame()

    csv_cfg = table_cfg.get("csv", {}) or {}
    read_kwargs = {
        "sep": csv_cfg.get("sep", ","),
        "decimal": csv_cfg.get("decimal", "."),
        "encoding": csv_cfg.get("encoding", "utf-8"),
    }
    df = pd.read_csv(path, **read_kwargs)

    cleanup = table_cfg.get("cleanup", {}) or {}
    if cleanup.get("drop_empty_unnamed_columns", True):
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    normalize_date_column(df, table_cfg, table_id)
    return df


def normalize_date_column(df: pd.DataFrame, table_cfg: dict[str, Any], table_id: str) -> None:
    if df.empty:
        return
    date_col = table_cfg.get("date_column", "date")
    date_format = table_cfg.get("date_format")
    if date_col not in df.columns:
        raise KeyError(f"Table '{table_id}' has no date column '{date_col}'. Columns: {list(df.columns)[:20]}...")
    parsed = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    bad = parsed.isna() & df[date_col].notna()
    if bad.any():
        examples = df.loc[bad, date_col].astype(str).head(5).tolist()
        raise ValueError(
            f"Could not parse {bad.sum()} dates in table '{table_id}' column '{date_col}' "
            f"with format '{date_format}'. Examples: {examples}"
        )
    df["date"] = parsed.dt.strftime("%Y-%m-%d")


def read_configured_tables(cfg: dict[str, Any], base_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for table_id, table_cfg in (cfg.get("tables") or {}).items():
        tables[table_id] = read_table(table_id, table_cfg, base_dir)
    return tables


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
