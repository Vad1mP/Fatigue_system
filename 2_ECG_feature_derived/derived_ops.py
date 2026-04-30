from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass
class FormulaResult:
    value: float
    valid: bool
    reason: str = ""


def is_missing(x) -> bool:
    return pd.isna(x)


def safe_delta(left: float, right: float) -> FormulaResult:
    if is_missing(left) or is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")

    return FormulaResult(left - right, True)


def safe_percent_delta(left: float, right: float) -> FormulaResult:
    if is_missing(left) or is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")

    if right == 0:
        return FormulaResult(np.nan, False, "percent_delta_division_by_zero")

    return FormulaResult(100.0 * (left - right) / abs(right), True)


def safe_ratio(left: float, right: float) -> FormulaResult:
    if is_missing(left) or is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")

    if right == 0:
        return FormulaResult(np.nan, False, "ratio_division_by_zero")

    return FormulaResult(left / right, True)


def safe_log_ratio(left: float, right: float) -> FormulaResult:
    if is_missing(left) or is_missing(right):
        return FormulaResult(np.nan, False, "missing_value")

    if left <= 0 or right <= 0:
        return FormulaResult(np.nan, False, "log_ratio_requires_positive_values")

    return FormulaResult(math.log(left / right), True)


def safe_robust_z(value: float, center: float, scale: float) -> float:
    if scale == 0 or np.isnan(scale):
        return np.nan
    return (value - center) / scale


PAIRWISE_FORMULAS = {
    "delta": safe_delta,
    "percent_delta": safe_percent_delta,
    "ratio": safe_ratio,
    "log_ratio": safe_log_ratio,
}