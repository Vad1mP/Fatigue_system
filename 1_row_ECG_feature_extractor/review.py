from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FS_BASE, ADC_MIN, ADC_MAX
from features import (
    prepare_ecg_for_hrv,
    prepare_ecg_for_morphology,
    detect_rpeaks_neurokit,
    compute_basic_qc,
    compute_rr_from_rpeaks,
    drop_edge_rpeaks,
    extract_beats_around_rpeaks,
    filter_beats_by_template_correlation,
    compute_median_beat,
    extract_morphology_features_full_v2,
    detrend_beat_linear,
    smooth_beat_savgol,
)

from typing import Any, Dict, List, Optional

# =========================================================
# IO / SERIALIZATION
# =========================================================

def load_ecg_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].astype(float).to_numpy()


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _restore_numpy_fields(obj):
    """
    Рекурсивно восстанавливает numpy-массивы только для известных полей.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"signal", "signal_filtered", "rpeaks", "rpeaks_no_edge"} and v is not None:
                out[k] = np.asarray(v)
            else:
                out[k] = _restore_numpy_fields(v)
        return out
    if isinstance(obj, list):
        return [_restore_numpy_fields(v) for v in obj]
    return obj


def save_review_result(review_result, review_path: Path):
    review_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(review_result)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_review_result(review_path: Path):
    if not review_path.exists():
        return None
    with open(review_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return _restore_numpy_fields(obj)


# =========================================================
# BASIC HELPERS
# =========================================================

def cut_signal_by_time(signal: np.ndarray, sampling_rate: float, start_sec: float, end_sec: float) -> np.ndarray:
    start_idx = max(0, int(round(start_sec * sampling_rate)))
    end_idx = min(len(signal), int(round(end_sec * sampling_rate)))
    return signal[start_idx:end_idx]


def remove_last_peak(rpeaks):
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) == 0:
        return rpeaks
    return rpeaks[:-1]


def add_peak_by_time(ecg_signal, rpeaks, sampling_rate, time_sec, search_radius=10):
    x = np.asarray(ecg_signal, dtype=float)
    idx = int(round(time_sec * sampling_rate))
    idx = max(0, min(idx, len(x) - 1))

    left = max(0, idx - search_radius)
    right = min(len(x), idx + search_radius + 1)

    if right <= left:
        new_peak = idx
    else:
        seg = x[left:right]
        local = np.argmax(np.abs(seg - np.median(seg)))
        new_peak = left + local

    out = np.unique(np.sort(np.append(np.asarray(rpeaks, dtype=int), new_peak)))
    return out


def snap_peak_to_local_extremum(ecg_signal, peak_idx, search_radius=10):
    x = np.asarray(ecg_signal, dtype=float)
    left = max(0, peak_idx - search_radius)
    right = min(len(x), peak_idx + search_radius + 1)

    if right <= left:
        return int(peak_idx)

    seg = x[left:right]
    local = np.argmax(np.abs(seg - np.median(seg)))
    return int(left + local)


def remove_visible_peaks(rpeaks, local_map, local_numbers):
    to_remove = set(local_map[n] for n in local_numbers if n in local_map)
    return np.array([p for p in rpeaks if p not in to_remove], dtype=int)


def snap_visible_peak(ecg_signal, rpeaks, local_map, local_number, search_radius=10):
    if local_number not in local_map:
        return np.asarray(rpeaks, dtype=int)

    old_peak = int(local_map[local_number])
    new_peak = snap_peak_to_local_extremum(
        ecg_signal=ecg_signal,
        peak_idx=old_peak,
        search_radius=search_radius,
    )

    out = np.asarray(rpeaks, dtype=int).copy()
    out[out == old_peak] = new_peak
    out = np.unique(np.sort(out))
    return out


def select_review_rpeaks(signal_review_result, use_no_edge=True):
    if use_no_edge and signal_review_result.get("rpeaks_no_edge") is not None:
        return np.asarray(signal_review_result["rpeaks_no_edge"], dtype=int)
    return np.asarray(signal_review_result["rpeaks"], dtype=int)


def _fmt(v, digits=2):
    try:
        if v is None:
            return "NA"
        v = float(v)
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


# =========================================================
# R-PEAK REVIEW HELPERS
# =========================================================

def find_suspicious_peaks(
    ecg_signal,
    rpeaks,
    sampling_rate,
    edge_guard_sec=0.5,
    rr_min_sec=0.35,
    rr_max_sec=2.0,
    local_ratio_low=0.70,
    local_ratio_high=1.50,
):
    x = np.asarray(ecg_signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)

    suspicious = np.zeros(len(rpeaks), dtype=bool)
    reasons = [[] for _ in range(len(rpeaks))]

    if len(rpeaks) == 0:
        return suspicious, reasons

    edge_guard = int(round(edge_guard_sec * sampling_rate))
    edge_bad = (rpeaks < edge_guard) | (rpeaks >= len(x) - edge_guard)
    for i, bad in enumerate(edge_bad):
        if bad:
            suspicious[i] = True
            reasons[i].append("edge")

    if len(rpeaks) >= 2:
        rr = np.diff(rpeaks) / sampling_rate

        bad_rr = (rr < rr_min_sec) | (rr > rr_max_sec)
        for i, bad in enumerate(bad_rr):
            if bad:
                suspicious[i] = True
                suspicious[i + 1] = True
                reasons[i].append("bad_rr")
                reasons[i + 1].append("bad_rr")

        for i in range(len(rr)):
            left = max(0, i - 2)
            right = min(len(rr), i + 3)
            local_med = np.median(rr[left:right])
            if np.isfinite(local_med) and local_med > 1e-9:
                if rr[i] < local_ratio_low * local_med or rr[i] > local_ratio_high * local_med:
                    suspicious[i] = True
                    suspicious[i + 1] = True
                    reasons[i].append("rr_outlier")
                    reasons[i + 1].append("rr_outlier")

    return suspicious, reasons


def build_suspicious_windows(rpeaks, suspicious_mask, sampling_rate, window_sec=8.0):
    rpeaks = np.asarray(rpeaks, dtype=int)
    suspicious_mask = np.asarray(suspicious_mask, dtype=bool)

    if len(rpeaks) == 0 or len(suspicious_mask) != len(rpeaks):
        return []

    bad_rp = rpeaks[suspicious_mask]
    if len(bad_rp) == 0:
        return []

    bad_times = np.sort(np.unique(np.round(bad_rp / sampling_rate, 3)))

    windows = []
    for t in bad_times:
        start = max(0.0, t - 0.5 * window_sec)
        if len(windows) == 0 or abs(start - windows[-1]) > 0.5 * window_sec:
            windows.append(start)

    return windows


def plot_ecg_review_paged(
    ecg_signal,
    rpeaks,
    sampling_rate,
    window_start_sec=0.0,
    window_sec=15.0,
    suspicious_mask=None,
    title="ECG review",
    figsize=(15, 8),
):
    x = np.asarray(ecg_signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)
    t = np.arange(len(x)) / sampling_rate

    total_sec = len(x) / sampling_rate
    if total_sec <= 0:
        raise ValueError("Пустой сигнал.")

    window_sec = max(2.0, float(window_sec))
    window_start_sec = max(0.0, min(window_start_sec, max(0.0, total_sec - window_sec)))
    window_end_sec = min(total_sec, window_start_sec + window_sec)

    start_idx = int(round(window_start_sec * sampling_rate))
    end_idx = int(round(window_end_sec * sampling_rate))
    visible = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 2]}
    )

    ax_top.plot(t, x, linewidth=0.8)
    ax_top.axvspan(window_start_sec, window_end_sec, alpha=0.2)
    ax_top.set_xlim(0, total_sec)
    ax_top.set_ylabel("Amp")
    ax_top.set_title(title)
    ax_top.grid(alpha=0.3)

    if len(rpeaks) > 0:
        valid = (rpeaks >= 0) & (rpeaks < len(x))
        rp = rpeaks[valid]
        ax_top.scatter(t[rp], x[rp], s=8, zorder=3)

    if suspicious_mask is not None and len(suspicious_mask) == len(rpeaks):
        bad_rp = rpeaks[suspicious_mask]
        if len(bad_rp) > 0:
            ax_top.scatter(t[bad_rp], x[bad_rp], s=22, marker="x", zorder=4)

    tt = t[start_idx:end_idx]
    xx = x[start_idx:end_idx]
    ax_bot.plot(tt, xx, linewidth=1.0)
    ax_bot.set_xlim(window_start_sec, window_end_sec)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Amp")
    ax_bot.grid(alpha=0.3)

    local_map = {}
    for j, rp in enumerate(visible, start=1):
        ax_bot.scatter(t[rp], x[rp], s=28, zorder=4)
        ax_bot.text(
            t[rp],
            x[rp],
            str(j),
            fontsize=9,
            ha="center",
            va="bottom",
        )
        local_map[j] = int(rp)

    plt.tight_layout()
    plt.show()

    return local_map, window_start_sec, window_end_sec, total_sec


def print_rpeak_review_help():
    print("\nДоступные команды:")
    print("  ok")
    print("      принять сигнал и пики для HRV")
    print("  skip")
    print("      отклонить сигнал целиком")
    print("  n")
    print("      следующее окно")
    print("  p")
    print("      предыдущее окно")
    print("  goto T")
    print("      перейти к времени T секунд")
    print("  win SEC")
    print("      изменить ширину окна")
    print("  crop START END")
    print("      обрезать запись по времени")
    print("  recompute")
    print("      заново детектировать пики на текущем сигнале")
    print("  remove_last")
    print("      удалить последний пик")
    print("  removev N1 N2 ...")
    print("      удалить видимые пики по локальным номерам")
    print("  snapv N [RADIUS]")
    print("      передвинуть видимый пик N на локальный экстремум")
    print("  add T [RADIUS]")
    print("      добавить пик около времени T секунд")
    print("  badnext")
    print("      перейти к следующему подозрительному окну")
    print("  badlist")
    print("      показать список подозрительных окон")
    print("  peaks")
    print("      показать видимые пики: локальный номер -> индекс -> время")
    print("  help")
    print("      показать справку\n")


# =========================================================
# MORPHOLOGY REVIEW HELPERS
# =========================================================

def build_morphology_review(
    ecg_signal,
    rpeaks,
    sampling_rate,
    morph_lowcut=0.3,
    morph_highcut=35.0,
    morph_pre_ms=200,
    morph_post_ms=400,
    corr_threshold=0.8,
    smooth_ms=15,
):
    out = {
        "error": None,
        "qc": {
            "n_beats_extracted": 0,
            "n_beats_good": 0,
            "corr_min": np.nan,
            "corr_median": np.nan,
            "corr_max": np.nan,
        },
        "median_beat": None,
        "beat_plot": None,
        "beats_good_plot": [],
        "morph_feats": {"qrs": {}, "p": {}, "t": {}},
    }

    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        out["error"] = "too_few_rpeaks"
        return out

    try:
        ecg_morph, _ = prepare_ecg_for_morphology(
            ecg_signal=ecg_signal,
            sampling_rate=sampling_rate,
            lowcut=morph_lowcut,
            highcut=morph_highcut,
            trim_seconds=0.0,
        )

        beats, valid_rpeaks = extract_beats_around_rpeaks(
            ecg_signal=ecg_morph,
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            pre_ms=morph_pre_ms,
            post_ms=morph_post_ms,
        )

        beats_good, corrs, template, keep_mask = filter_beats_by_template_correlation(
            beats,
            corr_threshold=corr_threshold,
        )

        median_beat = compute_median_beat(beats_good)

        out["qc"] = {
            "n_beats_extracted": int(len(beats)),
            "n_beats_good": int(len(beats_good)),
            "corr_min": float(np.nanmin(corrs)) if len(corrs) > 0 else np.nan,
            "corr_median": float(np.nanmedian(corrs)) if len(corrs) > 0 else np.nan,
            "corr_max": float(np.nanmax(corrs)) if len(corrs) > 0 else np.nan,
        }

        if median_beat is None:
            out["error"] = "no_median_beat"
            return out

        morph_feats = extract_morphology_features_full_v2(
            median_beat=median_beat,
            sampling_rate=sampling_rate,
            pre_ms=morph_pre_ms,
            search_ms=60,
            smooth_ms=smooth_ms,
        )

        beat_plot, _ = detrend_beat_linear(
            median_beat,
            sampling_rate=sampling_rate,
            edge_ms=40,
        )
        beat_plot = smooth_beat_savgol(
            beat_plot,
            sampling_rate=sampling_rate,
            win_ms=smooth_ms,
            polyorder=3,
        )

        inverted = bool(morph_feats.get("qrs", {}).get("QRS_inverted_for_analysis", False))
        if inverted:
            beat_plot = -beat_plot

        beats_good_plot = []
        for b in beats_good[:20]:
            bb, _ = detrend_beat_linear(
                b,
                sampling_rate=sampling_rate,
                edge_ms=40,
            )
            bb = smooth_beat_savgol(
                bb,
                sampling_rate=sampling_rate,
                win_ms=smooth_ms,
                polyorder=3,
            )
            if inverted:
                bb = -bb
            beats_good_plot.append(bb)

        out["median_beat"] = median_beat
        out["beat_plot"] = beat_plot
        out["beats_good_plot"] = beats_good_plot
        out["morph_feats"] = morph_feats
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def morphology_qc_gate(morph_debug, morphology_qc_profile=None):
    """
    Gate для качества морфологии.

    Принимает результат build_morphology_review().
    Проверяет:
    - есть ли ошибка построения morphology debug
    - достаточно ли beats
    - достаточно ли good beats
    - достаточно ли высокий corr_median
    - найден ли QRS
    - попадает ли QRS_duration в допустимый диапазон
    """
    profile = morphology_qc_profile or {}

    min_beats_extracted = profile.get("min_beats_extracted", 0)
    min_beats_good = profile.get("min_beats_good", 0)
    good_beats_ratio_min = profile.get("good_beats_ratio_min", None)
    corr_median_min = profile.get("corr_median_min", None)
    corr_min_min = profile.get("corr_min_min", None)

    require_qrs = bool(profile.get("require_qrs", True))
    qrs_duration_min_ms = profile.get("qrs_duration_min_ms", None)
    qrs_duration_max_ms = profile.get("qrs_duration_max_ms", None)

    require_p = bool(profile.get("require_p", False))
    require_t = bool(profile.get("require_t", False))

    failed = False
    reasons = []

    if morph_debug is None:
        return {
            "passed": False,
            "stage": "morphology",
            "metrics": {},
            "thresholds": profile,
            "reasons": ["morph_debug is None"],
        }

    error = morph_debug.get("error")
    if error is not None:
        failed = True
        reasons.append(f"morphology_error: {error}")

    qc = morph_debug.get("qc", {}) or {}
    feats = morph_debug.get("morph_feats", {}) or {}

    qrs = feats.get("qrs", {}) or {}
    p = feats.get("p", {}) or {}
    t = feats.get("t", {}) or {}

    n_beats_extracted = qc.get("n_beats_extracted", 0)
    n_beats_good = qc.get("n_beats_good", 0)
    corr_min = qc.get("corr_min", np.nan)
    corr_median = qc.get("corr_median", np.nan)
    corr_max = qc.get("corr_max", np.nan)

    if n_beats_extracted < min_beats_extracted:
        failed = True
        reasons.append(f"n_beats_extracted={n_beats_extracted} < {min_beats_extracted}")

    if n_beats_good < min_beats_good:
        failed = True
        reasons.append(f"n_beats_good={n_beats_good} < {min_beats_good}")

    good_beats_ratio = (
        float(n_beats_good) / float(n_beats_extracted)
        if n_beats_extracted and n_beats_extracted > 0
        else np.nan
    )

    if good_beats_ratio_min is not None:
        if not np.isfinite(good_beats_ratio) or good_beats_ratio < good_beats_ratio_min:
            failed = True
            reasons.append(f"good_beats_ratio={good_beats_ratio} < {good_beats_ratio_min}")

    if corr_median_min is not None:
        if not np.isfinite(corr_median) or corr_median < corr_median_min:
            failed = True
            reasons.append(f"corr_median={corr_median} < {corr_median_min}")

    if corr_min_min is not None:
        if not np.isfinite(corr_min) or corr_min < corr_min_min:
            failed = True
            reasons.append(f"corr_min={corr_min} < {corr_min_min}")

    qrs_idx = qrs.get("QRS_main_idx", np.nan)
    qrs_duration = qrs.get("QRS_duration_ms", np.nan)

    if require_qrs:
        if not np.isfinite(qrs_idx):
            failed = True
            reasons.append("QRS_main_idx is missing")

        if not np.isfinite(qrs_duration):
            failed = True
            reasons.append("QRS_duration_ms is missing")

    if qrs_duration_min_ms is not None:
        if not np.isfinite(qrs_duration) or qrs_duration < qrs_duration_min_ms:
            failed = True
            reasons.append(f"QRS_duration_ms={qrs_duration} < {qrs_duration_min_ms}")

    if qrs_duration_max_ms is not None:
        if not np.isfinite(qrs_duration) or qrs_duration > qrs_duration_max_ms:
            failed = True
            reasons.append(f"QRS_duration_ms={qrs_duration} > {qrs_duration_max_ms}")

    if require_p and not bool(p.get("P_present", False)):
        failed = True
        reasons.append("P wave required but not detected")

    if require_t and not bool(t.get("T_present", False)):
        failed = True
        reasons.append("T wave required but not detected")

    return {
        "passed": not failed,
        "stage": "morphology",
        "metrics": {
            "n_beats_extracted": n_beats_extracted,
            "n_beats_good": n_beats_good,
            "good_beats_ratio": good_beats_ratio,
            "corr_min": corr_min,
            "corr_median": corr_median,
            "corr_max": corr_max,
            "QRS_main_idx": qrs_idx,
            "QRS_duration_ms": qrs_duration,
            "P_present": bool(p.get("P_present", False)),
            "T_present": bool(t.get("T_present", False)),
        },
        "thresholds": {
            "min_beats_extracted": min_beats_extracted,
            "min_beats_good": min_beats_good,
            "good_beats_ratio_min": good_beats_ratio_min,
            "corr_median_min": corr_median_min,
            "corr_min_min": corr_min_min,
            "require_qrs": require_qrs,
            "qrs_duration_min_ms": qrs_duration_min_ms,
            "qrs_duration_max_ms": qrs_duration_max_ms,
            "require_p": require_p,
            "require_t": require_t,
        },
        "reasons": reasons,
    }


def _scatter_labeled(ax, x, y, label, dy=0.0):
    ax.scatter([x], [y], s=35, zorder=5)
    ax.text(x, y + dy, label, fontsize=9, ha="center", va="bottom")


def plot_morphology_review(
    morph_debug,
    sampling_rate,
    title="Morphology review",
    figsize=(12, 6),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Time relative to R (ms)")
    ax.set_ylabel("Amp")

    if morph_debug is None:
        ax.text(0.5, 0.5, "Morphology debug is None", transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.show()
        return

    if morph_debug.get("error") is not None:
        ax.text(
            0.5, 0.5,
            f"Morphology unavailable:\n{morph_debug['error']}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        plt.tight_layout()
        plt.show()
        return

    beat = morph_debug.get("beat_plot")
    feats = morph_debug.get("morph_feats", {})
    qrs = feats.get("qrs", {})
    p = feats.get("p", {})
    t = feats.get("t", {})

    if beat is None or len(beat) == 0:
        ax.text(0.5, 0.5, "No median beat", transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.show()
        return

    qrs_idx = qrs.get("QRS_main_idx")
    if qrs_idx is None or not np.isfinite(qrs_idx):
        ax.text(0.5, 0.5, "No QRS peak index", transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.show()
        return

    qrs_idx = int(qrs_idx)
    t_ms = (np.arange(len(beat)) - qrs_idx) / sampling_rate * 1000.0

    for bb in morph_debug.get("beats_good_plot", []):
        if len(bb) == len(beat):
            ax.plot(t_ms, bb, linewidth=0.8, alpha=0.25)

    ax.plot(t_ms, beat, linewidth=2.0)
    ax.axhline(0.0, linewidth=0.8, alpha=0.4)

    def idx_to_t(idx):
        if idx is None or not np.isfinite(idx):
            return None
        return (int(idx) - qrs_idx) / sampling_rate * 1000.0

    qrs_on = qrs.get("QRS_onset_idx")
    qrs_off = qrs.get("QRS_offset_idx")
    if qrs_on is not None and qrs_off is not None and np.isfinite(qrs_on) and np.isfinite(qrs_off):
        qrs_on = int(qrs_on)
        qrs_off = int(qrs_off)
        ax.axvspan(idx_to_t(qrs_on), idx_to_t(qrs_off), alpha=0.12)
        ax.axvline(idx_to_t(qrs_on), linestyle="--", alpha=0.7)
        ax.axvline(idx_to_t(qrs_off), linestyle="--", alpha=0.7)

    dy = 0.03 * np.ptp(beat) if np.ptp(beat) > 0 else 0.0
    _scatter_labeled(ax, 0.0, beat[qrs_idx], "R", dy=dy)

    # Q/S: сначала пробуем явные индексы, если их нет — fallback через интервалы
    q_idx = qrs.get("Q_idx")
    s_idx = qrs.get("S_idx")

    if (q_idx is None or not np.isfinite(q_idx)) and np.isfinite(qrs.get("RQ_interval_ms", np.nan)):
        q_idx = int(round(qrs_idx - float(qrs["RQ_interval_ms"]) * sampling_rate / 1000.0))
    if (s_idx is None or not np.isfinite(s_idx)) and np.isfinite(qrs.get("RS_interval_ms", np.nan)):
        s_idx = int(round(qrs_idx + float(qrs["RS_interval_ms"]) * sampling_rate / 1000.0))

    if q_idx is not None and np.isfinite(q_idx):
        q_idx = int(q_idx)
        if 0 <= q_idx < len(beat):
            _scatter_labeled(ax, idx_to_t(q_idx), beat[q_idx], "Q")
    if s_idx is not None and np.isfinite(s_idx):
        s_idx = int(s_idx)
        if 0 <= s_idx < len(beat):
            _scatter_labeled(ax, idx_to_t(s_idx), beat[s_idx], "S")

    if bool(p.get("P_present", False)):
        p_on = p.get("P_onset_idx")
        p_off = p.get("P_offset_idx")
        p_idx = p.get("P_peak_idx")

        if p_on is not None and p_off is not None and np.isfinite(p_on) and np.isfinite(p_off):
            ax.axvspan(idx_to_t(int(p_on)), idx_to_t(int(p_off)), alpha=0.10)

        if p_idx is not None and np.isfinite(p_idx):
            p_idx = int(p_idx)
            if 0 <= p_idx < len(beat):
                _scatter_labeled(ax, idx_to_t(p_idx), beat[p_idx], "P")

    if bool(t.get("T_present", False)):
        t_on = t.get("T_onset_idx")
        t_off = t.get("T_offset_idx")
        t_idx = t.get("T_peak_idx")

        if t_on is not None and t_off is not None and np.isfinite(t_on) and np.isfinite(t_off):
            ax.axvspan(idx_to_t(int(t_on)), idx_to_t(int(t_off)), alpha=0.10)

        if t_idx is not None and np.isfinite(t_idx):
            t_idx = int(t_idx)
            if 0 <= t_idx < len(beat):
                _scatter_labeled(ax, idx_to_t(t_idx), beat[t_idx], "T")

    qc = morph_debug.get("qc", {})
    info_lines = [
        f"beats good/extracted = {qc.get('n_beats_good', 0)}/{qc.get('n_beats_extracted', 0)}",
        f"corr median          = {_fmt(qc.get('corr_median'), 3)}",
        f"QRS duration ms      = {_fmt(qrs.get('QRS_duration_ms'))}",
        f"R width half ms      = {_fmt(qrs.get('R_width_half_ms'))}",
        f"PR interval ms       = {_fmt(p.get('PR_interval_ms'))}",
        f"QT_like ms           = {_fmt(t.get('QT_like_ms'))}",
        f"P amp                = {_fmt(p.get('P_amp'), 3)}",
        f"Q amp                = {_fmt(qrs.get('Q_amp'), 3)}",
        f"R amp                = {_fmt(qrs.get('QRS_main_amp'), 3)}",
        f"S amp                = {_fmt(qrs.get('S_amp'), 3)}",
        f"T amp                = {_fmt(t.get('T_amp'), 3)}",
    ]
    ax.text(
        0.01, 0.99,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


def print_morphology_review_help():
    print("\nДоступные команды морфологического review:")
    print("  ok")
    print("      принять морфологию")
    print("  skip")
    print("      отклонить морфологию, но сигнал для HRV оставить")
    print("  corr THR")
    print("      изменить threshold для template correlation (0..1)")
    print("  recompute")
    print("      пересчитать морфологию с текущим threshold")
    print("  help")
    print("      показать справку\n")


# =========================================================
# REVIEW STAGE 1: R-PEAKS
# =========================================================

def review_rpeaks_record(
    ecg_signal,
    sampling_rate=FS_BASE,
    date=None,
    phase=None,
    record_type=None,
    detector_method="neurokit",
    lowcut=0.5,
    highcut=30.0,
    trim_seconds=0.0,
    edge_guard_sec=0.5,
    adc_min=ADC_MIN,
    adc_max=ADC_MAX,
    initial_window_sec=15.0,
):
    original_signal = np.asarray(ecg_signal, dtype=float).copy()
    current_signal = original_signal.copy()
    current_trim_seconds = trim_seconds

    action_log = []
    window_sec = float(initial_window_sec)
    window_start_sec = 0.0

    def recompute_current(signal, trim_seconds_local):
        ecg_raw_trimmed, ecg_filt_trimmed, trim_n = prepare_ecg_for_hrv(
            ecg_signal=signal,
            sampling_rate=sampling_rate,
            lowcut=lowcut,
            highcut=highcut,
            trim_seconds=trim_seconds_local,
        )

        rpeaks = detect_rpeaks_neurokit(
            ecg_filtered=ecg_filt_trimmed,
            sampling_rate=sampling_rate,
            method=detector_method,
        )

        return ecg_raw_trimmed, ecg_filt_trimmed, rpeaks, trim_n

    ecg_raw_trimmed, ecg_filt_trimmed, rpeaks, trim_n = recompute_current(
        current_signal,
        current_trim_seconds,
    )

    while True:
        qc = compute_basic_qc(
            ecg_signal=ecg_raw_trimmed,
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            adc_min=adc_min,
            adc_max=adc_max,
        )

        rpeaks_no_edge = drop_edge_rpeaks(
            rpeaks,
            len(ecg_raw_trimmed),
            sampling_rate,
            edge_guard_sec=edge_guard_sec,
        )

        rr_all = compute_rr_from_rpeaks(rpeaks, sampling_rate)
        rr_no_edge = compute_rr_from_rpeaks(rpeaks_no_edge, sampling_rate)

        suspicious_mask, suspicious_reasons = find_suspicious_peaks(
            ecg_signal=ecg_filt_trimmed,
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            edge_guard_sec=edge_guard_sec,
        )

        suspicious_windows = build_suspicious_windows(
            rpeaks=rpeaks,
            suspicious_mask=suspicious_mask,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
        )

        meta_str = " | ".join([
            str(date) if date is not None else "date=?",
            str(phase) if phase is not None else "phase=?",
            str(record_type) if record_type is not None else "record_type=?",
            f"n_peaks={len(rpeaks)}",
            f"n_no_edge={len(rpeaks_no_edge)}",
            f"HR≈{qc['mean_hr_bpm_rough']:.1f}" if np.isfinite(qc["mean_hr_bpm_rough"]) else "HR=?",
            f"trim={current_trim_seconds}s",
            f"guard={edge_guard_sec}s",
            f"win={window_sec}s",
        ])

        local_map, ws, we, total_sec = plot_ecg_review_paged(
            ecg_signal=ecg_filt_trimmed,
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            window_start_sec=window_start_sec,
            window_sec=window_sec,
            suspicious_mask=suspicious_mask,
            title=meta_str,
        )

        print("\nQC:")
        print(f"  clipping_ratio          = {qc['clipping_ratio']}")
        print(f"  rr_phys_bad_ratio       = {qc['rr_phys_bad_ratio']}")
        print(f"  n_rpeaks_all            = {len(rpeaks)}")
        print(f"  n_rpeaks_no_edge        = {len(rpeaks_no_edge)}")
        print(f"  mean_hr_bpm_all         = {qc['mean_hr_bpm_rough']}")
        print(f"  suspicious_peaks        = {int(np.sum(suspicious_mask))}")
        print(f"  suspicious_windows      = {len(suspicious_windows)}")

        if len(rr_all) > 0:
            print(f"  RR_all median [sec]     = {np.median(rr_all):.4f}")
        if len(rr_no_edge) > 0:
            print(f"  RR_no_edge median [sec] = {np.median(rr_no_edge):.4f}")

        print(f"  current_window          = [{ws:.2f}, {we:.2f}] / {total_sec:.2f} sec")
        print_rpeak_review_help()

        cmd = input("Команда: ").strip()
        if not cmd:
            continue

        parts = cmd.split()
        op = parts[0].lower()

        if op == "ok":
            action_log.append({
                "action": "ok",
                "trim_seconds": current_trim_seconds,
                "edge_guard_sec": edge_guard_sec,
            })

            return {
                "status": "accepted",
                "signal": ecg_raw_trimmed,
                "signal_filtered": ecg_filt_trimmed,
                "rpeaks": np.asarray(rpeaks, dtype=int),
                "rpeaks_no_edge": np.asarray(rpeaks_no_edge, dtype=int),
                "sampling_rate": sampling_rate,
                "log": action_log,
                "qc": qc,
            }

        elif op == "skip":
            action_log.append({"action": "skip"})
            return {
                "status": "skipped",
                "signal": None,
                "signal_filtered": None,
                "rpeaks": None,
                "rpeaks_no_edge": None,
                "sampling_rate": sampling_rate,
                "log": action_log,
                "qc": qc,
            }

        elif op == "n":
            window_start_sec = min(
                max(0.0, total_sec - window_sec),
                window_start_sec + window_sec
            )
            action_log.append({
                "action": "next_window",
                "window_start_sec": window_start_sec,
                "window_sec": window_sec,
            })

        elif op == "p":
            window_start_sec = max(0.0, window_start_sec - window_sec)
            action_log.append({
                "action": "prev_window",
                "window_start_sec": window_start_sec,
                "window_sec": window_sec,
            })

        elif op == "goto":
            if len(parts) != 2:
                print("Нужно: goto T")
                continue
            try:
                t_sec = float(parts[1])
            except ValueError:
                print("T должно быть числом.")
                continue

            window_start_sec = max(0.0, min(t_sec, max(0.0, total_sec - window_sec)))
            action_log.append({
                "action": "goto",
                "time_sec": t_sec,
                "window_start_sec": window_start_sec,
            })

        elif op == "win":
            if len(parts) != 2:
                print("Нужно: win SEC")
                continue
            try:
                new_window_sec = float(parts[1])
            except ValueError:
                print("SEC должно быть числом.")
                continue

            if new_window_sec < 2.0:
                print("Минимум 2 секунды.")
                continue

            window_sec = new_window_sec
            window_start_sec = max(0.0, min(window_start_sec, max(0.0, total_sec - window_sec)))
            action_log.append({
                "action": "win",
                "window_sec": window_sec,
            })

        elif op == "crop":
            if len(parts) != 3:
                print("Нужно: crop START END")
                continue
            try:
                start_sec = float(parts[1])
                end_sec = float(parts[2])
            except ValueError:
                print("START и END должны быть числами.")
                continue

            if end_sec <= start_sec:
                print("END должен быть больше START.")
                continue

            current_signal = cut_signal_by_time(
                current_signal,
                sampling_rate,
                start_sec,
                end_sec,
            )

            action_log.append({
                "action": "crop",
                "start_sec": start_sec,
                "end_sec": end_sec,
            })

            try:
                ecg_raw_trimmed, ecg_filt_trimmed, rpeaks, trim_n = recompute_current(
                    current_signal,
                    current_trim_seconds,
                )
                window_start_sec = 0.0
            except Exception as e:
                print(f"Ошибка после crop: {e}")
                continue

        elif op == "recompute":
            try:
                ecg_raw_trimmed, ecg_filt_trimmed, rpeaks, trim_n = recompute_current(
                    current_signal,
                    current_trim_seconds,
                )
            except Exception as e:
                print(f"Ошибка при recompute: {e}")
                continue

            action_log.append({
                "action": "recompute",
                "trim_seconds": current_trim_seconds,
            })

        elif op == "remove_last":
            old_n = len(rpeaks)
            rpeaks = remove_last_peak(rpeaks)
            action_log.append({
                "action": "remove_last",
                "n_before": old_n,
                "n_after": len(rpeaks),
            })

        elif op == "removev":
            if len(parts) < 2:
                print("Нужно: removev N1 N2 ...")
                continue
            try:
                nums = [int(x) for x in parts[1:]]
            except ValueError:
                print("Локальные номера должны быть целыми.")
                continue

            before = len(rpeaks)
            rpeaks = remove_visible_peaks(rpeaks, local_map, nums)
            after = len(rpeaks)

            action_log.append({
                "action": "removev",
                "visible_numbers": nums,
                "n_before": before,
                "n_after": after,
            })

        elif op == "snapv":
            if len(parts) not in (2, 3):
                print("Нужно: snapv N [RADIUS]")
                continue

            try:
                n_local = int(parts[1])
                radius = int(parts[2]) if len(parts) == 3 else 10
            except ValueError:
                print("N и RADIUS должны быть целыми.")
                continue

            if n_local not in local_map:
                print("Такого локального номера нет в текущем окне.")
                continue

            old_peak = local_map[n_local]
            rpeaks = snap_visible_peak(
                ecg_signal=ecg_filt_trimmed,
                rpeaks=rpeaks,
                local_map=local_map,
                local_number=n_local,
                search_radius=radius,
            )

            action_log.append({
                "action": "snapv",
                "visible_number": n_local,
                "old_peak": int(old_peak),
                "radius": radius,
            })

        elif op == "add":
            if len(parts) not in (2, 3):
                print("Нужно: add T [RADIUS]")
                continue

            try:
                time_sec = float(parts[1])
                radius = int(parts[2]) if len(parts) == 3 else 10
            except ValueError:
                print("T должно быть числом, RADIUS целым.")
                continue

            rpeaks = add_peak_by_time(
                ecg_signal=ecg_filt_trimmed,
                rpeaks=rpeaks,
                sampling_rate=sampling_rate,
                time_sec=time_sec,
                search_radius=radius,
            )

            action_log.append({
                "action": "add",
                "time_sec": time_sec,
                "radius": radius,
            })

        elif op == "badnext":
            if len(suspicious_windows) == 0:
                print("Подозрительных окон не найдено.")
                continue

            current_center = window_start_sec + 0.5 * window_sec
            future = [w for w in suspicious_windows if (w + 0.5 * window_sec) > current_center]

            if len(future) == 0:
                print("Следующих подозрительных окон нет.")
                continue

            window_start_sec = future[0]
            action_log.append({
                "action": "badnext",
                "window_start_sec": window_start_sec,
            })

        elif op == "badlist":
            if len(suspicious_windows) == 0:
                print("Подозрительных окон не найдено.")
            else:
                print("Подозрительные окна:")
                for i, w in enumerate(suspicious_windows, start=1):
                    print(f"  {i:>2}. [{w:.2f}, {w + window_sec:.2f}] sec")
            action_log.append({
                "action": "badlist",
                "n_windows": len(suspicious_windows),
            })

        elif op == "peaks":
            if len(local_map) == 0:
                print("В текущем окне видимых пиков нет.")
            else:
                print("Видимые пики:")
                for n_local, abs_idx in local_map.items():
                    t_sec = abs_idx / sampling_rate
                    print(f"  {n_local:>2} -> idx={abs_idx}, t={t_sec:.3f} sec")
            action_log.append({
                "action": "peaks",
                "n_visible": len(local_map),
            })

        elif op == "help":
            print_rpeak_review_help()

        else:
            print("Неизвестная команда. Введи help для справки.")


# =========================================================
# REVIEW STAGE 2: MORPHOLOGY
# =========================================================

def review_morphology_record(
        ecg_signal,
        rpeaks,
        sampling_rate=FS_BASE,
        date=None,
        phase=None,
        record_type=None,
        morph_lowcut=0.3,
        morph_highcut=35.0,
        morph_pre_ms=200,
        morph_post_ms=400,
        corr_threshold=0.8,
        smooth_ms=15,
        morphology_qc_profile=None,
):
    signal = np.asarray(ecg_signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)

    current_corr_threshold = float(corr_threshold)
    action_log = []

    while True:
        morph_debug = build_morphology_review(
            ecg_signal=signal,
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            morph_lowcut=morph_lowcut,
            morph_highcut=morph_highcut,
            morph_pre_ms=morph_pre_ms,
            morph_post_ms=morph_post_ms,
            corr_threshold=current_corr_threshold,
            smooth_ms=smooth_ms,
        )

        morph_gate = morphology_qc_gate(
            morph_debug=morph_debug,
            morphology_qc_profile=morphology_qc_profile,
        )

        meta_str = " | ".join([
            str(date) if date is not None else "date=?",
            str(phase) if phase is not None else "phase=?",
            str(record_type) if record_type is not None else "record_type=?",
            f"corr_thr={current_corr_threshold:.2f}",
        ])

        plot_morphology_review(
            morph_debug=morph_debug,
            sampling_rate=sampling_rate,
            title=meta_str,
        )

        print("\nMorphology:")
        if morph_debug.get("error") is not None:
            print(f"  error                  = {morph_debug['error']}")
        else:
            qc = morph_debug["qc"]
            mf = morph_debug["morph_feats"]
            qrs = mf.get("qrs", {})
            p = mf.get("p", {})
            t = mf.get("t", {})

            print(f"  n_beats_extracted      = {qc.get('n_beats_extracted')}")
            print(f"  n_beats_good           = {qc.get('n_beats_good')}")
            print(f"  corr_min               = {qc.get('corr_min')}")
            print(f"  corr_median            = {qc.get('corr_median')}")
            print(f"  corr_max               = {qc.get('corr_max')}")
            print(f"  QRS_duration_ms        = {qrs.get('QRS_duration_ms')}")
            print(f"  R_width_half_ms        = {qrs.get('R_width_half_ms')}")
            print(f"  PR_interval_ms         = {p.get('PR_interval_ms')}")
            print(f"  QT_like_ms             = {t.get('QT_like_ms')}")
            print(f"  P_amp                  = {p.get('P_amp')}")
            print(f"  Q_amp                  = {qrs.get('Q_amp')}")
            print(f"  R_amp                  = {qrs.get('QRS_main_amp')}")
            print(f"  S_amp                  = {qrs.get('S_amp')}")
            print(f"  T_amp                  = {t.get('T_amp')}")
            print("\nMorphology QC gate:")
            print(f"  passed                 = {morph_gate['passed']}")
            if len(morph_gate["reasons"]) > 0:
                print("  reasons:")
                for reason in morph_gate["reasons"]:
                    print(f"    - {reason}")

        print_morphology_review_help()

        cmd = input("Команда morphology: ").strip()
        if not cmd:
            continue

        parts = cmd.split()
        op = parts[0].lower()

        if op == "ok":
            action_log.append({
                "action": "ok",
                "corr_threshold": current_corr_threshold,
            })
            return {
                "status": "accepted",
                "qc": morph_debug.get("qc"),
                "qc_gate": morph_gate,
                "features_preview": morph_debug.get("morph_feats"),
                "corr_threshold": current_corr_threshold,
                "error": morph_debug.get("error"),
                "log": action_log,
            }

        elif op == "skip":
            action_log.append({
                "action": "skip",
                "corr_threshold": current_corr_threshold,
            })
            return {
                "status": "skipped",
                "qc": morph_debug.get("qc"),
                "qc_gate": morph_gate,
                "features_preview": morph_debug.get("morph_feats"),
                "corr_threshold": current_corr_threshold,
                "error": morph_debug.get("error"),
                "log": action_log,
            }

        elif op == "corr":
            if len(parts) != 2:
                print("Нужно: corr THR")
                continue
            try:
                thr = float(parts[1])
            except ValueError:
                print("THR должно быть числом.")
                continue

            if not (0.0 <= thr <= 1.0):
                print("Ожидается порог в диапазоне 0..1")
                continue

            current_corr_threshold = thr
            action_log.append({
                "action": "corr",
                "corr_threshold": current_corr_threshold,
            })

        elif op == "recompute":
            action_log.append({
                "action": "recompute",
                "corr_threshold": current_corr_threshold,
            })
            continue

        elif op == "help":
            print_morphology_review_help()

        else:
            print("Неизвестная команда. Введи help для справки.")


# =========================================================
# ORCHESTRATOR
# =========================================================

def review_ecg_record(
    ecg_signal,
    sampling_rate=FS_BASE,
    date=None,
    phase=None,
    record_type=None,
    detector_method="neurokit",
    lowcut=0.5,
    highcut=30.0,
    trim_seconds=0.0,
    edge_guard_sec=0.5,
    adc_min=ADC_MIN,
    adc_max=ADC_MAX,
    initial_window_sec=15.0,
    morph_lowcut=0.3,
    morph_highcut=35.0,
    morph_pre_ms=200,
    morph_post_ms=400,
    corr_threshold=0.8,
    smooth_ms=15,
    morphology_qc_profile=None,
):
    signal_review = review_rpeaks_record(
        ecg_signal=ecg_signal,
        sampling_rate=sampling_rate,
        date=date,
        phase=phase,
        record_type=record_type,
        detector_method=detector_method,
        lowcut=lowcut,
        highcut=highcut,
        trim_seconds=trim_seconds,
        edge_guard_sec=edge_guard_sec,
        adc_min=adc_min,
        adc_max=adc_max,
        initial_window_sec=initial_window_sec,
    )

    if signal_review["status"] != "accepted":
        return {
            "signal_review": signal_review,
            "morphology_review": {
                "status": "not_run",
                "qc": None,
                "features_preview": None,
                "corr_threshold": None,
                "error": None,
                "log": [],
            },
            "meta": {
                "date": date,
                "phase": phase,
                "record_type": record_type,
                "sampling_rate": sampling_rate,
            },
        }

    rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=True)
    if len(rpeaks_for_morph) < 2:
        rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=False)

    morphology_review = review_morphology_record(
        ecg_signal=signal_review["signal"],
        rpeaks=rpeaks_for_morph,
        sampling_rate=sampling_rate,
        date=date,
        phase=phase,
        record_type=record_type,
        morph_lowcut=morph_lowcut,
        morph_highcut=morph_highcut,
        morph_pre_ms=morph_pre_ms,
        morph_post_ms=morph_post_ms,
        corr_threshold=corr_threshold,
        smooth_ms=smooth_ms,
        morphology_qc_profile=morphology_qc_profile,
    )

    return {
        "signal_review": signal_review,
        "morphology_review": morphology_review,
        "qc_gate": None,
        "meta": {
            "date": date,
            "phase": phase,
            "record_type": record_type,
            "sampling_rate": sampling_rate,
        },
    }


# =========================================================
# AUTO GATE
# =========================================================

def compute_rpeak_amplitude_qc(
    ecg_signal,
    rpeaks,
    sampling_rate,
    amp_ratio_low=0.40,
    amp_ratio_high=2.50,
    min_median_amp=None,
    local_baseline_sec=0.20,
):
    """
    Оценивает амплитудную согласованность R-пиков.

    Амплитуда пика считается как abs(signal[rpeak] - local_baseline),
    где local_baseline — медиана сигнала в окрестности пика.

    Это не медицинская амплитуда R-зубца, а технический QC-признак:
    похожи ли найденные R-пики друг на друга по выраженности.
    """
    x = np.asarray(ecg_signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)

    if len(x) == 0 or len(rpeaks) == 0:
        return {
            "rpeak_amp_median": np.nan,
            "rpeak_amp_min": np.nan,
            "rpeak_amp_max": np.nan,
            "rpeak_amp_bad_ratio": np.nan,
            "rpeak_amp_too_low_ratio": np.nan,
            "rpeak_amp_too_high_ratio": np.nan,
            "rpeak_amp_global_too_low": False,
        }

    radius = max(1, int(round(local_baseline_sec * sampling_rate)))

    amps = []
    for rp in rpeaks:
        if rp < 0 or rp >= len(x):
            continue

        left = max(0, rp - radius)
        right = min(len(x), rp + radius + 1)

        local = x[left:right]
        baseline = np.median(local) if len(local) else np.median(x)

        amp = abs(float(x[rp]) - float(baseline))
        amps.append(amp)

    amps = np.asarray(amps, dtype=float)
    amps = amps[np.isfinite(amps)]

    if len(amps) == 0:
        return {
            "rpeak_amp_median": np.nan,
            "rpeak_amp_min": np.nan,
            "rpeak_amp_max": np.nan,
            "rpeak_amp_bad_ratio": np.nan,
            "rpeak_amp_too_low_ratio": np.nan,
            "rpeak_amp_too_high_ratio": np.nan,
            "rpeak_amp_global_too_low": False,
        }

    med = float(np.median(amps))

    if not np.isfinite(med) or med <= 1e-12:
        too_low_ratio = np.nan
        too_high_ratio = np.nan
        bad_ratio = 1.0
    else:
        ratios = amps / med
        too_low = ratios < amp_ratio_low
        too_high = ratios > amp_ratio_high

        too_low_ratio = float(np.mean(too_low))
        too_high_ratio = float(np.mean(too_high))
        bad_ratio = float(np.mean(too_low | too_high))

    global_too_low = False
    if min_median_amp is not None:
        global_too_low = (not np.isfinite(med)) or (med < float(min_median_amp))

    return {
        "rpeak_amp_median": med,
        "rpeak_amp_min": float(np.min(amps)),
        "rpeak_amp_max": float(np.max(amps)),
        "rpeak_amp_bad_ratio": bad_ratio,
        "rpeak_amp_too_low_ratio": too_low_ratio,
        "rpeak_amp_too_high_ratio": too_high_ratio,
        "rpeak_amp_global_too_low": bool(global_too_low),
    }


def auto_review_gate(
    signal,
    fs,
    signal_qc_profile=None,
    adc_min=0,
    adc_max=675,
    detector_method="neurokit",
    lowcut=0.5,
    highcut=30,
):
    """
    Gate для пригодности записи к HRV/RR-анализу.

    Пользователь в protocol.yaml задаёт expected_hr_bpm,
    а сюда уже приходит технический профиль:
      rr_min_sec
      rr_max_sec
      bad_rr_fraction_max
      suspicious_peak_fraction_max
      etc.
    """
    profile = signal_qc_profile or {}

    rr_min_sec = profile.get("rr_min_sec", 0.3)
    rr_max_sec = profile.get("rr_max_sec", 2.0)

    bad_rr_fraction_max = profile.get("bad_rr_fraction_max", 0.10)
    suspicious_peak_fraction_max = profile.get("suspicious_peak_fraction_max", 0.15)
    min_rpeaks = profile.get("min_rpeaks", 3)
    edge_guard_sec = profile.get("edge_guard_sec", 0.5)

    local_rr_ratio_low = profile.get("local_rr_ratio_low", 0.70)
    local_rr_ratio_high = profile.get("local_rr_ratio_high", 1.50)

    amp_ratio_low = profile.get("rpeak_amp_ratio_low", 0.40)
    amp_ratio_high = profile.get("rpeak_amp_ratio_high", 2.50)
    amp_bad_ratio_max = profile.get("rpeak_amp_bad_ratio_max", 0.15)
    amp_median_min = profile.get("rpeak_amp_median_min", None)

    try:
        ecg_raw, ecg_filt, _ = prepare_ecg_for_hrv(
            ecg_signal=signal,
            sampling_rate=fs,
            lowcut=lowcut,
            highcut=highcut,
            trim_seconds=0.0,
        )

        rpeaks = detect_rpeaks_neurokit(
            ecg_filtered=ecg_filt,
            sampling_rate=fs,
            method=detector_method,
        )

        qc = compute_basic_qc(
            ecg_signal=ecg_raw,
            rpeaks=rpeaks,
            sampling_rate=fs,
            adc_min=adc_min,
            adc_max=adc_max,
            rr_min_sec=rr_min_sec,
            rr_max_sec=rr_max_sec,
        )

        suspicious_mask, _ = find_suspicious_peaks(
            ecg_signal=ecg_filt,
            rpeaks=rpeaks,
            sampling_rate=fs,
            edge_guard_sec=edge_guard_sec,
            rr_min_sec=rr_min_sec,
            rr_max_sec=rr_max_sec,
            local_ratio_low=local_rr_ratio_low,
            local_ratio_high=local_rr_ratio_high,
        )

        suspicious_ratio = (
            float(np.mean(suspicious_mask))
            if len(suspicious_mask) > 0
            else 0.0
        )

        amp_qc = compute_rpeak_amplitude_qc(
            ecg_signal=ecg_filt,
            rpeaks=rpeaks,
            sampling_rate=fs,
            amp_ratio_low=amp_ratio_low,
            amp_ratio_high=amp_ratio_high,
            min_median_amp=amp_median_min,
        )

        n_rpeaks = int(len(rpeaks))

        failed = False
        reasons = []

        rr_bad = qc.get("rr_phys_bad_ratio")
        if np.isfinite(rr_bad) and rr_bad > bad_rr_fraction_max:
            failed = True
            reasons.append(f"bad_rr_fraction={rr_bad} > {bad_rr_fraction_max}")

        if suspicious_ratio > suspicious_peak_fraction_max:
            failed = True
            reasons.append(
                f"suspicious_peak_fraction={suspicious_ratio} > {suspicious_peak_fraction_max}"
            )

        if n_rpeaks < min_rpeaks:
            failed = True
            reasons.append(f"n_rpeaks={n_rpeaks} < {min_rpeaks}")

        amp_bad_ratio = amp_qc.get("rpeak_amp_bad_ratio")
        if np.isfinite(amp_bad_ratio) and amp_bad_ratio > amp_bad_ratio_max:
            failed = True
            reasons.append(f"rpeak_amp_bad_ratio={amp_bad_ratio} > {amp_bad_ratio_max}")

        if amp_qc.get("rpeak_amp_global_too_low", False):
            failed = True
            reasons.append(
                f"rpeak_amp_median={amp_qc.get('rpeak_amp_median')} < {amp_median_min}"
            )

        metrics = {
            "n_rpeaks": n_rpeaks,
            "rr_min_sec": rr_min_sec,
            "rr_max_sec": rr_max_sec,
            "suspicious_ratio": suspicious_ratio,
            "rr_phys_bad_ratio": rr_bad,
            **amp_qc,
        }

        return {
            "passed": not failed,
            "need_review": failed,
            "stage": "signal",
            "qc": qc,
            "metrics": metrics,
            "thresholds": profile,
            "reasons": reasons,
        }

    except Exception as e:
        return {
            "passed": False,
            "need_review": True,
            "stage": "signal",
            "qc": None,
            "metrics": {},
            "thresholds": profile,
            "reasons": [f"exception:{type(e).__name__}: {e}"],
        }