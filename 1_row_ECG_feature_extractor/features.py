import numpy as np
from scipy.signal import welch
from scipy.signal import savgol_filter
import neurokit2 as nk

from config import FS_BASE, ADC_MIN, ADC_MAX

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if len(x) > 0 else np.nan


def safe_std(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=ddof)) if len(x) > 1 else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if len(x) > 0 else np.nan


def safe_min(x):
    x = np.asarray(x, dtype=float)
    return float(np.min(x)) if len(x) > 0 else np.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(x)) if len(x) > 0 else np.nan

def compute_dfa(rr_sec, scales_short=(4, 16), scales_long=(16, 64)):
    rr_sec = np.asarray(rr_sec, dtype=float)

    if len(rr_sec) < 20:
        return np.nan, np.nan

    x = rr_sec - np.mean(rr_sec)
    y = np.cumsum(x)

    def compute_fluctuation(n):
        n = int(n)
        if n < 2 or n >= len(y):
            return np.nan

        n_segments = len(y) // n
        if n_segments < 2:
            return np.nan

        rms = []

        for i in range(n_segments):
            seg = y[i * n:(i + 1) * n]
            t = np.arange(len(seg))

            # линейный тренд
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)

            detrended = seg - trend
            rms.append(np.sqrt(np.mean(detrended ** 2)))

        return np.sqrt(np.mean(np.array(rms) ** 2))

    def compute_alpha(scale_range):
        ns = np.arange(scale_range[0], scale_range[1])
        Fs = np.array([compute_fluctuation(n) for n in ns])

        valid = (~np.isnan(Fs)) & (Fs > 0)
        if np.sum(valid) < 3:
            return np.nan

        log_n = np.log(ns[valid])
        log_F = np.log(Fs[valid])

        slope, _ = np.polyfit(log_n, log_F, 1)
        return float(slope)

    alpha1 = compute_alpha(scales_short)
    alpha2 = compute_alpha(scales_long)

    return alpha1, alpha2

# =========================================================
# 2. ПОДГОТОВКА СИГНАЛА
# =========================================================

def prepare_ecg_for_hrv(
        ecg_signal,
        sampling_rate,
        lowcut=0.5,
        highcut=30,
        filter_method="butterworth",
        filter_order=2,
        trim_seconds=0.5,
):
    ecg = np.asarray(ecg_signal, dtype=float)

    ecg_filt = nk.signal_filter(
        ecg,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        method=filter_method,
        order=filter_order,
    )

    trim_n = int(trim_seconds * sampling_rate)

    if trim_n > 0:
        if 2 * trim_n >= len(ecg):
            raise ValueError("Слишком короткий сигнал для обрезки краёв.")
        ecg_raw_trimmed = ecg[trim_n:-trim_n]
        ecg_filt_trimmed = ecg_filt[trim_n:-trim_n]
    else:
        ecg_raw_trimmed = ecg
        ecg_filt_trimmed = ecg_filt

    return ecg_raw_trimmed, ecg_filt_trimmed, trim_n

def prepare_ecg_for_morphology(
        ecg_signal,
        sampling_rate,
        lowcut=0.3,
        highcut=35,
        filter_method="butterworth",
        filter_order=2,
        trim_seconds=0.0,
):
    ecg = np.asarray(ecg_signal, dtype=float)

    ecg_filt = nk.signal_filter(
        ecg,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        method=filter_method,
        order=filter_order,
    )

    trim_n = int(trim_seconds * sampling_rate)

    if trim_n > 0:
        if 2 * trim_n >= len(ecg):
            raise ValueError("Слишком короткий сигнал для обрезки краёв.")
        ecg_trimmed = ecg_filt[trim_n:-trim_n]
    else:
        ecg_trimmed = ecg_filt

    return ecg_trimmed, trim_n


def detect_rpeaks_neurokit(ecg_filtered, sampling_rate, method="neurokit"):
    _, info = nk.ecg_peaks(
        ecg_filtered,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=False,
    )
    return np.asarray(info["ECG_R_Peaks"], dtype=int)

def compute_basic_qc(
    ecg_signal,
    rpeaks,
    sampling_rate,
    adc_min=0,
    adc_max=675,
    rr_min_sec=0.3,
    rr_max_sec=2.0,
):
    x = np.asarray(ecg_signal, dtype=float)

    clip_low = np.mean(x <= adc_min + 2) if len(x) else np.nan
    clip_high = np.mean(x >= adc_max - 2) if len(x) else np.nan
    clipping_ratio = float(clip_low + clip_high) if len(x) else np.nan

    rr_sec = np.diff(rpeaks) / sampling_rate if len(rpeaks) >= 2 else np.array([])

    rr_phys_bad_ratio = (
        float(np.mean((rr_sec < rr_min_sec) | (rr_sec > rr_max_sec)))
        if len(rr_sec) > 0
        else np.nan
    )

    mean_hr = 60.0 / np.mean(rr_sec) if len(rr_sec) > 0 else np.nan

    return {
        "clipping_ratio": clipping_ratio,
        "rr_phys_bad_ratio": rr_phys_bad_ratio,
        "n_rpeaks": int(len(rpeaks)),
        "mean_hr_bpm_rough": float(mean_hr) if np.isfinite(mean_hr) else np.nan,
    }

def compute_rr_from_rpeaks(rpeaks, sampling_rate):
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return np.array([])
    return np.diff(rpeaks) / sampling_rate

def drop_edge_rpeaks(rpeaks, signal_len, sampling_rate, edge_guard_sec=0.5):
    rpeaks = np.asarray(rpeaks, dtype=int)
    guard = int(round(edge_guard_sec * sampling_rate))
    return rpeaks[(rpeaks >= guard) & (rpeaks < signal_len - guard)]

# =========================================================
# 3. HRV: TIME-DOMAIN
# =========================================================

def compute_time_domain_hrv(rr_sec):
    rr_sec = np.asarray(rr_sec, dtype=float)

    if len(rr_sec) < 3:
        return {
            "n_rr": len(rr_sec),
            "MeanNN_ms": np.nan,
            "MedianNN_ms": np.nan,
            "SDNN_ms": np.nan,
            "RMSSD_ms": np.nan,
            "SDSD_ms": np.nan,
            "NN_min_ms": np.nan,
            "NN_max_ms": np.nan,
            "NN_range_ms": np.nan,
            "pNN20_percent": np.nan,
            "pNN50_percent": np.nan,
            "MeanHR_bpm": np.nan,
            "HR_min_bpm": np.nan,
            "HR_max_bpm": np.nan,
            "HR_std_bpm": np.nan,
        }

    rr_ms = rr_sec * 1000.0
    diff_rr_ms = np.diff(rr_ms)

    hr_bpm = 60.0 / rr_sec

    return {
        "n_rr": int(len(rr_sec)),
        "MeanNN_ms": safe_mean(rr_ms),
        "MedianNN_ms": safe_median(rr_ms),
        "SDNN_ms": safe_std(rr_ms),
        "RMSSD_ms": float(np.sqrt(np.mean(diff_rr_ms ** 2))),
        "SDSD_ms": safe_std(diff_rr_ms),
        "NN_min_ms": safe_min(rr_ms),
        "NN_max_ms": safe_max(rr_ms),
        "NN_range_ms": float(safe_max(rr_ms) - safe_min(rr_ms)),
        "pNN20_percent": float(100.0 * np.mean(np.abs(diff_rr_ms) > 20.0)),
        "pNN50_percent": float(100.0 * np.mean(np.abs(diff_rr_ms) > 50.0)),
        "MeanHR_bpm": safe_mean(hr_bpm),
        "HR_min_bpm": safe_min(hr_bpm),
        "HR_max_bpm": safe_max(hr_bpm),
        "HR_std_bpm": safe_std(hr_bpm),
    }


# =========================================================
# 4. HRV: FREQUENCY-DOMAIN
# =========================================================

def rr_to_resampled_tachogram(rr_sec, fs_resample=4.0):
    """
    Перевод RR в равномерно ресемплированную тахограмму.
    """
    rr_sec = np.asarray(rr_sec, dtype=float)
    if len(rr_sec) < 4:
        return None, None

    t_beats = np.cumsum(rr_sec)
    t_beats = t_beats - t_beats[0]

    if t_beats[-1] <= 0:
        return None, None

    t_uniform = np.arange(0, t_beats[-1], 1.0 / fs_resample)
    rr_interp = np.interp(t_uniform, t_beats, rr_sec)

    return t_uniform, rr_interp


def bandpower_from_psd(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < fmax)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_frequency_domain_hrv(rr_sec, fs_resample=4.0):
    """
    Осторожный frequency-domain блок.
    Для коротких окон может быть шумным.
    """
    rr_sec = np.asarray(rr_sec, dtype=float)

    t_uniform, rr_interp = rr_to_resampled_tachogram(rr_sec, fs_resample=fs_resample)
    if rr_interp is None or len(rr_interp) < 16:
        return {
            "VLF_power": np.nan,
            "LF_power": np.nan,
            "HF_power": np.nan,
            "Total_power": np.nan,
            "LF_HF_ratio": np.nan,
            "LF_nu": np.nan,
            "HF_nu": np.nan,
            "LF_peak_Hz": np.nan,
            "HF_peak_Hz": np.nan,
        }

    rr_detrended = rr_interp - np.mean(rr_interp)

    freqs, psd = welch(
        rr_detrended,
        fs=fs_resample,
        nperseg=min(len(rr_detrended), 256),
        scaling="density",
    )

    vlf = bandpower_from_psd(freqs, psd, 0.0033, 0.04)
    lf = bandpower_from_psd(freqs, psd, 0.04, 0.15)
    hf = bandpower_from_psd(freqs, psd, 0.15, 0.40)
    total = bandpower_from_psd(freqs, psd, 0.0033, 0.40)

    lf_hf = lf / hf if np.isfinite(lf) and np.isfinite(hf) and hf > 1e-12 else np.nan
    lf_nu = 100.0 * lf / (lf + hf) if np.isfinite(lf) and np.isfinite(hf) and (lf + hf) > 1e-12 else np.nan
    hf_nu = 100.0 * hf / (lf + hf) if np.isfinite(lf) and np.isfinite(hf) and (lf + hf) > 1e-12 else np.nan

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf_peak = float(freqs[lf_mask][np.argmax(psd[lf_mask])]) if np.sum(lf_mask) > 0 else np.nan
    hf_peak = float(freqs[hf_mask][np.argmax(psd[hf_mask])]) if np.sum(hf_mask) > 0 else np.nan

    return {
        "VLF_power": vlf,
        "LF_power": lf,
        "HF_power": hf,
        "Total_power": total,
        "LF_HF_ratio": lf_hf,
        "LF_nu": lf_nu,
        "HF_nu": hf_nu,
        "LF_peak_Hz": lf_peak,
        "HF_peak_Hz": hf_peak,
    }


# =========================================================
# 5. HRV: NONLINEAR
# =========================================================

def compute_poincare_features(rr_sec):
    rr_sec = np.asarray(rr_sec, dtype=float)
    rr_sec = rr_sec[np.isfinite(rr_sec)]

    if len(rr_sec) < 3:
        return {
            "SD1_ms": np.nan,
            "SD2_ms": np.nan,
            "SD1_SD2_ratio": np.nan,
            "Ellipse_area": np.nan,
        }

    rr_ms = rr_sec * 1000.0
    diff_rr = np.diff(rr_ms)

    if len(diff_rr) < 2:
        sd1 = np.nan
    else:
        var_diff = np.var(diff_rr, ddof=1)
        sd1_arg = var_diff / 2.0
        sd1 = np.sqrt(sd1_arg) if sd1_arg >= 0 else np.nan

    var_rr = np.var(rr_ms, ddof=1) if len(rr_ms) > 2 else np.nan
    var_diff = np.var(diff_rr, ddof=1) if len(diff_rr) > 1 else np.nan

    sd2_arg = 2.0 * var_rr - 0.5 * var_diff

    # Из-за численной ошибки sd2_arg иногда может быть чуть ниже нуля,
    # например -1e-12. Это можно считать нулём.
    if np.isfinite(sd2_arg):
        if sd2_arg < 0 and abs(sd2_arg) < 1e-9:
            sd2_arg = 0.0

        sd2 = np.sqrt(sd2_arg) if sd2_arg >= 0 else np.nan
    else:
        sd2 = np.nan

    ratio = sd1 / sd2 if np.isfinite(sd1) and np.isfinite(sd2) and sd2 > 1e-12 else np.nan
    area = np.pi * sd1 * sd2 if np.isfinite(sd1) and np.isfinite(sd2) else np.nan

    return {
        "SD1_ms": float(sd1) if np.isfinite(sd1) else np.nan,
        "SD2_ms": float(sd2) if np.isfinite(sd2) else np.nan,
        "SD1_SD2_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "Ellipse_area": float(area) if np.isfinite(area) else np.nan,
    }


def approximate_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)

    def _phi(mm):
        patterns = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        C = []
        for p in patterns:
            d = np.max(np.abs(patterns - p), axis=1)
            C.append(np.mean(d <= r))
        C = np.asarray(C)
        C = np.where(C <= 0, 1e-12, C)
        return np.mean(np.log(C))

    return float(_phi(m) - _phi(m + 1))


def sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)

    def _count(mm):
        count = 0
        total = 0
        for i in range(n - mm):
            tmpl = x[i:i + mm]
            for j in range(i + 1, n - mm + 1):
                comp = x[j:j + mm]
                if np.max(np.abs(tmpl - comp)) <= r:
                    count += 1
                total += 1
        return count, total

    B, _ = _count(m)
    A, _ = _count(m + 1)

    if B == 0 or A == 0:
        return np.nan

    return float(-np.log(A / B))


def simple_shannon_entropy(x, bins=10):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N < 20:
        return np.nan

    L = []
    k_vals = np.arange(1, kmax)

    for k in k_vals:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue

            diffs = np.abs(np.diff(x[idx]))
            norm = (N - 1) / (len(idx) * k)

            Lmk = np.sum(diffs) * norm
            Lk.append(Lmk)

        if len(Lk) > 0:
            L.append(np.mean(Lk))
        else:
            L.append(np.nan)

    L = np.array(L)
    valid = (~np.isnan(L)) & (L > 0)

    if np.sum(valid) < 3:
        return np.nan

    log_k = np.log(1.0 / k_vals[valid])
    log_L = np.log(L[valid])

    slope, _ = np.polyfit(log_k, log_L, 1)
    return float(slope)

def compute_nonlinear_hrv(rr_sec):
    rr_sec = np.asarray(rr_sec, dtype=float)
    rr_ms = rr_sec * 1000.0

    poincare = compute_poincare_features(rr_sec)

    alpha1, alpha2 = compute_dfa(rr_sec)
    fd = higuchi_fd(rr_ms)

    return {
        **poincare,
        "ApproxEntropy": approximate_entropy(rr_ms),
        "SampleEntropy": sample_entropy(rr_ms),
        "ShannonEntropy": simple_shannon_entropy(rr_ms, bins=10),
        "DFA_alpha1": alpha1,
        "DFA_alpha2": alpha2,
        "FractalDimension": fd,
    }


# =========================================================
# 6. MORPHOLOGY: BEATS
# =========================================================
def _odd_int(n: int) -> int:
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def smooth_beat_savgol(x, sampling_rate, win_ms=15, polyorder=3):
    x = np.asarray(x, dtype=float)

    win = _odd_int(round(win_ms * sampling_rate / 1000.0))
    min_win = polyorder + 2
    if min_win % 2 == 0:
        min_win += 1
    win = max(win, min_win)

    if len(x) < win or win < 5:
        return x.copy()

    polyorder = min(polyorder, win - 2)
    return savgol_filter(x, window_length=win, polyorder=polyorder, mode="interp")


def detrend_beat_linear(beat, sampling_rate, edge_ms=40):
    """
    Линейная baseline-коррекция по краям окна.
    """
    x = np.asarray(beat, dtype=float)
    if len(x) < 5:
        return x.copy(), np.zeros_like(x)

    m = max(3, int(round(edge_ms * sampling_rate / 1000.0)))
    m = min(m, max(1, len(x) // 4))

    left_level = float(np.mean(x[:m]))
    right_level = float(np.mean(x[-m:]))

    baseline = np.linspace(left_level, right_level, len(x))
    y = x - baseline
    y = y - np.median(y)

    return y, baseline


def estimate_edge_noise(x, sampling_rate, edge_ms=40):
    """
    Грубая оценка шума по краям median beat.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return 0.0, 0.0

    m = max(3, int(round(edge_ms * sampling_rate / 1000.0)))
    m = min(m, max(1, len(x) // 4))

    if len(x) >= 2 * m:
        edge = np.concatenate([x[:m], x[-m:]])
    else:
        edge = x

    sigma = float(np.std(edge, ddof=1)) if len(edge) > 1 else 0.0

    dx = np.gradient(x) * sampling_rate
    if len(dx) >= 2 * m:
        edge_dx = np.concatenate([dx[:m], dx[-m:]])
    else:
        edge_dx = dx

    sigma_dx = float(np.std(edge_dx, ddof=1)) if len(edge_dx) > 1 else 0.0

    return sigma, sigma_dx


def find_main_qrs_peak_strict(beat, sampling_rate, pre_ms=200, search_ms=60):
    """
    Ищем главный экстремум QRS в окрестности ожидаемого R.
    """
    x = np.asarray(beat, dtype=float)

    expected_idx = int(round(pre_ms * sampling_rate / 1000.0))
    search_radius = int(round(search_ms * sampling_rate / 1000.0))

    left = max(0, expected_idx - search_radius)
    right = min(len(x), expected_idx + search_radius + 1)

    if right <= left:
        return None, None

    seg = x[left:right]
    local_idx = int(np.argmax(np.abs(seg)))
    qrs_idx = left + local_idx
    polarity = "positive" if x[qrs_idx] >= 0 else "negative"

    return qrs_idx, polarity


def normalize_qrs_polarity(beat, qrs_idx):
    """
    Если главный пик направлен вниз, инвертируем beat.
    После этого QRS анализируем как будто R всегда положительный.
    """
    x = np.asarray(beat, dtype=float)
    if qrs_idx is None:
        return x.copy(), False

    if x[qrs_idx] < 0:
        return -x, True
    return x.copy(), False


def compute_area_baseline_corrected(signal, left_idx, right_idx, sampling_rate):
    """
    Площадь между onset и offset с вычитанием линейной baseline между концами сегмента.
    """
    if left_idx is None or right_idx is None or right_idx <= left_idx:
        return np.nan

    seg = np.asarray(signal[left_idx:right_idx + 1], dtype=float)
    if len(seg) < 2:
        return np.nan

    baseline = np.linspace(seg[0], seg[-1], len(seg))
    seg_corr = seg - baseline

    return float(np.trapezoid(seg_corr, dx=1.0 / sampling_rate))


def compute_width_at_fraction(signal, peak_idx, sampling_rate, fraction=0.5):
    """
    Универсальная ширина волны на доле от |амплитуды|.
    """
    x = np.asarray(signal, dtype=float)

    if peak_idx is None or peak_idx < 0 or peak_idx >= len(x):
        return np.nan

    amp = float(x[peak_idx])
    amp_abs = abs(amp)
    if amp_abs < 1e-12:
        return np.nan

    thr = fraction * amp_abs

    left = peak_idx
    while left > 0 and abs(x[left]) > thr:
        left -= 1

    right = peak_idx
    while right < len(x) - 1 and abs(x[right]) > thr:
        right += 1

    return float((right - left) / sampling_rate * 1000.0)


def find_qrs_onset_offset(
    beat,
    qrs_idx,
    sampling_rate,
    noise_std,
    noise_dx_std,
    search_left_ms=120,
    search_right_ms=160,
):
    """
    Поиск onset/offset QRS по сочетанию амплитуды и производной.
    """
    x = np.asarray(beat, dtype=float)
    dx = np.gradient(x) * sampling_rate

    left_lim = max(0, qrs_idx - int(round(search_left_ms * sampling_rate / 1000.0)))
    right_lim = min(len(x) - 1, qrs_idx + int(round(search_right_ms * sampling_rate / 1000.0)))

    local_left = max(0, qrs_idx - int(round(40 * sampling_rate / 1000.0)))
    local_right = min(len(x), qrs_idx + int(round(40 * sampling_rate / 1000.0)) + 1)

    local_dx_peak = float(np.max(np.abs(dx[local_left:local_right]))) if local_right > local_left else 0.0

    amp_thr = max(0.05 * max(x[qrs_idx], 0.0), 2.5 * noise_std)
    dx_thr = max(0.08 * local_dx_peak, 2.5 * noise_dx_std)

    onset = left_lim
    consec = 2

    for i in range(qrs_idx - 1, left_lim + consec - 2, -1):
        sl = slice(i - consec + 1, i + 1)
        if np.all(np.abs(x[sl]) <= amp_thr) and np.all(np.abs(dx[sl]) <= dx_thr):
            onset = i
            break

    offset = right_lim
    for i in range(qrs_idx + 1, right_lim - consec + 2):
        sl = slice(i, i + consec)
        if np.all(np.abs(x[sl]) <= amp_thr) and np.all(np.abs(dx[sl]) <= dx_thr):
            offset = i
            break

    if onset >= qrs_idx:
        onset = max(0, qrs_idx - int(round(40 * sampling_rate / 1000.0)))

    if offset <= qrs_idx:
        offset = min(len(x) - 1, qrs_idx + int(round(60 * sampling_rate / 1000.0)))

    return onset, offset, amp_thr, dx_thr


def detect_wave_candidate(
    beat,
    start_idx,
    end_idx,
    sampling_rate,
    noise_std,
    r_amp,
    min_rel_amp=0.03,
    min_width_ms=20,
    max_width_ms=180,
):
    """
    Универсальный детектор P/T-кандидата в заданном окне.
    Ищет наибольший по модулю экстремум и проверяет его:
    - по амплитуде,
    - по ширине,
    - по SNR.
    """
    x = np.asarray(beat, dtype=float)

    if start_idx is None or end_idx is None:
        return None
    if end_idx - start_idx < 3:
        return None

    seg = x[start_idx:end_idx]
    if len(seg) < 3:
        return None

    peak_rel = int(np.argmax(np.abs(seg)))
    peak_idx = start_idx + peak_rel
    amp = float(x[peak_idx])
    amp_abs = abs(amp)

    amp_thr = max(min_rel_amp * max(r_amp, 1e-9), 2.5 * max(noise_std, 1e-9))
    if amp_abs < amp_thr:
        return None

    bound_thr = max(0.20 * amp_abs, 1.5 * max(noise_std, 1e-9))

    left = peak_idx
    while left > start_idx and abs(x[left]) > bound_thr:
        left -= 1

    right = peak_idx
    while right < end_idx - 1 and abs(x[right]) > bound_thr:
        right += 1

    width_ms = (right - left) / sampling_rate * 1000.0
    if width_ms < min_width_ms or width_ms > max_width_ms:
        return None

    snr = amp_abs / max(noise_std, 1e-9)
    polarity = "positive" if amp >= 0 else "negative"

    return {
        "peak_idx": int(peak_idx),
        "amp": float(amp),
        "amp_abs": float(amp_abs),
        "left_idx": int(left),
        "right_idx": int(right),
        "width_ms": float(width_ms),
        "snr": float(snr),
        "polarity": polarity,
    }


def extract_morphology_features_full_v2(
    median_beat,
    sampling_rate,
    pre_ms=200,
    search_ms=60,
    smooth_ms=15,
):
    """
    Более строгая версия морфологического блока:
    - baseline correction
    - сглаживание
    - нормализация полярности QRS
    - onset/offset для QRS
    - quality-gate для P/T
    """

    empty = {
        "qrs": {},
        "p": {},
        "t": {},
    }

    if median_beat is None:
        return empty

    beat_raw = np.asarray(median_beat, dtype=float)
    if len(beat_raw) < 10:
        return empty

    # -------------------------------------------------
    # 1. Предобработка median beat
    # -------------------------------------------------
    beat_detr, baseline = detrend_beat_linear(
        beat_raw,
        sampling_rate=sampling_rate,
        edge_ms=40,
    )

    beat_smooth = smooth_beat_savgol(
        beat_detr,
        sampling_rate=sampling_rate,
        win_ms=smooth_ms,
        polyorder=3,
    )

    noise_std, noise_dx_std = estimate_edge_noise(
        beat_smooth,
        sampling_rate=sampling_rate,
        edge_ms=40,
    )

    # -------------------------------------------------
    # 2. Главный QRS-пик и нормализация полярности
    # -------------------------------------------------
    qrs_idx_raw, raw_polarity = find_main_qrs_peak_strict(
        beat_smooth,
        sampling_rate=sampling_rate,
        pre_ms=pre_ms,
        search_ms=search_ms,
    )

    if qrs_idx_raw is None:
        return empty

    beat_norm, inverted = normalize_qrs_polarity(beat_smooth, qrs_idx_raw)
    qrs_idx = int(qrs_idx_raw)
    r_amp = float(beat_norm[qrs_idx])

    # -------------------------------------------------
    # 3. QRS onset / offset
    # -------------------------------------------------
    qrs_onset, qrs_offset, qrs_amp_thr, qrs_dx_thr = find_qrs_onset_offset(
        beat=beat_norm,
        qrs_idx=qrs_idx,
        sampling_rate=sampling_rate,
        noise_std=noise_std,
        noise_dx_std=noise_dx_std,
        search_left_ms=120,
        search_right_ms=160,
    )

    qrs_duration_ms = float((qrs_offset - qrs_onset) / sampling_rate * 1000.0)

    # Q и S относительно нормализованного beat (где R вверх)
    if qrs_idx > qrs_onset:
        q_idx = qrs_onset + int(np.argmin(beat_norm[qrs_onset:qrs_idx + 1]))
        q_amp = float(beat_norm[q_idx])
    else:
        q_idx = qrs_idx
        q_amp = np.nan

    if qrs_offset > qrs_idx:
        s_idx = qrs_idx + int(np.argmin(beat_norm[qrs_idx:qrs_offset + 1]))
        s_amp = float(beat_norm[s_idx])
    else:
        s_idx = qrs_idx
        s_amp = np.nan

    # Производная для slope
    dx = np.gradient(beat_norm) * sampling_rate
    up_slope = float(np.max(dx[qrs_onset:qrs_idx + 1])) if qrs_idx > qrs_onset else np.nan
    down_slope = float(np.min(dx[qrs_idx:qrs_offset + 1])) if qrs_offset > qrs_idx else np.nan

    qrs_area = compute_area_baseline_corrected(
        beat_norm,
        qrs_onset,
        qrs_offset,
        sampling_rate,
    )

    seg_qrs = beat_norm[qrs_onset:qrs_offset + 1]
    pos_area = float(np.trapezoid(np.clip(seg_qrs, 0, None), dx=1.0 / sampling_rate)) if len(seg_qrs) > 1 else np.nan
    neg_area = float(np.trapezoid(np.clip(seg_qrs, None, 0), dx=1.0 / sampling_rate)) if len(seg_qrs) > 1 else np.nan

    qrs_width_50_ms = compute_width_at_fraction(
        beat_norm,
        qrs_idx,
        sampling_rate,
        fraction=0.5,
    )
    qrs_width_20_ms = compute_width_at_fraction(
        beat_norm,
        qrs_idx,
        sampling_rate,
        fraction=0.2,
    )

    morph_qrs = {
        "QRS_main_idx": int(qrs_idx),
        "R_idx": int(qrs_idx),  # можно как alias, если хочешь
        "Q_idx": int(q_idx) if q_idx is not None else np.nan,
        "S_idx": int(s_idx) if s_idx is not None else np.nan,

        "QRS_polarity_raw": raw_polarity,
        "QRS_inverted_for_analysis": bool(inverted),

        "QRS_main_amp": float(r_amp),
        "Q_amp": q_amp,
        "S_amp": s_amp,
        "RS_amp": float(r_amp - s_amp) if np.isfinite(s_amp) else np.nan,

        "QRS_onset_idx": int(qrs_onset),
        "QRS_offset_idx": int(qrs_offset),
        "QRS_duration_ms": qrs_duration_ms,

        "RQ_interval_ms": float((qrs_idx - q_idx) / sampling_rate * 1000.0) if q_idx is not None else np.nan,
        "RS_interval_ms": float((s_idx - qrs_idx) / sampling_rate * 1000.0) if s_idx is not None else np.nan,

        "R_width_half_ms": qrs_width_50_ms,
        "QRS_width_ms": qrs_duration_ms,
        "QRS_width_20_ms": qrs_width_20_ms,
        "QRS_left_width_ms": float((qrs_idx - qrs_onset) / sampling_rate * 1000.0),
        "QRS_right_width_ms": float((qrs_offset - qrs_idx) / sampling_rate * 1000.0),

        "QRS_area": qrs_area,
        "Positive_area": pos_area,
        "Negative_area": neg_area,

        "R_up_slope": up_slope,
        "R_down_slope": down_slope,

        "Noise_std": float(noise_std),
        "Noise_dx_std": float(noise_dx_std),
        "QRS_amp_threshold": float(qrs_amp_thr),
        "QRS_dx_threshold": float(qrs_dx_thr),
    }

    # -------------------------------------------------
    # 4. P-wave
    # -------------------------------------------------
    p_start = max(0, qrs_onset - int(round(250 * sampling_rate / 1000.0)))
    p_end = max(p_start + 3, qrs_onset - int(round(40 * sampling_rate / 1000.0)))

    p_candidate = detect_wave_candidate(
        beat=beat_norm,
        start_idx=p_start,
        end_idx=p_end,
        sampling_rate=sampling_rate,
        noise_std=noise_std,
        r_amp=r_amp,
        min_rel_amp=0.03,
        min_width_ms=20,
        max_width_ms=180,
    )

    if p_candidate is None:
        morph_p = {
            "P_present": False,
            "P_polarity": None,
            "P_amp": np.nan,
            "P_width_ms": np.nan,
            "PR_interval_ms": np.nan,              # onset-to-onset
            "PR_peak_interval_ms": np.nan,         # peak-to-peak
            "P_area": np.nan,
            "P_onset_idx": np.nan,
            "P_offset_idx": np.nan,
            "P_peak_idx": np.nan,
            "P_snr": np.nan,
        }
    else:
        p_idx = p_candidate["peak_idx"]
        p_on = p_candidate["left_idx"]
        p_off = p_candidate["right_idx"]
        p_amp = p_candidate["amp"]
        p_area = compute_area_baseline_corrected(
            beat_norm,
            p_on,
            p_off,
            sampling_rate,
        )

        morph_p = {
            "P_present": True,
            "P_polarity": p_candidate["polarity"],
            "P_amp": float(p_amp),
            "P_width_ms": float(p_candidate["width_ms"]),
            "PR_interval_ms": float((qrs_onset - p_on) / sampling_rate * 1000.0),
            "PR_peak_interval_ms": float((qrs_idx - p_idx) / sampling_rate * 1000.0),
            "P_area": p_area,
            "P_onset_idx": int(p_on),
            "P_offset_idx": int(p_off),
            "P_peak_idx": int(p_idx),
            "P_snr": float(p_candidate["snr"]),
        }

    # -------------------------------------------------
    # 5. T-wave
    # -------------------------------------------------
    t_start = min(len(beat_norm) - 2, qrs_offset + int(round(50 * sampling_rate / 1000.0)))
    t_end = min(len(beat_norm), qrs_offset + int(round(380 * sampling_rate / 1000.0)))

    t_candidate = detect_wave_candidate(
        beat=beat_norm,
        start_idx=t_start,
        end_idx=t_end,
        sampling_rate=sampling_rate,
        noise_std=noise_std,
        r_amp=r_amp,
        min_rel_amp=0.05,
        min_width_ms=40,
        max_width_ms=320,
    )

    if t_candidate is None:
        morph_t = {
            "T_present": False,
            "T_polarity": None,
            "T_amp": np.nan,
            "RT_interval_ms": np.nan,              # peak-to-peak, для совместимости
            "T_peak_interval_ms": np.nan,
            "T_width_ms": np.nan,
            "T_area": np.nan,
            "T_onset_idx": np.nan,
            "T_offset_idx": np.nan,
            "T_peak_idx": np.nan,
            "T_snr": np.nan,
            "QT_like_ms": np.nan,                  # onset QRS -> offset T
        }
    else:
        t_idx = t_candidate["peak_idx"]
        t_on = t_candidate["left_idx"]
        t_off = t_candidate["right_idx"]
        t_amp = t_candidate["amp"]
        t_area = compute_area_baseline_corrected(
            beat_norm,
            t_on,
            t_off,
            sampling_rate,
        )

        morph_t = {
            "T_present": True,
            "T_polarity": t_candidate["polarity"],
            "T_amp": float(t_amp),
            "RT_interval_ms": float((t_idx - qrs_idx) / sampling_rate * 1000.0),
            "T_peak_interval_ms": float((t_idx - qrs_idx) / sampling_rate * 1000.0),
            "T_width_ms": float(t_candidate["width_ms"]),
            "T_area": t_area,
            "T_onset_idx": int(t_on),
            "T_offset_idx": int(t_off),
            "T_peak_idx": int(t_idx),
            "T_snr": float(t_candidate["snr"]),
            "QT_like_ms": float((t_off - qrs_onset) / sampling_rate * 1000.0),
        }

    return {
        "qrs": morph_qrs,
        "p": morph_p,
        "t": morph_t,
    }

def assess_ecg_quality(
    ecg_signal,
    sampling_rate=FS_BASE,
    adc_min=ADC_MIN,
    adc_max=ADC_MAX,
    detector_method="neurokit",
    lowcut=0.5,
    highcut=30,
    filter_method="butterworth",
    filter_order=2,
    trim_seconds=0.5,
):
    ecg_raw_trimmed, ecg_filt_trimmed, trim_n = prepare_ecg_for_hrv(
        ecg_signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        filter_method=filter_method,
        filter_order=filter_order,
        trim_seconds=trim_seconds,
    )

    clip_low = np.mean(ecg_raw_trimmed <= adc_min + 2) if len(ecg_raw_trimmed) else np.nan
    clip_high = np.mean(ecg_raw_trimmed >= adc_max - 2) if len(ecg_raw_trimmed) else np.nan
    clipping_ratio = float(clip_low + clip_high) if len(ecg_raw_trimmed) else np.nan

    rpeaks = detect_rpeaks_neurokit(
        ecg_filtered=ecg_filt_trimmed,
        sampling_rate=sampling_rate,
        method=detector_method,
    )

    rr_sec = np.diff(rpeaks) / sampling_rate if len(rpeaks) >= 2 else np.array([])
    rr_phys_bad_ratio = float(np.mean((rr_sec < 0.3) | (rr_sec > 2.0))) if len(rr_sec) > 0 else np.nan

    hrv = compute_time_domain_hrv(rr_sec)

    duration_sec = len(ecg_raw_trimmed) / sampling_rate
    n_samples = len(ecg_raw_trimmed)

    return {
        "clipping_ratio": clipping_ratio,
        "rr_phys_bad_ratio": rr_phys_bad_ratio,
        "rpeaks": rpeaks,
        "rr_sec": rr_sec,
        "ecg_raw_trimmed": ecg_raw_trimmed,
        "ecg_filt_trimmed": ecg_filt_trimmed,
        "trim_n": int(trim_n),
        **hrv,
        "n_samples": int(n_samples),
        "duration_sec": float(duration_sec),
    }

def extract_beats_around_rpeaks(
        ecg_signal,
        rpeaks,
        sampling_rate,
        pre_ms=200,
        post_ms=400,
):
    x = np.asarray(ecg_signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)

    pre = int(pre_ms * sampling_rate / 1000)
    post = int(post_ms * sampling_rate / 1000)

    beats = []
    valid_rpeaks = []

    for rp in rpeaks:
        left = rp - pre
        right = rp + post
        if left >= 0 and right < len(x):
            beats.append(x[left:right + 1].copy())
            valid_rpeaks.append(rp)

    if len(beats) == 0:
        return np.empty((0, pre + post + 1)), np.array([], dtype=int)

    return np.asarray(beats), np.asarray(valid_rpeaks, dtype=int)

def filter_beats_by_template_correlation(beats, corr_threshold=0.8):
    if len(beats) == 0:
        return beats, np.array([]), None, np.array([], dtype=bool)

    beats_norm = []
    for b in beats:
        bb = b - np.mean(b)
        s = np.std(bb)
        beats_norm.append(bb / s if s > 1e-9 else np.zeros_like(bb))
    beats_norm = np.asarray(beats_norm)

    template = np.median(beats_norm, axis=0)

    corrs = []
    for b in beats_norm:
        c = np.corrcoef(b, template)[0, 1]
        corrs.append(c)
    corrs = np.asarray(corrs)

    keep_mask = corrs >= corr_threshold
    beats_good = beats[keep_mask]

    return beats_good, corrs, template, keep_mask

def compute_median_beat(beats):
    if len(beats) == 0:
        return None
    return np.median(beats, axis=0)

def extract_all_features_from_segment(
    ecg_signal,
    sampling_rate,
    rpeaks=None,
    adc_min=ADC_MIN,
    adc_max=ADC_MAX,
    detector_method="neurokit",
    hrv_lowcut=0.5,
    hrv_highcut=30,
    morph_lowcut=0.3,
    morph_highcut=35,
    trim_seconds=0.5,
    morph_pre_ms=200,
    morph_post_ms=400,
    corr_threshold=0.8,
    smooth_ms=15,
):
    """
    Единый экстрактор признаков.

    Если rpeaks is None:
        - выполняет фильтрацию для HRV
        - автоматически детектирует R-пики
        - считает признаки по детектированным пикам

    Если rpeaks переданы:
        - использует их как уже подтверждённые
        - считает признаки напрямую по ним

    ВАЖНО:
        Переданные rpeaks должны быть индексами относительно ecg_signal.
    """
    ecg_signal = np.asarray(ecg_signal, dtype=float)

    if rpeaks is None:
        result = assess_ecg_quality(
            ecg_signal=ecg_signal,
            sampling_rate=sampling_rate,
            adc_min=adc_min,
            adc_max=adc_max,
            detector_method=detector_method,
            lowcut=hrv_lowcut,
            highcut=hrv_highcut,
            trim_seconds=trim_seconds,
        )

        signal_used = np.asarray(result["ecg_raw_trimmed"], dtype=float)
        signal_filtered_hrv = np.asarray(result["ecg_filt_trimmed"], dtype=float)
        used_rpeaks = np.asarray(result["rpeaks"], dtype=int)
        rr_sec = np.asarray(result["rr_sec"], dtype=float)

        base_qc = {
            "clipping_ratio": result["clipping_ratio"],
            "rr_phys_bad_ratio": result["rr_phys_bad_ratio"],
            "n_rpeaks": int(len(used_rpeaks)),
        }
        rpeaks_source = "detected"

    else:
        signal_used = np.asarray(ecg_signal, dtype=float)
        signal_filtered_hrv = None
        used_rpeaks = np.asarray(rpeaks, dtype=int)
        rr_sec = compute_rr_from_rpeaks(used_rpeaks, sampling_rate)

        basic_qc = compute_basic_qc(
            ecg_signal=signal_used,
            rpeaks=used_rpeaks,
            sampling_rate=sampling_rate,
            adc_min=adc_min,
            adc_max=adc_max,
        )

        base_qc = {
            "clipping_ratio": basic_qc["clipping_ratio"],
            "rr_phys_bad_ratio": basic_qc["rr_phys_bad_ratio"],
            "n_rpeaks": int(len(used_rpeaks)),
        }
        rpeaks_source = "provided"
        result = None

    # ---------- HRV ----------
    hrv_time = compute_time_domain_hrv(rr_sec)
    hrv_freq = compute_frequency_domain_hrv(rr_sec)
    hrv_nonlinear = compute_nonlinear_hrv(rr_sec)

    # ---------- Morphology ----------
    ecg_morph, _ = prepare_ecg_for_morphology(
        ecg_signal=signal_used,
        sampling_rate=sampling_rate,
        lowcut=morph_lowcut,
        highcut=morph_highcut,
        trim_seconds=0.0,
    )

    beats, valid_rpeaks = extract_beats_around_rpeaks(
        ecg_signal=ecg_morph,
        rpeaks=used_rpeaks,
        sampling_rate=sampling_rate,
        pre_ms=morph_pre_ms,
        post_ms=morph_post_ms,
    )

    beats_good, corrs, template, keep_mask = filter_beats_by_template_correlation(
        beats,
        corr_threshold=corr_threshold,
    )

    median_beat = compute_median_beat(beats_good)

    morph_feats = extract_morphology_features_full_v2(
        median_beat=median_beat,
        sampling_rate=sampling_rate,
        pre_ms=morph_pre_ms,
        search_ms=60,
        smooth_ms=smooth_ms,
    )

    # ---------- QC ----------
    qc = {
        **base_qc,
        "n_beats_extracted": int(len(beats)),
        "n_beats_good": int(len(beats_good)),
        "corr_min": float(np.nanmin(corrs)) if len(corrs) > 0 else np.nan,
        "corr_median": float(np.nanmedian(corrs)) if len(corrs) > 0 else np.nan,
        "corr_max": float(np.nanmax(corrs)) if len(corrs) > 0 else np.nan,
    }

    # ---------- META ----------
    meta = {
        "sampling_rate": float(sampling_rate),
        "segment_duration_sec": float(len(ecg_signal) / sampling_rate),
        "analyzed_duration_sec": float(len(signal_used) / sampling_rate),
        "rpeaks_source": rpeaks_source,
    }

    features = {
        "meta": meta,
        "qc": qc,
        "hrv_time": hrv_time,
        "hrv_freq": hrv_freq,
        "hrv_nonlinear": hrv_nonlinear,
        "morph_qrs": morph_feats["qrs"],
        "morph_p": morph_feats["p"],
        "morph_t": morph_feats["t"],
    }

    debug = {
        "assess_result": result,
        "ecg_signal_used": signal_used,
        "ecg_filtered_hrv": signal_filtered_hrv,
        "ecg_morph": ecg_morph,
        "used_rpeaks": used_rpeaks,
        "valid_rpeaks_for_beats": valid_rpeaks,
        "beats": beats,
        "beats_good": beats_good,
        "corrs": corrs,
        "template": template,
        "keep_mask": keep_mask,
        "median_beat": median_beat,
    }

    return features, debug