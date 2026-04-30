from pathlib import Path
import numpy as np
import pandas as pd

from protocol import (
    load_protocol_config,
    validate_protocol_config,
    normalize_protocol_config,
    build_record_runtime_config,
    resolve_existing_file,
    estimate_sampling_rate_from_runtime,
    resolve_segments,
)

from features import (
    prepare_ecg_for_hrv,
    detect_rpeaks_neurokit,
    compute_basic_qc,
    drop_edge_rpeaks,
    extract_all_features_from_segment,
)

from review import (
    load_ecg_csv,
    load_review_result,
    save_review_result,
    review_ecg_record,
    review_rpeaks_record,
    review_morphology_record,
    auto_review_gate,
    build_morphology_review,
    morphology_qc_gate,
    select_review_rpeaks,
)


# =========================================================
# SMALL HELPERS
# =========================================================
def safe_segment_name(label, window_idx=None):
    if window_idx is not None:
        return f"window_{int(window_idx):02d}"

    label = str(label)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    return safe or "segment"

def flatten_features(feature_dict, prefix_sep="__"):
    flat = {}

    for group, values in feature_dict.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{group}{prefix_sep}{k}"] = v
        else:
            flat[group] = values

    return flat


def build_review_path(review_dir: Path, runtime_cfg, suffix=None):
    """
    Путь к review-файлу.

    Whole-record:
      _reviews/date/phase/recording.review.json

    Windowed:
      _reviews/date/phase/recording/window_01.review.json
    """
    date = runtime_cfg["date"]
    phase = runtime_cfg["phase_id"]
    rec_id = runtime_cfg["recording_id"]

    if suffix is None:
        return review_dir / date / phase / f"{rec_id}.review.json"

    return review_dir / date / phase / rec_id / suffix


def get_signal_qc_profile(runtime_cfg):
    """
    Новый формат: runtime_cfg["signal_qc"].
    fallback на "qc" оставлен только чтобы не падать, если protocol.py ещё не до конца обновлён.
    """
    return runtime_cfg.get("signal_qc") or runtime_cfg.get("qc") or {}


def get_morphology_qc_profile(runtime_cfg):
    return runtime_cfg.get("morphology_qc")


def morphology_should_be_evaluated(runtime_cfg):
    """
    Морфологический review/gate запускаем только если:
    - в feature_groups есть morphology или feature_groups не задан;
    - задан morphology_qc_profile.
    """
    feature_groups = runtime_cfg.get("feature_groups")
    morphology_qc = get_morphology_qc_profile(runtime_cfg)

    if feature_groups is not None and "morphology" not in set(feature_groups):
        return False

    if morphology_qc is None:
        return False

    return True


def empty_morphology_review(status="not_run"):
    return {
        "status": status,
        "qc": None,
        "qc_gate": None,
        "features_preview": None,
        "corr_threshold": None,
        "error": None,
        "log": [],
    }


def resolve_effective_review_mode(global_interactive_mode, runtime_cfg):
    """
    Возвращает режим review:
      True       -> ручной review
      False      -> полностью автоматический
      "bad_only" -> ручной только если gate не прошёл

    Если global_interactive_mode is None, берём review_mode из protocol.yaml.
    """
    if global_interactive_mode is not None:
        return global_interactive_mode

    review_mode = runtime_cfg.get("review_mode", "bad_only")

    mapping = {
        "manual": True,
        "manual_all": True,
        "manual_each_window": True,
        "bad_only": "bad_only",
        "auto": False,
        "none": False,
    }

    return mapping.get(review_mode, True)


def build_extraction_kwargs_from_runtime(runtime_cfg):
    proc = runtime_cfg.get("processing", {}) or {}
    adc = runtime_cfg.get("adc", {}) or {}

    hrv_filter = proc.get("hrv_filter", {}) or {}
    morph_filter = proc.get("morphology_filter", {}) or {}
    morph = proc.get("morphology", {}) or {}

    return {
        "adc_min": adc.get("min", 0),
        "adc_max": adc.get("max", 675),
        "detector_method": proc.get("detector_method", "neurokit"),

        "hrv_lowcut": hrv_filter.get("lowcut", 0.5),
        "hrv_highcut": hrv_filter.get("highcut", 30),
        "trim_seconds": hrv_filter.get("trim_seconds", 0.5),

        "morph_lowcut": morph_filter.get("lowcut", 0.3),
        "morph_highcut": morph_filter.get("highcut", 35),

        "morph_pre_ms": morph.get("pre_ms", 200),
        "morph_post_ms": morph.get("post_ms", 400),
        "corr_threshold": morph.get("corr_threshold", 0.8),
        "smooth_ms": morph.get("smooth_ms", 15),
    }


def filter_feature_groups(features, feature_groups):
    """
    Оставляет только нужные группы признаков.
    Всегда сохраняет meta и qc.
    """
    if not feature_groups:
        return features

    out = {
        "meta": dict(features.get("meta", {})),
        "qc": dict(features.get("qc", {})),
    }

    fg = set(feature_groups)

    if "hrv_time" in fg and "hrv_time" in features:
        out["hrv_time"] = features["hrv_time"]

    if "hrv_extended" in fg:
        if "hrv_freq" in features:
            out["hrv_freq"] = features["hrv_freq"]
        if "hrv_nonlinear" in features:
            out["hrv_nonlinear"] = features["hrv_nonlinear"]

    if "morphology" in fg:
        if "morph_qrs" in features:
            out["morph_qrs"] = features["morph_qrs"]
        if "morph_p" in features:
            out["morph_p"] = features["morph_p"]
        if "morph_t" in features:
            out["morph_t"] = features["morph_t"]

    return out


def invalidate_morphology_features(features):
    """
    Если морфология не прошла review/QC, зануляем морфологические признаки,
    чтобы они случайно не попали в модель.
    """
    out = {
        k: (v.copy() if isinstance(v, dict) else v)
        for k, v in features.items()
    }

    for group_name in ("morph_qrs", "morph_p", "morph_t"):
        if group_name in out and isinstance(out[group_name], dict):
            out[group_name] = {k: np.nan for k in out[group_name].keys()}

    if "meta" in out and isinstance(out["meta"], dict):
        out["meta"]["morphology_validated"] = False

    return out

def make_hrv_review(status, qc_gate=None, source=None, log=None):
    return {
        "status": status,
        "qc_gate": qc_gate,
        "source": source,
        "log": log or [],
    }


def hrv_is_validated(review_result):
    """
    HRV валидна только если hrv_review.status == accepted.

    Если qc_gate есть, он тоже должен быть passed=True.
    Но для ручного accepted можно записывать qc_gate passed=True.
    """
    if not isinstance(review_result, dict):
        return False

    hrv_review = review_result.get("hrv_review")
    if not isinstance(hrv_review, dict):
        return False

    if hrv_review.get("status") != "accepted":
        return False

    qc_gate = hrv_review.get("qc_gate")
    if isinstance(qc_gate, dict):
        return bool(qc_gate.get("passed", False))

    return True


def invalidate_hrv_features(features):
    """
    Если HRV не прошла QC/manual review, зануляем HRV-признаки,
    но оставляем morphology, если она валидна.
    """
    out = {
        k: (v.copy() if isinstance(v, dict) else v)
        for k, v in features.items()
    }

    for group_name in ("hrv_time", "hrv_freq", "hrv_nonlinear"):
        if group_name in out and isinstance(out[group_name], dict):
            out[group_name] = {k: np.nan for k in out[group_name].keys()}

    if "meta" in out and isinstance(out["meta"], dict):
        out["meta"]["hrv_validated"] = False

    return out

def morphology_is_validated(morphology_review):
    """
    Финальное правило: морфология валидна только если:
    - пользователь/авто-gate приняли её;
    - morphology_qc_gate прошёл.
    """
    if not isinstance(morphology_review, dict):
        return False

    if morphology_review.get("status") != "accepted":
        return False

    qc_gate = morphology_review.get("qc_gate")
    if not isinstance(qc_gate, dict):
        return False

    return bool(qc_gate.get("passed", False))


def gate_reasons_to_string(gate):
    if not isinstance(gate, dict):
        return None
    reasons = gate.get("reasons", [])
    if not reasons:
        return ""
    return "; ".join(str(x) for x in reasons)


# =========================================================
# AUTO REVIEW BUILDERS
# =========================================================

def build_auto_signal_review_result(
    signal,
    fs,
    runtime_cfg,
    extraction_kwargs,
    log_action="auto_accept",
    extra_log=None,
):
    """
    Автоматически принимает signal_review.
    Используется в режимах auto и bad_only, если signal gate прошёл.
    """
    signal_qc = get_signal_qc_profile(runtime_cfg)
    adc_min = extraction_kwargs["adc_min"]
    adc_max = extraction_kwargs["adc_max"]

    ecg_raw, ecg_filt, trim_n = prepare_ecg_for_hrv(
        ecg_signal=signal,
        sampling_rate=fs,
        lowcut=extraction_kwargs["hrv_lowcut"],
        highcut=extraction_kwargs["hrv_highcut"],
        trim_seconds=0.0,
    )

    rpeaks = detect_rpeaks_neurokit(
        ecg_filtered=ecg_filt,
        sampling_rate=fs,
        method=extraction_kwargs["detector_method"],
    )

    rpeaks_no_edge = drop_edge_rpeaks(
        rpeaks,
        len(ecg_raw),
        sampling_rate=fs,
        edge_guard_sec=signal_qc.get("edge_guard_sec", 0.5),
    )

    qc = compute_basic_qc(
        ecg_signal=ecg_raw,
        rpeaks=rpeaks,
        sampling_rate=fs,
        adc_min=adc_min,
        adc_max=adc_max,
    )

    log_item = {"action": log_action}
    if extra_log is not None:
        log_item["extra"] = extra_log

    return {
        "status": "accepted",
        "signal": np.asarray(ecg_raw, dtype=float),
        "signal_filtered": np.asarray(ecg_filt, dtype=float),
        "rpeaks": np.asarray(rpeaks, dtype=int),
        "rpeaks_no_edge": np.asarray(rpeaks_no_edge, dtype=int),
        "sampling_rate": fs,
        "log": [log_item],
        "qc": qc,
    }


def build_auto_morphology_review_result(
    signal_review,
    fs,
    runtime_cfg,
    extraction_kwargs,
    log_action_prefix="auto_morphology",
):
    """
    Автоматически строит morphology debug, прогоняет morphology_qc_gate
    и возвращает morphology_review.

    Если gate passed -> status="accepted".
    Если gate failed -> status="skipped".
    """
    if not morphology_should_be_evaluated(runtime_cfg):
        return empty_morphology_review(status="not_run")

    morphology_qc = get_morphology_qc_profile(runtime_cfg)

    rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=True)
    if len(rpeaks_for_morph) < 2:
        rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=False)

    morph_debug = build_morphology_review(
        ecg_signal=signal_review["signal"],
        rpeaks=rpeaks_for_morph,
        sampling_rate=fs,
        morph_lowcut=extraction_kwargs["morph_lowcut"],
        morph_highcut=extraction_kwargs["morph_highcut"],
        morph_pre_ms=extraction_kwargs["morph_pre_ms"],
        morph_post_ms=extraction_kwargs["morph_post_ms"],
        corr_threshold=extraction_kwargs["corr_threshold"],
        smooth_ms=extraction_kwargs["smooth_ms"],
    )

    qc_gate = morphology_qc_gate(
        morph_debug=morph_debug,
        morphology_qc_profile=morphology_qc,
    )

    passed = bool(qc_gate.get("passed", False))
    status = "accepted" if passed else "skipped"

    return {
        "status": status,
        "qc": morph_debug.get("qc"),
        "qc_gate": qc_gate,
        "features_preview": morph_debug.get("morph_feats"),
        "corr_threshold": extraction_kwargs["corr_threshold"],
        "error": morph_debug.get("error"),
        "log": [
            {
                "action": f"{log_action_prefix}_{'accept' if passed else 'reject'}",
                "qc_gate": qc_gate,
            }
        ],
    }


def build_review_result_from_parts(
    signal_review,
    hrv_review,
    morphology_review,
    runtime_cfg,
    fs,
):
    meta = {
        "date": runtime_cfg["date"],
        "phase": runtime_cfg["phase_id"],
        "record_type": runtime_cfg["recording_id"],
        "sampling_rate": fs,
    }

    if "segment_label" in runtime_cfg:
        meta["segment_label"] = runtime_cfg.get("segment_label")
        meta["segment_start_sec"] = runtime_cfg.get("segment_start_sec")
        meta["segment_end_sec"] = runtime_cfg.get("segment_end_sec")
        meta["segment_window_idx"] = runtime_cfg.get("segment_window_idx")

    return {
        "signal_review": signal_review,
        "hrv_review": hrv_review,
        "morphology_review": morphology_review,
        "meta": meta,
    }


# =========================================================
# REVIEW ACQUISITION
# =========================================================

def run_manual_signal_review_only(signal, fs, runtime_cfg, extraction_kwargs):
    signal_qc = get_signal_qc_profile(runtime_cfg)
    review_record_type = runtime_cfg["recording_id"]

    if runtime_cfg.get("segment_label") is not None:
        review_record_type = f"{review_record_type} / segment={runtime_cfg.get('segment_label')}"

    return review_rpeaks_record(
        ecg_signal=signal,
        sampling_rate=fs,
        date=runtime_cfg["date"],
        phase=runtime_cfg["phase_id"],
        record_type=review_record_type,
        detector_method=extraction_kwargs["detector_method"],
        lowcut=extraction_kwargs["hrv_lowcut"],
        highcut=extraction_kwargs["hrv_highcut"],
        trim_seconds=0.0,
        edge_guard_sec=signal_qc.get("edge_guard_sec", 0.5),
        adc_min=extraction_kwargs["adc_min"],
        adc_max=extraction_kwargs["adc_max"],
        initial_window_sec=15.0,
    )


def run_manual_full_review(signal, fs, runtime_cfg, extraction_kwargs):
    signal_qc = get_signal_qc_profile(runtime_cfg)

    if not morphology_should_be_evaluated(runtime_cfg):
        signal_review = run_manual_signal_review_only(
            signal=signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
        )

        return build_review_result_from_parts(
            signal_review=signal_review,
            morphology_review=empty_morphology_review(status="not_run"),
            runtime_cfg=runtime_cfg,
            fs=fs,
        )

    return review_ecg_record(
        ecg_signal=signal,
        sampling_rate=fs,
        date=runtime_cfg["date"],
        phase=runtime_cfg["phase_id"],
        record_type=runtime_cfg["recording_id"],
        detector_method=extraction_kwargs["detector_method"],
        lowcut=extraction_kwargs["hrv_lowcut"],
        highcut=extraction_kwargs["hrv_highcut"],
        trim_seconds=0.0,
        edge_guard_sec=signal_qc.get("edge_guard_sec", 0.5),
        adc_min=extraction_kwargs["adc_min"],
        adc_max=extraction_kwargs["adc_max"],
        initial_window_sec=15.0,
        morph_lowcut=extraction_kwargs["morph_lowcut"],
        morph_highcut=extraction_kwargs["morph_highcut"],
        morph_pre_ms=extraction_kwargs["morph_pre_ms"],
        morph_post_ms=extraction_kwargs["morph_post_ms"],
        corr_threshold=extraction_kwargs["corr_threshold"],
        smooth_ms=extraction_kwargs["smooth_ms"],
        morphology_qc_profile=get_morphology_qc_profile(runtime_cfg),
    )


def acquire_whole_record_review(signal, fs, runtime_cfg, extraction_kwargs, interactive_mode):
    """
    Получает review_result для static / breath_hold.

    Важно:
    - signal_review = техническая основа: signal + rpeaks
    - hrv_review = пригодность для HRV
    - morphology_review = пригодность для morphology

    В bad_only ручной review показывается отдельно:
    - R-peak review только если signal_gate failed
    - morphology review только если morphology_gate failed
    """
    effective_mode = resolve_effective_review_mode(interactive_mode, runtime_cfg)
    signal_qc = get_signal_qc_profile(runtime_cfg)

    # =====================================================
    # 1. AUTO: всё автоматически
    # =====================================================
    if effective_mode is False:
        signal_gate = auto_review_gate(
            signal=signal,
            fs=fs,
            signal_qc_profile=signal_qc,
            adc_min=extraction_kwargs["adc_min"],
            adc_max=extraction_kwargs["adc_max"],
            detector_method=extraction_kwargs["detector_method"],
            lowcut=extraction_kwargs["hrv_lowcut"],
            highcut=extraction_kwargs["hrv_highcut"],
        )

        signal_review = build_auto_signal_review_result(
            signal=signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
            log_action="auto_signal",
            extra_log=signal_gate,
        )

        hrv_review = make_hrv_review(
            status="accepted" if signal_gate.get("passed", False) else "skipped",
            qc_gate=signal_gate,
            source="auto",
            log=[{"action": "auto_hrv_gate"}],
        )

        morphology_review = build_auto_morphology_review_result(
            signal_review=signal_review,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
            log_action_prefix="auto_morphology",
        )

        return build_review_result_from_parts(
            signal_review=signal_review,
            hrv_review=hrv_review,
            morphology_review=morphology_review,
            runtime_cfg=runtime_cfg,
            fs=fs,
        )

    # =====================================================
    # 2. MANUAL: всё вручную
    # =====================================================
    if effective_mode is True:
        signal_review = run_manual_signal_review_only(
            signal=signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
        )

        if signal_review["status"] == "accepted":
            hrv_review = make_hrv_review(
                status="accepted",
                qc_gate={
                    "passed": True,
                    "stage": "signal",
                    "reasons": ["manual_rpeak_review_accepted"],
                },
                source="manual",
                log=[{"action": "manual_hrv_accept"}],
            )
        else:
            hrv_review = make_hrv_review(
                status="skipped",
                qc_gate={
                    "passed": False,
                    "stage": "signal",
                    "reasons": ["manual_rpeak_review_skipped"],
                },
                source="manual",
                log=[{"action": "manual_hrv_skip"}],
            )

        if signal_review["status"] != "accepted":
            return build_review_result_from_parts(
                signal_review=signal_review,
                hrv_review=hrv_review,
                morphology_review=empty_morphology_review(status="not_run"),
                runtime_cfg=runtime_cfg,
                fs=fs,
            )

        if morphology_should_be_evaluated(runtime_cfg):
            rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=True)
            if len(rpeaks_for_morph) < 2:
                rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=False)

            morphology_review = review_morphology_record(
                ecg_signal=signal_review["signal"],
                rpeaks=rpeaks_for_morph,
                sampling_rate=fs,
                date=runtime_cfg["date"],
                phase=runtime_cfg["phase_id"],
                record_type=(
                    f"{runtime_cfg['recording_id']} / segment={runtime_cfg.get('segment_label')}"
                    if runtime_cfg.get("segment_label") is not None
                    else runtime_cfg["recording_id"]
                ),
                morph_lowcut=extraction_kwargs["morph_lowcut"],
                morph_highcut=extraction_kwargs["morph_highcut"],
                morph_pre_ms=extraction_kwargs["morph_pre_ms"],
                morph_post_ms=extraction_kwargs["morph_post_ms"],
                corr_threshold=extraction_kwargs["corr_threshold"],
                smooth_ms=extraction_kwargs["smooth_ms"],
                morphology_qc_profile=get_morphology_qc_profile(runtime_cfg),
            )
        else:
            morphology_review = empty_morphology_review(status="not_run")

        return build_review_result_from_parts(
            signal_review=signal_review,
            hrv_review=hrv_review,
            morphology_review=morphology_review,
            runtime_cfg=runtime_cfg,
            fs=fs,
        )

    # =====================================================
    # 3. BAD_ONLY: независимые gate для HRV и morphology
    # =====================================================
    if effective_mode == "bad_only":
        signal_gate = auto_review_gate(
            signal=signal,
            fs=fs,
            signal_qc_profile=signal_qc,
            adc_min=extraction_kwargs["adc_min"],
            adc_max=extraction_kwargs["adc_max"],
            detector_method=extraction_kwargs["detector_method"],
            lowcut=extraction_kwargs["hrv_lowcut"],
            highcut=extraction_kwargs["hrv_highcut"],
        )

        print(
            f"[GATE signal] {runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
            f"{runtime_cfg['recording_id']} | passed={signal_gate.get('passed')} "
            f"reasons={signal_gate.get('reasons')}"
        )

        # Сначала всегда строим auto technical signal_review.
        # Даже если HRV-gate failed, эта версия может быть пригодна для morphology.
        auto_signal_review = build_auto_signal_review_result(
            signal=signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
            log_action="auto_signal_for_bad_only",
            extra_log=signal_gate,
        )

        if signal_gate.get("passed", False):
            signal_review = auto_signal_review
            hrv_review = make_hrv_review(
                status="accepted",
                qc_gate=signal_gate,
                source="auto",
                log=[{"action": "auto_hrv_accept"}],
            )
        else:
            manual_signal_review = run_manual_signal_review_only(
                signal=signal,
                fs=fs,
                runtime_cfg=runtime_cfg,
                extraction_kwargs=extraction_kwargs,
            )

            if manual_signal_review["status"] == "accepted":
                signal_review = manual_signal_review
                hrv_review = make_hrv_review(
                    status="accepted",
                    qc_gate={
                        "passed": True,
                        "stage": "signal",
                        "reasons": ["manual_rpeak_review_accepted_after_gate_failed"],
                        "previous_gate": signal_gate,
                    },
                    source="manual",
                    log=[{"action": "manual_hrv_accept_after_gate_failed"}],
                )
            else:
                # Важно:
                # HRV отклонена, но auto_signal_review оставляем как техническую основу
                # для возможной морфологии.
                signal_review = auto_signal_review
                hrv_review = make_hrv_review(
                    status="skipped",
                    qc_gate={
                        "passed": False,
                        "stage": "signal",
                        "reasons": ["manual_rpeak_review_skipped_after_gate_failed"],
                        "previous_gate": signal_gate,
                    },
                    source="manual_skipped",
                    log=[{"action": "manual_hrv_skip_after_gate_failed"}],
                )

        # Теперь morphology проверяется отдельно по текущей technical signal_review.
        morphology_review = build_auto_morphology_review_result(
            signal_review=signal_review,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
            log_action_prefix="auto_morphology",
        )

        morph_gate = morphology_review.get("qc_gate")
        if isinstance(morph_gate, dict):
            print(
                f"[GATE morphology] {runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
                f"{runtime_cfg['recording_id']} | passed={morph_gate.get('passed')} "
                f"reasons={morph_gate.get('reasons')}"
            )

        if (
            morphology_should_be_evaluated(runtime_cfg)
            and morphology_review.get("status") != "accepted"
        ):
            rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=True)
            if len(rpeaks_for_morph) < 2:
                rpeaks_for_morph = select_review_rpeaks(signal_review, use_no_edge=False)

            morphology_review = review_morphology_record(
                ecg_signal=signal_review["signal"],
                rpeaks=rpeaks_for_morph,
                sampling_rate=fs,
                date=runtime_cfg["date"],
                phase=runtime_cfg["phase_id"],
                record_type=(
                    f"{runtime_cfg['recording_id']} / segment={runtime_cfg.get('segment_label')}"
                    if runtime_cfg.get("segment_label") is not None
                    else runtime_cfg["recording_id"]
                ),
                morph_lowcut=extraction_kwargs["morph_lowcut"],
                morph_highcut=extraction_kwargs["morph_highcut"],
                morph_pre_ms=extraction_kwargs["morph_pre_ms"],
                morph_post_ms=extraction_kwargs["morph_post_ms"],
                corr_threshold=extraction_kwargs["corr_threshold"],
                smooth_ms=extraction_kwargs["smooth_ms"],
                morphology_qc_profile=get_morphology_qc_profile(runtime_cfg),
            )

        return build_review_result_from_parts(
            signal_review=signal_review,
            hrv_review=hrv_review,
            morphology_review=morphology_review,
            runtime_cfg=runtime_cfg,
            fs=fs,
        )

    # fallback
    return acquire_whole_record_review(
        signal=signal,
        fs=fs,
        runtime_cfg=runtime_cfg,
        extraction_kwargs=extraction_kwargs,
        interactive_mode=True,
    )


def acquire_window_signal_review(
    seg_signal,
    fs,
    runtime_cfg,
    extraction_kwargs,
    window_idx,
    start_sec,
    end_sec,
    interactive_mode,
):
    effective_mode = resolve_effective_review_mode(interactive_mode, runtime_cfg)
    signal_qc = get_signal_qc_profile(runtime_cfg)

    if effective_mode is False:
        signal_gate = auto_review_gate(
            signal=seg_signal,
            fs=fs,
            signal_qc_profile=signal_qc,
            adc_min=extraction_kwargs["adc_min"],
            adc_max=extraction_kwargs["adc_max"],
            detector_method=extraction_kwargs["detector_method"],
            lowcut=extraction_kwargs["hrv_lowcut"],
            highcut=extraction_kwargs["hrv_highcut"],
        )

        signal_review = build_auto_signal_review_result(
            signal=seg_signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            extraction_kwargs=extraction_kwargs,
            log_action="auto_window_signal",
            extra_log=signal_gate,
        )

        hrv_review = make_hrv_review(
            status="accepted" if signal_gate.get("passed", False) else "skipped",
            qc_gate=signal_gate,
            source="auto",
            log=[{"action": "auto_window_hrv_gate"}],
        )

        review_result = build_review_result_from_parts(
            signal_review=signal_review,
            hrv_review=hrv_review,
            morphology_review=empty_morphology_review(status="not_run"),
            runtime_cfg=runtime_cfg,
            fs=fs,
        )

        review_result["meta"]["window_idx"] = window_idx
        review_result["meta"]["start_sec"] = start_sec
        review_result["meta"]["end_sec"] = end_sec

        return review_result

    if effective_mode == "bad_only":
        signal_gate = auto_review_gate(
            signal=seg_signal,
            fs=fs,
            signal_qc_profile=signal_qc,
            adc_min=extraction_kwargs["adc_min"],
            adc_max=extraction_kwargs["adc_max"],
            detector_method=extraction_kwargs["detector_method"],
            lowcut=extraction_kwargs["hrv_lowcut"],
            highcut=extraction_kwargs["hrv_highcut"],
        )

        print(
            f"[GATE signal window {window_idx}] {runtime_cfg['date']} | "
            f"{runtime_cfg['phase_id']} | {runtime_cfg['recording_id']} | "
            f"passed={signal_gate.get('passed')} reasons={signal_gate.get('reasons')}"
        )

        if signal_gate.get("passed", False):
            signal_review = build_auto_signal_review_result(
                signal=seg_signal,
                fs=fs,
                runtime_cfg=runtime_cfg,
                extraction_kwargs=extraction_kwargs,
                log_action="auto_accept_good_window",
                extra_log=signal_gate,
            )

            hrv_review = make_hrv_review(
                status="accepted",
                qc_gate=signal_gate,
                source="auto",
                log=[{"action": "auto_window_hrv_accept"}],
            )

            review_result = build_review_result_from_parts(
                signal_review=signal_review,
                hrv_review=hrv_review,
                morphology_review=empty_morphology_review(status="not_run"),
                runtime_cfg=runtime_cfg,
                fs=fs,
            )

            review_result["meta"]["window_idx"] = window_idx
            review_result["meta"]["start_sec"] = start_sec
            review_result["meta"]["end_sec"] = end_sec

            return review_result

    # manual или bad_only после failed gate
    signal_review = review_rpeaks_record(
        ecg_signal=seg_signal,
        sampling_rate=fs,
        date=runtime_cfg["date"],
        phase=runtime_cfg["phase_id"],
        record_type=f"{runtime_cfg['recording_id']}_w{window_idx}",
        detector_method=extraction_kwargs["detector_method"],
        lowcut=extraction_kwargs["hrv_lowcut"],
        highcut=extraction_kwargs["hrv_highcut"],
        trim_seconds=0.0,
        edge_guard_sec=signal_qc.get("edge_guard_sec", 0.5),
        adc_min=extraction_kwargs["adc_min"],
        adc_max=extraction_kwargs["adc_max"],
        initial_window_sec=end_sec - start_sec,
    )

    if signal_review["status"] == "accepted":
        hrv_review = make_hrv_review(
            status="accepted",
            qc_gate={
                "passed": True,
                "stage": "signal",
                "reasons": ["manual_window_rpeak_review_accepted"],
            },
            source="manual",
            log=[{"action": "manual_window_hrv_accept"}],
        )
    else:
        hrv_review = make_hrv_review(
            status="skipped",
            qc_gate={
                "passed": False,
                "stage": "signal",
                "reasons": ["manual_window_rpeak_review_skipped"],
            },
            source="manual",
            log=[{"action": "manual_window_hrv_skip"}],
        )

    review_result = build_review_result_from_parts(
        signal_review=signal_review,
        hrv_review=hrv_review,
        morphology_review=empty_morphology_review(status="not_run"),
        runtime_cfg=runtime_cfg,
        fs=fs,
    )

    review_result["meta"]["window_idx"] = window_idx
    review_result["meta"]["start_sec"] = start_sec
    review_result["meta"]["end_sec"] = end_sec

    return review_result


# =========================================================
# FEATURE ROW BUILDING
# =========================================================

def build_output_row(
        runtime_cfg,
        fs,
        segment,
        features,
        review_result,
        review_source,
):
    qc = features.get("qc", {}) or {}

    signal_review = review_result["signal_review"]
    hrv_review = review_result.get("hrv_review", {})
    morphology_review = review_result["morphology_review"]

    hrv_gate = hrv_review.get("qc_gate") if isinstance(hrv_review, dict) else None
    morph_gate = morphology_review.get("qc_gate") if isinstance(morphology_review, dict) else None

    row = {
        "date": runtime_cfg["date"],
        "phase": runtime_cfg["phase_id"],
        "record_type": runtime_cfg["recording_id"],
        "aggregation_role": runtime_cfg.get("aggregation_role"),

        "segment_label": segment["label"],
        "window_idx": segment.get("window_idx"),
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "duration_sec": segment["end_sec"] - segment["start_sec"],
        "sampling_rate": fs,

        "processing_mode": runtime_cfg.get("processing_mode"),
        "processing_profile": runtime_cfg.get("processing_profile_name"),
        "segmentation_profile": runtime_cfg.get("segmentation_profile_name"),
        "quality_profile": runtime_cfg.get("quality_profile_name"),

        "review_source": review_source,

        "signal_review_status": signal_review.get("status"),

        "hrv_review_status": hrv_review.get("status"),
        "hrv_validated": hrv_is_validated(review_result),
        "hrv_review_source": hrv_review.get("source"),
        "hrv_qc_passed": hrv_gate.get("passed") if isinstance(hrv_gate, dict) else None,
        "hrv_qc_reasons": gate_reasons_to_string(hrv_gate),

        "morphology_review_status": morphology_review.get("status"),
        "morphology_validated": morphology_is_validated(morphology_review),
        "morphology_error": morphology_review.get("error"),
        "morph_corr_threshold": morphology_review.get("corr_threshold"),
        "morphology_qc_passed": morph_gate.get("passed") if isinstance(morph_gate, dict) else None,
        "morphology_qc_reasons": gate_reasons_to_string(morph_gate),

        "clipping_ratio": qc.get("clipping_ratio"),
        "rr_phys_bad_ratio": qc.get("rr_phys_bad_ratio"),
        "n_rpeaks": qc.get("n_rpeaks"),
        "n_beats_extracted": qc.get("n_beats_extracted"),
        "n_beats_good": qc.get("n_beats_good"),
        "corr_min": qc.get("corr_min"),
        "corr_median": qc.get("corr_median"),
        "corr_max": qc.get("corr_max"),
    }

    row.update(flatten_features(features))
    return row


# =========================================================
# PROCESSING: STATIC / BREATH-HOLD
# =========================================================

def process_whole_record_mode(
    signal,
    fs,
    runtime_cfg,
    review_result,
    review_source,
):
    """
    Для processing_mode static / breath_hold.

    Review проводится на whole-record.
    Потом запись режется на сегменты через resolve_segments().
    Для breath_hold это full/start/end из protocol.yaml.
    """
    rows = []

    signal_review = review_result["signal_review"]
    morphology_review = review_result["morphology_review"]

    if signal_review["status"] != "accepted":
        return rows

    reviewed_signal = np.asarray(signal_review["signal"], dtype=float)

    segments = resolve_segments(
        signal_len=len(reviewed_signal),
        sampling_rate=fs,
        segmentation_cfg=runtime_cfg["segmentation"],
    )

    extraction_kwargs = build_extraction_kwargs_from_runtime(runtime_cfg)
    feature_groups = runtime_cfg.get("feature_groups")

    for segment in segments:
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]

        start_idx = int(round(start_sec * fs))
        end_idx = int(round(end_sec * fs))

        seg_signal = reviewed_signal[start_idx:end_idx]

        if len(seg_signal) == 0:
            continue

        segment_duration = end_sec - start_sec
        use_no_edge = segment_duration > 10.0

        base_rpeaks = select_review_rpeaks(signal_review, use_no_edge=use_no_edge)
        mask = (base_rpeaks >= start_idx) & (base_rpeaks < end_idx)
        seg_rpeaks = base_rpeaks[mask] - start_idx

        if len(seg_rpeaks) < 2:
            print(
                f"[WARN] Недостаточно R-пиков: "
                f"{runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
                f"{runtime_cfg['recording_id']} | {segment['label']}"
            )
            continue

        features, debug = extract_all_features_from_segment(
            ecg_signal=seg_signal,
            sampling_rate=fs,
            rpeaks=seg_rpeaks,
            **extraction_kwargs,
        )

        features = filter_feature_groups(features, feature_groups)

        if not hrv_is_validated(review_result):
            features = invalidate_hrv_features(features)

        if not morphology_is_validated(morphology_review):
            features = invalidate_morphology_features(features)

        row = build_output_row(
            runtime_cfg=runtime_cfg,
            fs=fs,
            segment=segment,
            features=features,
            review_result=review_result,
            review_source=review_source,
        )

        rows.append(row)

    return rows


# =========================================================
# PROCESSING: WINDOWED
# =========================================================

def process_windowed_mode(
    signal,
    fs,
    runtime_cfg,
    review_dir: Path = None,
    interactive_mode=None,
):
    rows = []

    segments = resolve_segments(
        signal_len=len(signal),
        sampling_rate=fs,
        segmentation_cfg=runtime_cfg["segmentation"],
    )

    extraction_kwargs = build_extraction_kwargs_from_runtime(runtime_cfg)
    feature_groups = runtime_cfg.get("feature_groups")

    for segment in segments:
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        window_idx = segment.get("window_idx")

        start_idx = int(round(start_sec * fs))
        end_idx = int(round(end_sec * fs))

        seg_signal = np.asarray(signal[start_idx:end_idx], dtype=float)

        if len(seg_signal) < max(10, int(2.0 * fs)):
            continue

        review_source = "new"
        review_result = None

        if review_dir is not None:
            suffix = f"window_{window_idx:02d}.review.json"
            review_path = build_review_path(review_dir, runtime_cfg, suffix=suffix)

            if review_path.exists():
                review_result = load_review_result(review_path)
                review_source = "cached"

        if review_result is None:
            segment_runtime_cfg = dict(runtime_cfg)
            segment_runtime_cfg["segment_label"] = segment["label"]
            segment_runtime_cfg["segment_start_sec"] = start_sec
            segment_runtime_cfg["segment_end_sec"] = end_sec
            segment_runtime_cfg["segment_window_idx"] = window_idx
            review_result = acquire_window_signal_review(
                seg_signal=seg_signal,
                fs=fs,
                runtime_cfg=segment_runtime_cfg,
                extraction_kwargs=extraction_kwargs,
                window_idx=window_idx,
                start_sec=start_sec,
                end_sec=end_sec,
                interactive_mode=interactive_mode,
            )

            if review_dir is not None:
                save_review_result(review_result, review_path)

        signal_review = review_result["signal_review"]
        morphology_review = review_result["morphology_review"]

        if signal_review["status"] != "accepted":
            continue

        reviewed_signal = np.asarray(signal_review["signal"], dtype=float)
        reviewed_rpeaks = select_review_rpeaks(signal_review, use_no_edge=False)

        if len(reviewed_signal) == 0 or len(reviewed_rpeaks) < 2:
            continue

        features, debug = extract_all_features_from_segment(
            ecg_signal=reviewed_signal,
            sampling_rate=fs,
            rpeaks=reviewed_rpeaks,
            **extraction_kwargs,
        )

        features = filter_feature_groups(features, feature_groups)
        features = invalidate_morphology_features(features)

        row = build_output_row(
            runtime_cfg=runtime_cfg,
            fs=fs,
            segment=segment,
            features=features,
            review_result=review_result,
            review_source=review_source,
        )

        rows.append(row)

    return rows

def process_segment_review_mode(
    signal,
    fs,
    runtime_cfg,
    review_dir: Path = None,
    interactive_mode=None,
):
    """
    Для записей, где review_scope == 'segment'.

    Пример: breath_hold_default:
      - full
      - start
      - end

    Каждый сегмент проходит отдельный review и отдельно сохраняется.
    """
    rows = []

    segments = resolve_segments(
        signal_len=len(signal),
        sampling_rate=fs,
        segmentation_cfg=runtime_cfg["segmentation"],
    )

    extraction_kwargs = build_extraction_kwargs_from_runtime(runtime_cfg)
    feature_groups = runtime_cfg.get("feature_groups")

    for segment in segments:
        label = segment["label"]
        window_idx = segment.get("window_idx")
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]

        start_idx = int(round(start_sec * fs))
        end_idx = int(round(end_sec * fs))

        seg_signal = np.asarray(signal[start_idx:end_idx], dtype=float)

        if len(seg_signal) < max(10, int(2.0 * fs)):
            print(
                f"[WARN] Слишком короткий сегмент: "
                f"{runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
                f"{runtime_cfg['recording_id']} | {label}"
            )
            continue

        review_source = "new"
        review_result = None

        if review_dir is not None:
            seg_name = safe_segment_name(label, window_idx)
            suffix = f"{seg_name}.review.json"
            review_path = build_review_path(review_dir, runtime_cfg, suffix=suffix)

            if review_path.exists():
                candidate = load_review_result(review_path)

                # Если ты уже добавил новую структуру с hrv_review:
                if (
                    isinstance(candidate, dict)
                    and "signal_review" in candidate
                    and "morphology_review" in candidate
                ):
                    review_result = candidate
                    review_source = "cached"
                else:
                    print(f"[WARN] Старый/несовместимый review будет проигнорирован: {review_path}")

        if review_result is None:
            # Важно: acquire_whole_record_review здесь применяется к ОДНОМУ СЕГМЕНТУ,
            # поэтому в review попадёт не вся breath_hold запись, а только full/start/end.
            segment_runtime_cfg = dict(runtime_cfg)
            segment_runtime_cfg["segment_label"] = label
            segment_runtime_cfg["segment_start_sec"] = start_sec
            segment_runtime_cfg["segment_end_sec"] = end_sec
            segment_runtime_cfg["segment_window_idx"] = window_idx

            review_result = acquire_whole_record_review(
                signal=seg_signal,
                fs=fs,
                runtime_cfg=segment_runtime_cfg,
                extraction_kwargs=extraction_kwargs,
                interactive_mode=interactive_mode,
            )

            if review_dir is not None:
                save_review_result(review_result, review_path)

        signal_review = review_result["signal_review"]
        morphology_review = review_result["morphology_review"]

        if signal_review["status"] != "accepted":
            print(
                f"[INFO] Сегмент пропущен после signal review: "
                f"{runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
                f"{runtime_cfg['recording_id']} | {label}"
            )
            continue

        reviewed_signal = np.asarray(signal_review["signal"], dtype=float)

        # Для коротких start/end лучше не выкидывать edge-пики.
        segment_duration = len(reviewed_signal) / fs
        use_no_edge = segment_duration > 10.0

        reviewed_rpeaks = select_review_rpeaks(
            signal_review,
            use_no_edge=use_no_edge,
        )

        if len(reviewed_signal) == 0 or len(reviewed_rpeaks) < 2:
            print(
                f"[WARN] Недостаточно R-пиков после review: "
                f"{runtime_cfg['date']} | {runtime_cfg['phase_id']} | "
                f"{runtime_cfg['recording_id']} | {label}"
            )
            continue

        features, debug = extract_all_features_from_segment(
            ecg_signal=reviewed_signal,
            sampling_rate=fs,
            rpeaks=reviewed_rpeaks,
            **extraction_kwargs,
        )

        features = filter_feature_groups(features, feature_groups)

        # Если у тебя уже добавлен hrv_review:
        if "hrv_review" in review_result:
            if not hrv_is_validated(review_result):
                features = invalidate_hrv_features(features)

        if not morphology_is_validated(morphology_review):
            features = invalidate_morphology_features(features)

        row = build_output_row(
            runtime_cfg=runtime_cfg,
            fs=fs,
            segment=segment,
            features=features,
            review_result=review_result,
            review_source=review_source,
        )

        rows.append(row)

    return rows

# =========================================================
# MAIN RECORD DISPATCH
# =========================================================

def process_record_with_runtime_config(
    runtime_cfg,
    review_dir: Path = None,
    interactive_mode=None,
):
    rows = []

    file_path = resolve_existing_file(runtime_cfg)

    if file_path is None:
        print(
            f"[WARN] Не найден файл: "
            f"{runtime_cfg['date']} | {runtime_cfg['phase_id']} | {runtime_cfg['recording_id']}"
        )
        return rows

    signal = load_ecg_csv(file_path)
    fs = estimate_sampling_rate_from_runtime(signal, runtime_cfg)

    mode = runtime_cfg["processing_mode"]

    if mode in {"static", "breath_hold"}:
        review_scope = runtime_cfg.get("review_scope", "record")

        if review_scope == "segment":
            return process_segment_review_mode(
                signal=signal,
                fs=fs,
                runtime_cfg=runtime_cfg,
                review_dir=review_dir,
                interactive_mode=interactive_mode,
            )

        if review_scope == "record":
            review_source = "new"
            review_result = None

            if review_dir is not None:
                review_path = build_review_path(review_dir, runtime_cfg)

                if review_path.exists():
                    candidate = load_review_result(review_path)

                    if (
                            isinstance(candidate, dict)
                            and "signal_review" in candidate
                            and "morphology_review" in candidate
                    ):
                        review_result = candidate
                        review_source = "cached"
                    else:
                        print(f"[WARN] Старый/несовместимый review будет проигнорирован: {review_path}")

            if review_result is None:
                extraction_kwargs = build_extraction_kwargs_from_runtime(runtime_cfg)

                review_result = acquire_whole_record_review(
                    signal=signal,
                    fs=fs,
                    runtime_cfg=runtime_cfg,
                    extraction_kwargs=extraction_kwargs,
                    interactive_mode=interactive_mode,
                )

                if review_dir is not None:
                    save_review_result(review_result, review_path)

            return process_whole_record_mode(
                signal=signal,
                fs=fs,
                runtime_cfg=runtime_cfg,
                review_result=review_result,
                review_source=review_source,
            )

        raise ValueError(f"Unsupported review_scope: {review_scope}")

    if mode == "windowed":
        return process_windowed_mode(
            signal=signal,
            fs=fs,
            runtime_cfg=runtime_cfg,
            review_dir=review_dir,
            interactive_mode=interactive_mode,
        )

    raise ValueError(f"Unsupported processing_mode: {mode}")


# =========================================================
# PROTOCOL LOOP
# =========================================================

def process_protocol_day(
    protocol_cfg,
    root_dir: Path,
    date_str: str,
    review_dir: Path = None,
    interactive_mode=None,
):
    rows = []

    for phase in protocol_cfg["phases"]:
        phase_id = phase["id"]

        for recording in protocol_cfg["recordings"]:
            if phase_id not in recording["phases"]:
                continue

            runtime_cfg = build_record_runtime_config(
                protocol_cfg=protocol_cfg,
                date_value=date_str,
                phase_id=phase_id,
                recording=recording,
                root_dir=root_dir,
            )

            print(f"[PROCESS] {date_str} | {phase_id} | {recording['id']}")

            rec_rows = process_record_with_runtime_config(
                runtime_cfg=runtime_cfg,
                review_dir=review_dir,
                interactive_mode=interactive_mode,
            )

            rows.extend(rec_rows)

    return rows


def process_all_records_with_protocol(
    root_dir: Path,
    protocol_config_path: Path,
    output_csv: Path,
    review_dir: Path = None,
    interactive_mode=None,
):
    """
    interactive_mode:
        None       -> брать review_mode из protocol.yaml
        True       -> ручной review всех подходящих записей
        False      -> полностью автоматический режим
        "bad_only" -> ручной review только если gate не прошёл
    """
    root_dir = Path(root_dir)
    protocol_config_path = Path(protocol_config_path)
    output_csv = Path(output_csv)

    if review_dir is not None:
        review_dir = Path(review_dir)

    cfg_raw = load_protocol_config(protocol_config_path)
    validate_protocol_config(cfg_raw)
    cfg = normalize_protocol_config(cfg_raw)

    date_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    all_rows = []

    for day_dir in date_dirs:
        if review_dir is not None and day_dir.resolve() == review_dir.resolve():
            continue

        date_str = day_dir.name

        print(f"\n[INFO] Обработка дня: {date_str}")

        day_rows = process_protocol_day(
            protocol_cfg=cfg,
            root_dir=root_dir,
            date_str=date_str,
            review_dir=review_dir,
            interactive_mode=interactive_mode,
        )

        all_rows.extend(day_rows)

        if all_rows:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(all_rows)
            df = reorder_output_columns(df)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[AUTO-SAVE] Промежуточно сохранено строк: {len(all_rows)}")

    if all_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows)
        df = reorder_output_columns(df)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Сохранено строк: {len(all_rows)} -> {output_csv}")
    else:
        print("[WARN] Нет данных для сохранения.")


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переставляет столбцы выходного CSV так, чтобы таблица была удобнее
    для ручного анализа.

    Логика:
    1) идентификаторы записи
    2) флаги валидности
    3) признаки HRV
    4) признаки морфологии
    5) прочие признаки
    6) служебные поля
    """

    def existing(cols):
        return [c for c in cols if c in df.columns]

    def by_prefix(prefixes, exclude=None):
        exclude = set(exclude or [])
        out = []
        for c in df.columns:
            if c in exclude:
                continue
            if any(c.startswith(p) for p in prefixes):
                out.append(c)
        return out

    # -----------------------------------------------------
    # 1. Основные идентификаторы
    # -----------------------------------------------------
    id_cols = existing([
        "record_type",
        "phase",
        "date",
        "segment_label",
        "window_idx",
        "start_sec",
        "end_sec",
        "duration_sec",
    ])

    # -----------------------------------------------------
    # 2. Главные флаги пригодности
    # -----------------------------------------------------
    validity_cols = existing([
        "hrv_validated",
        "morphology_validated",
    ])

    # -----------------------------------------------------
    # 3. HRV: сначала самые полезные time-domain признаки
    # -----------------------------------------------------
    hrv_time_priority = existing([
        "hrv_time__n_rr",
        "hrv_time__MeanHR_bpm",
        "hrv_time__HR_min_bpm",
        "hrv_time__HR_max_bpm",
        "hrv_time__HR_std_bpm",
        "hrv_time__MeanNN_ms",
        "hrv_time__MedianNN_ms",
        "hrv_time__SDNN_ms",
        "hrv_time__RMSSD_ms",
        "hrv_time__SDSD_ms",
        "hrv_time__pNN20_percent",
        "hrv_time__pNN50_percent",
        "hrv_time__NN_min_ms",
        "hrv_time__NN_max_ms",
        "hrv_time__NN_range_ms",
    ])

    hrv_time_rest = [
        c for c in by_prefix(["hrv_time__"])
        if c not in hrv_time_priority
    ]

    # -----------------------------------------------------
    # 4. HRV extended: frequency + nonlinear
    # -----------------------------------------------------
    hrv_freq_priority = existing([
        "hrv_freq__VLF_power",
        "hrv_freq__LF_power",
        "hrv_freq__HF_power",
        "hrv_freq__Total_power",
        "hrv_freq__LF_HF_ratio",
        "hrv_freq__LF_nu",
        "hrv_freq__HF_nu",
        "hrv_freq__LF_peak_Hz",
        "hrv_freq__HF_peak_Hz",
    ])

    hrv_freq_rest = [
        c for c in by_prefix(["hrv_freq__"])
        if c not in hrv_freq_priority
    ]

    hrv_nonlinear_priority = existing([
        "hrv_nonlinear__SD1_ms",
        "hrv_nonlinear__SD2_ms",
        "hrv_nonlinear__SD1_SD2_ratio",
        "hrv_nonlinear__Ellipse_area",
        "hrv_nonlinear__ApproxEntropy",
        "hrv_nonlinear__SampleEntropy",
        "hrv_nonlinear__ShannonEntropy",
        "hrv_nonlinear__DFA_alpha1",
        "hrv_nonlinear__DFA_alpha2",
        "hrv_nonlinear__FractalDimension",
        "hrv_nonlinear__poincare_valid",
    ])

    hrv_nonlinear_rest = [
        c for c in by_prefix(["hrv_nonlinear__"])
        if c not in hrv_nonlinear_priority
    ]

    # -----------------------------------------------------
    # 5. Морфология: сначала QRS, потом P/T
    # -----------------------------------------------------
    morph_qrs_priority = existing([
        "morph_qrs__QRS_main_amp",
        "morph_qrs__Q_amp",
        "morph_qrs__S_amp",
        "morph_qrs__RS_amp",

        "morph_qrs__QRS_duration_ms",
        "morph_qrs__QRS_width_ms",
        "morph_qrs__QRS_width_20_ms",
        "morph_qrs__R_width_half_ms",
        "morph_qrs__QRS_left_width_ms",
        "morph_qrs__QRS_right_width_ms",

        "morph_qrs__RQ_interval_ms",
        "morph_qrs__RS_interval_ms",

        "morph_qrs__QRS_area",
        "morph_qrs__Positive_area",
        "morph_qrs__Negative_area",

        "morph_qrs__R_up_slope",
        "morph_qrs__R_down_slope",

        "morph_qrs__QRS_main_idx",
        "morph_qrs__Q_idx",
        "morph_qrs__S_idx",
        "morph_qrs__QRS_onset_idx",
        "morph_qrs__QRS_offset_idx",
    ])

    morph_qrs_rest = [
        c for c in by_prefix(["morph_qrs__"])
        if c not in morph_qrs_priority
    ]

    morph_p_priority = existing([
        "morph_p__P_present",
        "morph_p__P_amp",
        "morph_p__P_width_ms",
        "morph_p__PR_interval_ms",
        "morph_p__PR_peak_interval_ms",
        "morph_p__P_area",
        "morph_p__P_snr",
    ])

    morph_p_rest = [
        c for c in by_prefix(["morph_p__"])
        if c not in morph_p_priority
    ]

    morph_t_priority = existing([
        "morph_t__T_present",
        "morph_t__T_amp",
        "morph_t__T_width_ms",
        "morph_t__RT_interval_ms",
        "morph_t__T_peak_interval_ms",
        "morph_t__QT_like_ms",
        "morph_t__T_area",
        "morph_t__T_snr",
    ])

    morph_t_rest = [
        c for c in by_prefix(["morph_t__"])
        if c not in morph_t_priority
    ]

    # -----------------------------------------------------
    # 6. Служебные поля
    # -----------------------------------------------------
    service_cols = existing([
        "aggregation_role",

        "sampling_rate",
        "processing_mode",
        "processing_profile",
        "segmentation_profile",
        "signal_qc_profile",
        "morphology_qc_profile",

        "review_source",
        "signal_review_status",

        "hrv_review_status",
        "hrv_review_source",
        "hrv_qc_passed",
        "hrv_qc_reasons",

        "morphology_review_status",
        "morphology_error",
        "morph_corr_threshold",
        "morphology_qc_passed",
        "morphology_qc_reasons",

        "clipping_ratio",
        "rr_phys_bad_ratio",
        "n_rpeaks",
        "n_beats_extracted",
        "n_beats_good",
        "corr_min",
        "corr_median",
        "corr_max",
    ])

    # flattened meta/qc тоже считаем служебными
    service_prefix_cols = by_prefix([
        "meta__",
        "qc__",
    ])

    ordered = []

    groups = [
        id_cols,
        validity_cols,

        hrv_time_priority,
        hrv_time_rest,
        hrv_freq_priority,
        hrv_freq_rest,
        hrv_nonlinear_priority,
        hrv_nonlinear_rest,

        morph_qrs_priority,
        morph_qrs_rest,
        morph_p_priority,
        morph_p_rest,
        morph_t_priority,
        morph_t_rest,

        service_cols,
        service_prefix_cols,
    ]

    for group in groups:
        for c in group:
            if c not in ordered:
                ordered.append(c)

    # Всё, что не попало в группы, ставим перед служебным хвостом,
    # если это похоже на признак, или в самый конец, если неизвестно.
    already = set(ordered)

    unknown_cols = [c for c in df.columns if c not in already]

    # Разделим неизвестные на потенциально аналитические и явно служебные
    unknown_service_keywords = (
        "review",
        "qc",
        "profile",
        "source",
        "error",
        "threshold",
        "validated",
    )

    unknown_feature_cols = [
        c for c in unknown_cols
        if not any(k in c.lower() for k in unknown_service_keywords)
    ]

    unknown_service_cols = [
        c for c in unknown_cols
        if c not in unknown_feature_cols
    ]

    # Вставляем неизвестные признаки перед служебным хвостом
    final_cols = []

    main_analytical_cols = (
        id_cols
        + validity_cols
        + hrv_time_priority + hrv_time_rest
        + hrv_freq_priority + hrv_freq_rest
        + hrv_nonlinear_priority + hrv_nonlinear_rest
        + morph_qrs_priority + morph_qrs_rest
        + morph_p_priority + morph_p_rest
        + morph_t_priority + morph_t_rest
    )

    for c in main_analytical_cols:
        if c in df.columns and c not in final_cols:
            final_cols.append(c)

    for c in unknown_feature_cols:
        if c not in final_cols:
            final_cols.append(c)

    service_tail = service_cols + service_prefix_cols + unknown_service_cols

    for c in service_tail:
        if c in df.columns and c not in final_cols:
            final_cols.append(c)

    # На всякий случай добавляем любые оставшиеся колонки
    for c in df.columns:
        if c not in final_cols:
            final_cols.append(c)

    return df[final_cols]
