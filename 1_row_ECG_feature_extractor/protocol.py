from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import json

from technical_defaults import (
    get_technical_processing,
    get_technical_signal_qc,
    get_technical_morphology_qc,
)

class ProtocolConfigError(ValueError):
    """Ошибка структуры или содержимого конфигурации протокола."""

# =========================================================
# HELPERS для перехода на protocol
# =========================================================
def resolve_existing_file(runtime_cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Возвращает первый существующий файл из file_candidates.
    """
    for path in runtime_cfg.get("file_candidates", []):
        if Path(path).exists():
            return Path(path)
    return None


def estimate_sampling_rate_from_runtime(signal, runtime_cfg: Dict[str, Any]) -> float:
    """
    Оценивает sampling rate на основе runtime-конфига.
    Поддержано:
      - fixed
      - from_expected_duration
    """
    signal_len = len(signal)
    mode = runtime_cfg.get("sampling_rate_mode", "fixed")

    if mode == "fixed":
        fs = runtime_cfg.get("sampling_rate_hz")
        if fs is None:
            raise ProtocolConfigError(
                f"Для записи '{runtime_cfg.get('recording_id')}' не задан sampling_rate_hz."
            )
        return float(fs)

    if mode == "from_expected_duration":
        expected_duration = runtime_cfg.get("expected_duration_sec")
        if expected_duration is None or expected_duration <= 0:
            raise ProtocolConfigError(
                f"Для записи '{runtime_cfg.get('recording_id')}' "
                f"нужен expected_duration_sec при sampling_rate_mode='from_expected_duration'."
            )
        return float(signal_len) / float(expected_duration)

    raise ProtocolConfigError(
        f"Неподдерживаемый sampling_rate_mode='{mode}' "
        f"для записи '{runtime_cfg.get('recording_id')}'."
    )


def resolve_segments(signal_len: int, sampling_rate: float, segmentation_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Преобразует segmentation_profile в список сегментов:
      [{
          "label": ...,
          "start_sec": ...,
          "end_sec": ...,
          "window_idx": ...
      }]
    """
    total_sec = float(signal_len) / float(sampling_rate)
    mode = segmentation_cfg.get("mode", "full")

    if mode == "full":
        return [{
            "label": "full",
            "start_sec": 0.0,
            "end_sec": total_sec,
            "window_idx": None,
        }]

    if mode == "fixed_windows":
        window_sec = float(segmentation_cfg["window_sec"])
        step_sec = float(segmentation_cfg.get("step_sec", window_sec))
        min_window_sec = float(segmentation_cfg.get("min_window_sec", window_sec))
        drop_last_short = bool(segmentation_cfg.get("drop_last_short_window", True))

        segments = []
        start_sec = 0.0
        idx = 1

        while start_sec < total_sec - 1e-9:
            end_sec = min(start_sec + window_sec, total_sec)
            dur = end_sec - start_sec

            if dur < min_window_sec and drop_last_short:
                break

            segments.append({
                "label": "window",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_idx": idx,
            })

            idx += 1
            start_sec += step_sec

        return segments

    if mode == "named_subsegments":
        raw_segments = segmentation_cfg.get("segments", [])
        if not isinstance(raw_segments, list) or len(raw_segments) == 0:
            raise ProtocolConfigError("named_subsegments требует непустой список segments.")

        out = []

        for i, seg in enumerate(raw_segments):
            label = seg.get("label", f"segment_{i+1}")

            if "relative_to" in seg:
                rel = seg["relative_to"]
                if rel != "end":
                    raise ProtocolConfigError(
                        f"Сейчас поддержано только relative_to='end', получено '{rel}'."
                    )
                duration_sec = float(seg["duration_sec"])
                start_sec = max(0.0, total_sec - duration_sec)
                end_sec = total_sec
            else:
                start_sec = float(seg.get("start_sec", 0.0))
                end_raw = seg.get("end_sec", "full")
                end_sec = total_sec if end_raw == "full" else float(end_raw)

            start_sec = max(0.0, start_sec)
            end_sec = min(total_sec, end_sec)

            if end_sec <= start_sec:
                continue

            out.append({
                "label": label,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "window_idx": None,
            })

        return out

    raise ProtocolConfigError(f"Неподдерживаемый segmentation mode: '{mode}'")

def build_review_path(review_dir: Path, runtime_cfg, suffix=None):
    date = runtime_cfg["date"]
    phase = runtime_cfg["phase_id"]
    rec_id = runtime_cfg["recording_id"]

    if suffix is None:
        return review_dir / date / phase / f"{rec_id}.review.json"

    return review_dir / date / phase / rec_id / suffix

# ---------------------------------------------------------
# LOADING
# ---------------------------------------------------------

def load_protocol_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Загружает конфиг протокола из YAML или JSON.

    Поддерживаемые расширения:
      - .yaml / .yml
      - .json
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Для чтения YAML нужен пакет PyYAML. Установи: pip install pyyaml"
            ) from e

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    else:
        raise ProtocolConfigError(
            f"Неподдерживаемый формат конфигурации: {suffix}. "
            f"Ожидается .yaml/.yml или .json"
        )

    if not isinstance(data, dict):
        raise ProtocolConfigError("Корневой объект конфига должен быть словарём.")

    return data


# ---------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------

def _is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _ensure_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolConfigError(f"Раздел '{key}' должен быть словарём.")
    return value


def _normalize_phases_list(phases: Any) -> List[Dict[str, Any]]:
    """
    Принимает:
      - [{"id": "before", "label": "..."}]
      - ["before", "after"]

    Возвращает список словарей вида:
      [{"id": "before", "label": "before"}, ...]
    """
    if not isinstance(phases, list) or len(phases) == 0:
        raise ProtocolConfigError("Раздел 'phases' должен быть непустым списком.")

    out = []
    seen = set()

    for i, item in enumerate(phases):
        if isinstance(item, str):
            phase_id = item
            phase_label = item
        elif isinstance(item, dict):
            phase_id = item.get("id")
            phase_label = item.get("label", phase_id)
        else:
            raise ProtocolConfigError(
                f"Элемент phases[{i}] должен быть строкой или словарём."
            )

        if not _is_nonempty_string(phase_id):
            raise ProtocolConfigError(f"У phases[{i}] отсутствует корректный 'id'.")

        if phase_id in seen:
            raise ProtocolConfigError(f"Дублирующийся phase id: '{phase_id}'.")

        seen.add(phase_id)
        out.append({"id": phase_id, "label": phase_label})

    return out


def _resolve_recording_value(
    recording: Dict[str, Any],
    defaults: Dict[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    if key in recording:
        return recording[key]
    if key in defaults:
        return defaults[key]
    return fallback

# ---------------------------------------------------------
# Адаптеры для technical_defaults
# ---------------------------------------------------------

def _get_subject_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Поддерживает новый блок subject и старый subject_defaults.
    Новый предпочтительный формат:

    subject:
      sampling_rate_hz: 234.45
      adc_range:
        min: 0
        max: 675
    """
    if "subject_defaults" in cfg:
        return deepcopy(cfg.get("subject_defaults") or {})

    subject = deepcopy(cfg.get("subject") or {})

    out = {}

    if "sampling_rate_hz" in subject:
        out["sampling_rate_mode"] = "fixed"
        out["sampling_rate_hz"] = subject["sampling_rate_hz"]

    if "adc_range" in subject:
        out["adc"] = deepcopy(subject["adc_range"])

    return out


def _hr_bpm_to_rr_bounds(expected_hr_bpm: Dict[str, Any]) -> Dict[str, float]:
    """
    expected_hr_bpm:
      min: 40
      max: 180

    Превращается в:
      rr_min_sec = 60 / HRmax
      rr_max_sec = 60 / HRmin
    """
    hr_min = expected_hr_bpm.get("min")
    hr_max = expected_hr_bpm.get("max")

    if hr_min is None or hr_max is None:
        return {}

    hr_min = float(hr_min)
    hr_max = float(hr_max)

    if hr_min <= 0 or hr_max <= 0 or hr_min >= hr_max:
        raise ProtocolConfigError(
            "expected_hr_bpm должен иметь положительные min/max, где min < max."
        )

    return {
        "rr_min_sec": 60.0 / hr_max,
        "rr_max_sec": 60.0 / hr_min,
    }


def build_signal_qc_from_quality_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Пользователь задаёт понятный expected_hr_bpm.
    Код превращает его в технический signal_qc.
    """
    signal_qc = get_technical_signal_qc()

    expected_hr = profile.get("expected_hr_bpm")
    if isinstance(expected_hr, dict):
        signal_qc.update(_hr_bpm_to_rr_bounds(expected_hr))

    hrv_cfg = profile.get("hrv", {})
    if isinstance(hrv_cfg, dict):
        signal_qc["hrv_enabled"] = bool(hrv_cfg.get("enabled", True))
    else:
        signal_qc["hrv_enabled"] = True

    return signal_qc


def build_morphology_qc_from_quality_profile(profile: Dict[str, Any]):
    """
    Пользователь задаёт morphology.enabled и ожидаемый QRS.
    Код превращает это в технический morphology_qc.
    """
    morph_user = profile.get("morphology", {})

    if morph_user is None:
        return None

    if not isinstance(morph_user, dict):
        raise ProtocolConfigError("quality_profile.morphology должен быть словарём.")

    if not bool(morph_user.get("enabled", True)):
        return None

    morphology_qc = get_technical_morphology_qc()

    qrs = morph_user.get("expected_qrs_duration_ms", {})
    if isinstance(qrs, dict):
        if "min" in qrs:
            morphology_qc["qrs_duration_min_ms"] = float(qrs["min"])
        else:
            morphology_qc["qrs_duration_min_ms"] = morphology_qc[
                "qrs_duration_fallback_min_ms"
            ]

        if "max" in qrs:
            morphology_qc["qrs_duration_max_ms"] = float(qrs["max"])
        else:
            morphology_qc["qrs_duration_max_ms"] = morphology_qc[
                "qrs_duration_fallback_max_ms"
            ]
    else:
        morphology_qc["qrs_duration_min_ms"] = morphology_qc[
            "qrs_duration_fallback_min_ms"
        ]
        morphology_qc["qrs_duration_max_ms"] = morphology_qc[
            "qrs_duration_fallback_max_ms"
        ]

    morphology_qc["require_qrs"] = True
    morphology_qc["require_p"] = bool(morph_user.get("require_p_wave", False))
    morphology_qc["require_t"] = bool(morph_user.get("require_t_wave", False))

    return morphology_qc


def build_feature_groups_from_quality_profile(profile: Dict[str, Any], fallback=None):
    """
    Если пользователь не задал feature_groups явно,
    формируем их из hrv.enabled и morphology.enabled.
    """
    fallback = fallback or ["hrv_time", "hrv_extended", "morphology"]

    out = []

    hrv_cfg = profile.get("hrv", {})
    hrv_enabled = True
    if isinstance(hrv_cfg, dict):
        hrv_enabled = bool(hrv_cfg.get("enabled", True))

    morph_cfg = profile.get("morphology", {})
    morphology_enabled = True
    if morph_cfg is None:
        morphology_enabled = False
    elif isinstance(morph_cfg, dict):
        morphology_enabled = bool(morph_cfg.get("enabled", True))

    if hrv_enabled:
        out.extend(["hrv_time", "hrv_extended"])

    if morphology_enabled:
        out.append("morphology")

    return out if out else fallback

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_protocol_config(cfg: Dict[str, Any]) -> None:
    """
    Проверяет базовую корректность конфига.
    Бросает ProtocolConfigError с агрегированным сообщением при ошибках.
    """
    errors: List[str] = []

    if not isinstance(cfg, dict):
        raise ProtocolConfigError("Конфиг должен быть словарём.")

    try:
        phases = _normalize_phases_list(cfg.get("phases"))
        phase_ids = {p["id"] for p in phases}
    except Exception as e:
        raise ProtocolConfigError(str(e)) from e

    subject_defaults = _get_subject_defaults(cfg)
    storage = _ensure_dict(cfg, "storage")
    defaults = _ensure_dict(cfg, "defaults")
    quality_profiles = _ensure_dict(cfg, "quality_profiles")
    segmentation_profiles = _ensure_dict(cfg, "segmentation_profiles")


    # built-in fallback, чтобы defaults.segmentation_profile: full_record
    # не ломал validate_protocol_config()
    segmentation_profiles = dict(segmentation_profiles)
    segmentation_profiles.setdefault("full_record", {"mode": "full"})

    recordings = cfg.get("recordings")
    if not isinstance(recordings, list) or len(recordings) == 0:
        errors.append("Раздел 'recordings' должен быть непустым списком.")
        recordings = []

    # subject_defaults
    sampling_rate_mode = subject_defaults.get("sampling_rate_mode", "fixed")
    if sampling_rate_mode not in {"fixed", "from_expected_duration"}:
        errors.append(
            "subject_defaults.sampling_rate_mode должен быть 'fixed' "
            "или 'from_expected_duration'."
        )

    if sampling_rate_mode == "fixed":
        sr = subject_defaults.get("sampling_rate_hz")
        if sr is None:
            errors.append(
                "Для sampling_rate_mode='fixed' нужен subject_defaults.sampling_rate_hz."
            )

    adc = subject_defaults.get("adc", {})
    if adc is not None and not isinstance(adc, dict):
        errors.append("subject_defaults.adc должен быть словарём.")
    elif isinstance(adc, dict):
        if "min" in adc and "max" in adc and adc["min"] >= adc["max"]:
            errors.append("subject_defaults.adc.min должен быть меньше adc.max.")

    # storage
    if "phases_as_directories" in storage and not isinstance(storage["phases_as_directories"], bool):
        errors.append("storage.phases_as_directories должен быть bool.")

    # profiles sections
    for sec_name, sec in [
        ("segmentation_profiles", segmentation_profiles),
        ("quality_profiles", quality_profiles),
    ]:
        for k, v in sec.items():
            if not _is_nonempty_string(k):
                errors.append(f"В разделе '{sec_name}' найден пустой ключ.")
            if not isinstance(v, dict):
                errors.append(f"{sec_name}.{k} должен быть словарём.")

    # recordings
    seen_recording_ids = set()

    for i, rec in enumerate(recordings):
        if not isinstance(rec, dict):
            errors.append(f"recordings[{i}] должен быть словарём.")
            continue

        rec_id = rec.get("id")
        if not _is_nonempty_string(rec_id):
            errors.append(f"recordings[{i}] не содержит корректный 'id'.")
            continue

        if rec_id in seen_recording_ids:
            errors.append(f"Дублирующийся recording id: '{rec_id}'.")
        seen_recording_ids.add(rec_id)

        file_names = rec.get("file_names")
        if not isinstance(file_names, list) or len(file_names) == 0:
            errors.append(f"recordings[{i}] ('{rec_id}') должен иметь непустой list file_names.")
        else:
            for j, fn in enumerate(file_names):
                if not _is_nonempty_string(fn):
                    errors.append(f"recordings[{i}].file_names[{j}] должен быть непустой строкой.")

        rec_phases = rec.get("phases", list(phase_ids))
        if not isinstance(rec_phases, list) or len(rec_phases) == 0:
            errors.append(f"recordings[{i}] ('{rec_id}') должен иметь непустой list phases.")
        else:
            for ph in rec_phases:
                if ph not in phase_ids:
                    errors.append(
                        f"recordings[{i}] ('{rec_id}') ссылается на неизвестную phase '{ph}'."
                    )

        processing_mode = rec.get("mode", rec.get("processing_mode", "static"))
        if processing_mode not in {"static", "windowed", "breath_hold"}:
            errors.append(
                f"recordings[{i}] ('{rec_id}') имеет unsupported processing_mode='{processing_mode}'."
            )

        segmentation_profile_name = _resolve_recording_value(
            rec,
            defaults,
            "segmentation",
            _resolve_recording_value(rec, defaults, "segmentation_profile", "full_record"),
        )
        if segmentation_profile_name is not None and segmentation_profile_name not in segmentation_profiles:
            errors.append(
                f"recordings[{i}] ('{rec_id}') ссылается на неизвестный segmentation_profile "
                f"'{segmentation_profile_name}'."
            )

        quality_profile_name = _resolve_recording_value(
            rec,
            defaults,
            "quality_profile",
        )
        if quality_profile_name is None:
            errors.append(
                f"recordings[{i}] ('{rec_id}') не имеет quality_profile "
                f"и он не задан в defaults."
            )
        elif quality_profile_name not in quality_profiles:
            errors.append(
                f"recordings[{i}] ('{rec_id}') ссылается на неизвестный "
                f"quality_profile '{quality_profile_name}'."
            )

        review_scope = _resolve_recording_value(
            rec,
            defaults,
            "review_scope",
            "record",
        )
        if review_scope not in {"record", "segment"}:
            errors.append(
                f"recordings[{i}] ('{rec_id}') имеет unsupported review_scope='{review_scope}'. "
                f"Ожидается 'record' или 'segment'."
            )

    # built-in check for segmentation modes
    for seg_name, seg_cfg in segmentation_profiles.items():
        mode = seg_cfg.get("mode")
        if mode not in {"full", "fixed_windows", "named_subsegments"}:
            errors.append(
                f"segmentation_profiles.{seg_name}.mode должен быть "
                f"'full', 'fixed_windows' или 'named_subsegments'."
            )

    if errors:
        raise ProtocolConfigError("Ошибки в конфиге протокола:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_protocol_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Возвращает нормализованную копию конфига:
    - phases -> список словарей + map phases_by_id
    - recordings -> список с подставленными defaults
    - recordings_by_id
    - built-in full_record segmentation profile, если его нет
    """
    validate_protocol_config(cfg)

    out = deepcopy(cfg)

    out.setdefault("protocol", {})
    out.setdefault("subject_defaults", {})
    out.setdefault("storage", {})
    out.setdefault("defaults", {})
    out.setdefault("quality_profiles", {})
    out.setdefault("segmentation_profiles", {})


    out["subject_defaults"].setdefault("sampling_rate_mode", "fixed")

    # built-in segmentation fallback
    out["segmentation_profiles"].setdefault("full_record", {"mode": "full"})

    # default storage
    out["storage"].setdefault("date_format", "%Y-%m-%d")
    out["storage"].setdefault("phases_as_directories", True)
    out["storage"].setdefault("default_signal_extension", ".csv")

    # normalize phases
    normalized_phases = _normalize_phases_list(out["phases"])
    out["phases"] = normalized_phases
    out["phases_by_id"] = {p["id"]: p for p in normalized_phases}
    phase_ids = [p["id"] for p in normalized_phases]

    defaults = out["defaults"]

    normalized_recordings = []
    recordings_by_id = {}

    for rec in out["recordings"]:
        nr = deepcopy(rec)

        nr.setdefault("phases", phase_ids)
        nr.setdefault("processing_mode", "static")

        # Новое имя mode, но внутри runtime pipeline всё ещё ждёт processing_mode
        if "mode" not in nr and "processing_mode" in nr:
            nr["mode"] = nr["processing_mode"]

        nr.setdefault("mode", "static")

        if "segmentation" not in nr:
            nr["segmentation"] = defaults.get(
                "segmentation",
                defaults.get("segmentation_profile", "full_record"),
            )

        if "quality_profile" not in nr and "quality_profile" in defaults:
            nr["quality_profile"] = defaults["quality_profile"]

        if "review_mode" not in nr and "review_mode" in defaults:
            nr["review_mode"] = defaults["review_mode"]

        if "review_scope" not in nr:
            nr["review_scope"] = defaults.get("review_scope", "record")

        if "feature_groups" not in nr and "feature_groups" in defaults:
            nr["feature_groups"] = deepcopy(defaults["feature_groups"])

        normalized_recordings.append(nr)
        recordings_by_id[nr["id"]] = nr

    out["recordings"] = normalized_recordings
    out["recordings_by_id"] = recordings_by_id
    out["_normalized"] = True

    return out


# ---------------------------------------------------------
# RUNTIME CONFIG BUILDING
# ---------------------------------------------------------

def _format_date_value(date_value: Any, date_format: str) -> str:
    if isinstance(date_value, str):
        return date_value
    if hasattr(date_value, "strftime"):
        return date_value.strftime(date_format)
    return str(date_value)


def build_record_runtime_config(
    protocol_cfg: Dict[str, Any],
    date_value: Any,
    phase_id: str,
    recording: Union[str, Dict[str, Any]],
    root_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Строит runtime-конфиг для конкретной записи.

    recording:
      - либо id записи (str),
      - либо уже нормализованный словарь записи.
    """
    if not protocol_cfg.get("_normalized", False):
        protocol_cfg = normalize_protocol_config(protocol_cfg)

    root_dir = Path(root_dir)

    if phase_id not in protocol_cfg["phases_by_id"]:
        raise ProtocolConfigError(f"Неизвестная фаза: '{phase_id}'.")

    if isinstance(recording, str):
        try:
            rec = protocol_cfg["recordings_by_id"][recording]
        except KeyError as e:
            raise ProtocolConfigError(f"Неизвестная запись: '{recording}'.") from e
    elif isinstance(recording, dict):
        rec = deepcopy(recording)
    else:
        raise ProtocolConfigError("Аргумент recording должен быть str или dict.")

    if phase_id not in rec["phases"]:
        raise ProtocolConfigError(
            f"Запись '{rec['id']}' не предусмотрена для фазы '{phase_id}'."
        )

    subject_defaults = _get_subject_defaults(protocol_cfg)
    storage = protocol_cfg["storage"]

    sampling_rate_mode = rec.get(
        "sampling_rate_mode",
        subject_defaults.get("sampling_rate_mode", "fixed"),
    )
    sampling_rate_hz = rec.get(
        "sampling_rate_hz",
        subject_defaults.get("sampling_rate_hz"),
    )

    if sampling_rate_mode not in {"fixed", "from_expected_duration"}:
        raise ProtocolConfigError(
            f"Unsupported sampling_rate_mode='{sampling_rate_mode}' "
            f"для записи '{rec['id']}'."
        )

    if sampling_rate_mode == "fixed" and sampling_rate_hz is None:
        raise ProtocolConfigError(
            f"Для записи '{rec['id']}' нужен sampling_rate_hz при mode='fixed'."
        )

    adc_cfg = {
        "min": 0,
        "max": 675,
    }
    adc_cfg.update(deepcopy(subject_defaults.get("adc", {}) or {}))

    adc_override = rec.get("adc")
    if isinstance(adc_override, dict):
        adc_cfg.update(adc_override)

    date_str = _format_date_value(date_value, storage["date_format"])
    date_dir = root_dir / date_str

    if storage.get("phases_as_directories", True):
        base_dir = date_dir / phase_id
    else:
        base_dir = date_dir

    file_names = rec["file_names"]
    file_candidates = [base_dir / fn for fn in file_names]

    quality_profile_name = rec.get("quality_profile")
    if quality_profile_name is None:
        raise ProtocolConfigError(
            f"Для записи '{rec['id']}' не задан quality_profile."
        )

    try:
        quality_profile = deepcopy(protocol_cfg["quality_profiles"][quality_profile_name])
    except KeyError as e:
        raise ProtocolConfigError(
            f"Неизвестный quality_profile '{quality_profile_name}' "
            f"для записи '{rec['id']}'."
        ) from e

    processing_profile_name = "technical_default"
    processing_profile = get_technical_processing()

    segmentation_profile_name = rec.get("segmentation", rec.get("segmentation_profile", "full_record"))
    segmentation_profile = deepcopy(
        protocol_cfg["segmentation_profiles"].get(
            segmentation_profile_name,
            {"mode": "full"},
        )
    )

    signal_qc_profile = build_signal_qc_from_quality_profile(quality_profile)
    morphology_qc_profile = build_morphology_qc_from_quality_profile(quality_profile)

    runtime_cfg = {
        "protocol": {
            "name": protocol_cfg.get("protocol", {}).get("name"),
            "version": protocol_cfg.get("protocol", {}).get("version"),
        },
        "date": date_str,
        "phase_id": phase_id,
        "phase_label": protocol_cfg["phases_by_id"][phase_id].get("label", phase_id),

        "recording_id": rec["id"],
        "recording_label": rec.get("label", rec["id"]),
        "aggregation_role": rec.get("aggregation_role"),

        "root_dir": root_dir,
        "date_dir": date_dir,
        "base_dir": base_dir,
        "file_candidates": file_candidates,
        "preferred_file_path": file_candidates[0] if file_candidates else None,

        "sampling_rate_mode": sampling_rate_mode,
        "sampling_rate_hz": float(sampling_rate_hz) if sampling_rate_hz is not None else None,
        "expected_duration_sec": rec.get("expected_duration_sec"),

        "adc": {
            "min": adc_cfg.get("min"),
            "max": adc_cfg.get("max"),
        },

        "processing_mode": rec.get("mode", rec.get("processing_mode", "static")),
        "processing_profile_name": processing_profile_name,
        "processing": processing_profile,

        "segmentation_profile_name": segmentation_profile_name,
        "segmentation": segmentation_profile,

        "quality_profile_name": quality_profile_name,
        "quality_profile": quality_profile,

        # Эти поля оставляем для совместимости с pipeline.py
        "signal_qc_profile_name": quality_profile_name,
        "signal_qc": signal_qc_profile,

        "morphology_qc_profile_name": quality_profile_name if morphology_qc_profile is not None else None,
        "morphology_qc": morphology_qc_profile,

        "review_mode": rec.get("review_mode"),
        "review_scope": rec.get("review_scope", "record"),
        "feature_groups": deepcopy(
            rec.get(
                "feature_groups",
                build_feature_groups_from_quality_profile(
                    quality_profile,
                    fallback=protocol_cfg.get("defaults", {}).get("feature_groups"),
                ),
            )
        ),
    }

    return runtime_cfg

if __name__ == "__main__":
    cfg_raw = load_protocol_config("protocol.yaml")
    validate_protocol_config(cfg_raw)
    cfg = normalize_protocol_config(cfg_raw)

    runtime_cfg = build_record_runtime_config(
        protocol_cfg=cfg,
        date_value="2026-04-01",
        phase_id="before",
        recording="sit",
        root_dir=r"C:\Users\pv190\PyCharmMiscProject\records",
    )

    print(runtime_cfg["preferred_file_path"])
    print(runtime_cfg["processing_mode"])
    print(runtime_cfg["processing"])
    print(runtime_cfg["signal_qc"])
    print(runtime_cfg["morphology_qc"])