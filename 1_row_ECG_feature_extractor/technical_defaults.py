from copy import deepcopy


TECHNICAL_SIGNAL_QC = {
    # Доля грубо плохих RR-интервалов, после которой нужен ручной review
    "bad_rr_fraction_max": 0.10,

    # Доля подозрительных R-пиков, после которой нужен ручной review
    "suspicious_peak_fraction_max": 0.15,

    # Минимальное число R-пиков для анализа
    "min_rpeaks": 3,

    # Техническая защита краёв записи
    "edge_guard_sec": 0.5,

    # Проверка согласованности амплитуды R-пиков
    "rpeak_amp_ratio_low": 0.40,
    "rpeak_amp_ratio_high": 2.50,
    "rpeak_amp_bad_ratio_max": 0.15,
    "rpeak_amp_median_min": None,

    # Проверка локальных выбросов RR
    "local_rr_ratio_low": 0.70,
    "local_rr_ratio_high": 1.50,
}


TECHNICAL_MORPHOLOGY_QC = {
    # Сколько комплексов нужно для морфологии
    "min_beats_extracted": 3,
    "min_beats_good": 3,
    "good_beats_ratio_min": 0.40,

    # Корреляционная фильтрация комплексов
    "beat_corr_threshold": 0.70,
    "corr_median_min": 0.70,

    # Fallback-диапазон QRS, если пользователь не задал свой
    "qrs_duration_fallback_min_ms": 20,
    "qrs_duration_fallback_max_ms": 180,

    "require_qrs": True,
    "require_p": False,
    "require_t": False,
}


TECHNICAL_PROCESSING = {
    "detector_method": "neurokit",

    "hrv_filter": {
        "lowcut": 0.5,
        "highcut": 30,
        "order": 2,
        "trim_seconds": 0.5,
    },

    "morphology_filter": {
        "lowcut": 0.3,
        "highcut": 35,
        "order": 2,
        "trim_seconds": 0.0,
    },

    "morphology": {
        # Пока оставляем фиксированное окно внутри кода.
        # Позже сюда можно добавить mode: auto.
        "pre_ms": 200,
        "post_ms": 400,
        "smooth_ms": 15,
    },
}


def get_technical_processing():
    return deepcopy(TECHNICAL_PROCESSING)


def get_technical_signal_qc():
    return deepcopy(TECHNICAL_SIGNAL_QC)


def get_technical_morphology_qc():
    return deepcopy(TECHNICAL_MORPHOLOGY_QC)