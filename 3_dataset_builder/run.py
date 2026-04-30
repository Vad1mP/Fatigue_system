from pathlib import Path
import sys
import types

import yaml


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

INPUT_DIR = MODULE_DIR / "input"
OUTPUT_DIR = MODULE_DIR / "output"
PROTOCOL_CONFIG = PROJECT_DIR / "protocol.yaml"
RUNTIME_PROTOCOL_CONFIG = OUTPUT_DIR / "_runtime_protocol.yaml"

INPUT_TABLES = {
    "raw_ecg_features": INPUT_DIR / "features_protocol.csv",
    "ecg_derived": INPUT_DIR / "features_derived.csv",
    "context_computed": INPUT_DIR / "context_computed.csv",
    "context_derived": INPUT_DIR / "context_derived.csv",
}


def import_builder():
    package_name = "_fatigue_dataset_builder"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_DIR)]
    sys.modules.setdefault(package_name, package)

    from _fatigue_dataset_builder.builder import build_analysis_dataset

    return build_analysis_dataset


def build_runtime_protocol() -> Path:
    with PROTOCOL_CONFIG.open("r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f) or {}

    analysis_cfg = protocol.setdefault("analysis_dataset", {})
    table_cfgs = analysis_cfg.setdefault("tables", {})

    for table_id, path in INPUT_TABLES.items():
        table_cfgs.setdefault(table_id, {})
        table_cfgs[table_id]["path"] = str(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_PROTOCOL_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(protocol, f, allow_unicode=True, sort_keys=False)

    return RUNTIME_PROTOCOL_CONFIG


def main() -> None:
    runtime_protocol = build_runtime_protocol()
    build_analysis_dataset = import_builder()
    result = build_analysis_dataset(runtime_protocol, output_dir=OUTPUT_DIR)

    print("Analysis dataset built successfully")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
