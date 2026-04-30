from pathlib import Path

import yaml

from context_features import build_context_features_with_protocol


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

INPUT_DIR = MODULE_DIR / "input"
OUTPUT_DIR = MODULE_DIR / "output"
PROTOCOL_CONFIG = PROJECT_DIR / "protocol.yaml"
RUNTIME_PROTOCOL_CONFIG = OUTPUT_DIR / "_runtime_protocol.yaml"

INPUT_CSV = INPUT_DIR / "daily_data_extended.csv"
COMPUTED_OUTPUT_CSV = OUTPUT_DIR / "context_computed.csv"
DERIVED_OUTPUT_CSV = OUTPUT_DIR / "context_derived.csv"


def build_runtime_protocol() -> Path:
    with PROTOCOL_CONFIG.open("r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f) or {}

    context_cfg = protocol.setdefault("context_features", {})
    context_cfg["input"] = str(INPUT_CSV)
    context_cfg["computed_output"] = str(COMPUTED_OUTPUT_CSV)
    context_cfg["output"] = str(DERIVED_OUTPUT_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_PROTOCOL_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(protocol, f, allow_unicode=True, sort_keys=False)

    return RUNTIME_PROTOCOL_CONFIG


def main() -> None:
    runtime_protocol = build_runtime_protocol()
    build_context_features_with_protocol(
        root_dir=MODULE_DIR,
        protocol_config_path=runtime_protocol,
        debug=True,
    )


if __name__ == "__main__":
    main()
