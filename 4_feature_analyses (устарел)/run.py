from pathlib import Path
import importlib.util
import sys


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

INPUT_DIR = MODULE_DIR / "input"
OUTPUT_DIR = MODULE_DIR / "output"
PROTOCOL_CONFIG = PROJECT_DIR / "protocol.yaml"

DATASET_CSV = INPUT_DIR / "analysis_dataset.csv"
FEATURE_CATALOG_CSV = INPUT_DIR / "analysis_feature_catalog.csv"
FEATURE_ANALYSIS_SCRIPT = MODULE_DIR / "feature_analyses_v2" / "feature_analysis_v2.py"


def load_feature_analysis_module():
    spec = importlib.util.spec_from_file_location(
        "_feature_analysis_v2",
        FEATURE_ANALYSIS_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load feature analysis script: {FEATURE_ANALYSIS_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    module = load_feature_analysis_module()
    sys.argv = [
        str(FEATURE_ANALYSIS_SCRIPT),
        "--dataset",
        str(DATASET_CSV),
        "--catalog",
        str(FEATURE_CATALOG_CSV),
        "--protocol",
        str(PROTOCOL_CONFIG),
        "--out",
        str(OUTPUT_DIR),
    ]
    module.main()


if __name__ == "__main__":
    main()
