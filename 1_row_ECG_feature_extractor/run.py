from pathlib import Path

from pipeline import process_all_records_with_protocol


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

INPUT_DIR = MODULE_DIR / "input"
OUTPUT_DIR = MODULE_DIR / "output"
PROTOCOL_CONFIG = PROJECT_DIR / "protocol.yaml"

# The extractor expects date folders directly inside ROOT_DIR.
ROOT_DIR = INPUT_DIR / "sample_data"
OUTPUT_CSV = OUTPUT_DIR / "features_protocol.csv"
REVIEW_DIR = OUTPUT_DIR / "_reviews"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    process_all_records_with_protocol(
        root_dir=ROOT_DIR,
        protocol_config_path=PROTOCOL_CONFIG,
        output_csv=OUTPUT_CSV,
        review_dir=REVIEW_DIR,
        interactive_mode=None,  # None means: use review_mode from protocol.yaml.
    )


if __name__ == "__main__":
    main()
