from __future__ import annotations

import argparse

from .builder import build_analysis_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis_dataset.csv and diagnostic reports.")
    parser.add_argument("protocol", help="Path to protocol.yaml")
    parser.add_argument("--output-dir", default=None, help="Directory for output files. Defaults to protocol directory.")
    args = parser.parse_args()
    result = build_analysis_dataset(args.protocol, output_dir=args.output_dir)
    print("Analysis dataset built successfully")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
