from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from complaint_triage.analysis.error_analysis import run_error_analysis
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create error-analysis files from predictions.csv.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_error_analysis(args.predictions, args.output_dir)
    logger.info("Wrote error analysis to %s", args.output_dir)
    logger.info("Errors: %s", len(outputs["errors"]))


if __name__ == "__main__":
    main()
