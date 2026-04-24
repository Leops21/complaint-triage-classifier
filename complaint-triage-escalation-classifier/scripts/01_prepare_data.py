from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from complaint_triage.data.build_dataset import build_clean_dataset
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean CFPB complaint narratives.")
    parser.add_argument("--input", default="data/raw/complaints.csv")
    parser.add_argument("--output", default="data/interim/complaints_clean.csv")
    parser.add_argument("--profile-output", default="reports/metrics/clean_profile.json")
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--nrows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = build_clean_dataset(
        input_path=args.input,
        output_path=args.output,
        profile_output_path=args.profile_output,
        min_text_chars=args.min_text_chars,
        nrows=args.nrows,
    )
    logger.info("Wrote clean dataset to %s", args.output)
    logger.info("Clean rows with narratives: %s", len(clean))


if __name__ == "__main__":
    main()
