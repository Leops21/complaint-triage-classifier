from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from complaint_triage.modeling.evaluate_transformer import evaluate_transformer
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model.")
    parser.add_argument("--task", choices=["triage", "escalation"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clean_text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = evaluate_transformer(
        task=args.task,
        model_dir=args.model_dir,
        test_path=args.test,
        output_dir=args.output_dir,
        text_column=args.text_column,
    )
    logger.info("Wrote evaluation outputs to %s", args.output_dir)
    logger.info("Evaluated rows: %s", len(predictions))


if __name__ == "__main__":
    main()
