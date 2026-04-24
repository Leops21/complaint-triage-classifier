from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from complaint_triage.modeling.baseline import train_baseline
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--task", choices=["triage", "escalation"], required=True)
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clean_text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_baseline(
        task=args.task,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        text_column=args.text_column,
    )
    logger.info("Wrote baseline outputs to %s", args.output_dir)
    logger.info("Macro F1: %.4f", report["macro avg"]["f1-score"])


if __name__ == "__main__":
    main()
