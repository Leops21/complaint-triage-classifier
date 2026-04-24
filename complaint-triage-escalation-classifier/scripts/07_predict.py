from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json

from complaint_triage.modeling.predict import predict_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one complaint narrative.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = predict_texts(args.model_dir, [args.text])
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
