.PHONY: setup test lint profile prepare labels splits baseline-triage baseline-escalation train-distilbert-triage train-distilbert-escalation

setup:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

profile:
	python scripts/00_profile_raw_data.py --input data/raw/complaints.csv --output reports/metrics/raw_profile.json

prepare:
	python scripts/01_prepare_data.py --input data/raw/complaints.csv --output data/interim/complaints_clean.csv --profile-output reports/metrics/clean_profile.json

labels:
	python scripts/02_bootstrap_labels.py --input data/interim/complaints_clean.csv --schema configs/label_schema.yaml --output data/labeled/bootstrap_labels.csv --review-output data/labeled/manual_review_queue.csv --sample-size 8000

splits:
	python scripts/03_make_splits.py --input data/labeled/verified_labels.csv --output-dir data/processed

baseline-triage:
	python scripts/04_train_baseline.py --task triage --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv --output-dir models/baseline/triage

baseline-escalation:
	python scripts/04_train_baseline.py --task escalation --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv --output-dir models/baseline/escalation

train-distilbert-triage:
	python scripts/05_train_transformer.py --task triage --config configs/distilbert.yaml --train data/processed/train.csv --val data/processed/val.csv --output-dir models/distilbert/triage

train-distilbert-escalation:
	python scripts/05_train_transformer.py --task escalation --config configs/distilbert.yaml --train data/processed/train.csv --val data/processed/val.csv --output-dir models/distilbert/escalation
