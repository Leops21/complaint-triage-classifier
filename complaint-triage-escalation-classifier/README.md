# Customer Complaint Triage and Escalation Classifier

A transformer-based text classification project for routing consumer financial complaint narratives into operational triage categories and identifying complaints that should be escalated for urgent human review.

The project is designed for the CFPB Consumer Complaint Database CSV export. You will manually download the CSV and place it at:

```text
data/raw/complaints.csv
```

The raw file is intentionally not tracked in Git because it can be large.

---

## 1. Project goal

Large organizations receive high volumes of free-text complaints, support tickets, and user reports. Manual triage is slow, inconsistent, and expensive.

This project builds a reproducible machine learning system that reads the `Consumer complaint narrative` field and predicts:

1. **Triage category** — the operational team or workflow that should handle the complaint.
2. **Escalation label** — whether the complaint should be urgently reviewed by a specialist.

The system is stronger than basic sentiment analysis because it maps unstructured text to a real business workflow: routing, escalation, risk prioritization, and operational analytics.

---

## 2. Prediction tasks

### Task A — Multi-class triage classification

The model predicts one of six custom business labels:

| Label ID | Business label |
|---|---|
| `fraud_identity_risk` | Fraud / scam / identity risk |
| `billing_payment_dispute` | Billing / payment dispute |
| `account_access_management` | Account access / login / account management |
| `documentation_processing_verification` | Documentation / processing / verification issue |
| `customer_service_failure` | Customer service failure |
| `legal_compliance_regulatory` | Legal / compliance / regulatory concern |

### Task B — Binary escalation classification

The model predicts:

| Label ID | Meaning |
|---|---|
| `do_not_escalate` | Normal triage queue |
| `escalate` | Urgent human review |

For escalation, recall on `escalate` is treated as especially important because a false negative can be operationally expensive.

---

## 3. Dataset

Use the CFPB Consumer Complaint Database CSV export.

Expected raw columns:

```text
Date received
Product
Sub-product
Issue
Sub-issue
Consumer complaint narrative
Company public response
Company
State
ZIP code
Tags
Consumer consent provided?
Submitted via
Date sent to company
Company response to consumer
Timely response?
Consumer disputed?
Complaint ID
```

Only records with a non-empty `Consumer complaint narrative` are used.

---

## 4. Repository structure

```text
.
├── configs/
│   ├── base.yaml
│   ├── distilbert.yaml
│   ├── roberta.yaml
│   └── label_schema.yaml
├── data/
│   ├── raw/                  # Put complaints.csv here
│   ├── interim/              # Cleaned complaint records
│   ├── labeled/              # Weak labels + manually verified labels
│   └── processed/            # Train/val/test splits
├── docs/
│   ├── annotation_guidelines.md
│   ├── data_dictionary.md
│   ├── experiment_plan.md
│   ├── model_card.md
│   └── problem_statement.md
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── error_analysis/
├── scripts/
│   ├── 00_profile_raw_data.py
│   ├── 01_prepare_data.py
│   ├── 02_bootstrap_labels.py
│   ├── 03_make_splits.py
│   ├── 04_train_baseline.py
│   ├── 05_train_transformer.py
│   ├── 06_evaluate.py
│   ├── 07_predict.py
│   ├── 08_error_analysis.py
│   └── 09_cross_validate.py
├── src/complaint_triage/
│   ├── data/
│   ├── labeling/
│   ├── modeling/
│   ├── analysis/
│   └── utils/
└── tests/
```

---

## 5. Environment setup

### Option A — standard Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Option B — requirements file

```bash
pip install -r requirements.txt
```

---

## 6. End-to-end workflow

### Step 0 — Add the raw CFPB file

Place the manually downloaded file here:

```text
data/raw/complaints.csv
```

### Step 1 — Profile the raw data

```bash
python scripts/00_profile_raw_data.py \
  --input data/raw/complaints.csv \
  --output reports/metrics/raw_profile.json
```

### Step 2 — Clean and filter narratives

```bash
python scripts/01_prepare_data.py \
  --input data/raw/complaints.csv \
  --output data/interim/complaints_clean.csv \
  --profile-output reports/metrics/clean_profile.json
```

This step:

- validates required columns,
- keeps only non-empty narratives,
- removes duplicate complaint IDs,
- creates `raw_text`,
- creates `clean_text`,
- computes text length features.

### Step 3 — Create weak labels and manual review file

```bash
python scripts/02_bootstrap_labels.py \
  --input data/interim/complaints_clean.csv \
  --schema configs/label_schema.yaml \
  --output data/labeled/bootstrap_labels.csv \
  --review-output data/labeled/manual_review_queue.csv \
  --sample-size 8000
```

This step creates candidate labels using rules from `configs/label_schema.yaml`.

You should manually review `data/labeled/manual_review_queue.csv`, correct labels, and save the final file as:

```text
data/labeled/verified_labels.csv
```

The final file must contain:

```text
Complaint ID
clean_text
triage_label
escalation_label
```

Additional metadata columns are allowed.

### Step 4 — Create train/validation/test splits

```bash
python scripts/03_make_splits.py \
  --input data/labeled/verified_labels.csv \
  --output-dir data/processed \
  --test-size 0.15 \
  --val-size 0.15 \
  --seed 42
```

If `verified_labels.csv` is not ready, you can experiment with:

```bash
python scripts/03_make_splits.py \
  --input data/labeled/bootstrap_labels.csv \
  --output-dir data/processed
```

For the final report, use manually verified labels.

### Step 5 — Train a fast baseline

```bash
python scripts/04_train_baseline.py \
  --task triage \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --test data/processed/test.csv \
  --output-dir models/baseline/triage
```

```bash
python scripts/04_train_baseline.py \
  --task escalation \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --test data/processed/test.csv \
  --output-dir models/baseline/escalation
```

### Step 6 — Fine-tune DistilBERT

```bash
python scripts/05_train_transformer.py \
  --task triage \
  --config configs/distilbert.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/distilbert/triage
```

```bash
python scripts/05_train_transformer.py \
  --task escalation \
  --config configs/distilbert.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/distilbert/escalation
```

### Step 7 — Fine-tune RoBERTa

```bash
python scripts/05_train_transformer.py \
  --task triage \
  --config configs/roberta.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/roberta/triage
```

```bash
python scripts/05_train_transformer.py \
  --task escalation \
  --config configs/roberta.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/roberta/escalation
```

### Step 8 — Evaluate on the held-out test set

```bash
python scripts/06_evaluate.py \
  --task triage \
  --model-dir models/distilbert/triage \
  --test data/processed/test.csv \
  --output-dir reports/metrics/distilbert_triage
```

```bash
python scripts/06_evaluate.py \
  --task escalation \
  --model-dir models/distilbert/escalation \
  --test data/processed/test.csv \
  --output-dir reports/metrics/distilbert_escalation
```

This creates:

- `classification_report.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `predictions.csv`
- `pr_curve.png` for escalation
- `roc_curve.png` for escalation

### Step 9 — Error analysis

```bash
python scripts/08_error_analysis.py \
  --predictions reports/metrics/distilbert_triage/predictions.csv \
  --output-dir reports/error_analysis/distilbert_triage
```

### Step 10 — Cross-validation baseline

```bash
python scripts/09_cross_validate.py \
  --task triage \
  --input data/labeled/verified_labels.csv \
  --folds 3 \
  --output reports/metrics/cv_triage_baseline.json
```

---

## 7. Recommended experiment table for the final report

| Model | Task | Main metric | Secondary metric | Notes |
|---|---:|---:|---:|---|
| TF-IDF + Logistic Regression | Triage | Macro F1 | Weighted F1 | Fast baseline |
| DistilBERT | Triage | Macro F1 | Weighted F1 | Efficient transformer |
| RoBERTa-base | Triage | Macro F1 | Weighted F1 | Stronger transformer |
| TF-IDF + Logistic Regression | Escalation | Escalate recall | Escalate precision | Cost-sensitive baseline |
| DistilBERT | Escalation | Escalate recall | PR-AUC | Efficient transformer |
| RoBERTa-base | Escalation | Escalate recall | PR-AUC | Stronger transformer |

---

## 8. Notes on annotation quality

The weak labels are not intended to be final ground truth. They are a bootstrap tool.

For the final project, use:

```text
data/labeled/verified_labels.csv
```

This makes the project defensible because the final labels are your custom operational categories, not merely CFPB's original product labels.

---

## 9. Reproducibility

- All random seeds are configurable.
- All major scripts expose CLI arguments.
- Training outputs are written to versioned model/report directories.
- The raw dataset is never modified.
- The train/validation/test splits are deterministic given the same seed.

---

## 10. Ethical and privacy notes

The project uses public CFPB complaint narratives that consumers opted to publish. The repo still treats narratives as sensitive operational text:

- no raw data is committed,
- predictions are written locally,
- examples in docs are synthetic or paraphrased,
- manual reviewers should avoid copying private details into reports.

---

## 11. Useful official references

- CFPB Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
- CFPB Consumer Complaint Database API field reference: https://cfpb.github.io/api/ccdb/fields.html
- CFPB data-use and narrative publication details: https://www.consumerfinance.gov/complaint/data-use/
- Hugging Face text classification task guide: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
- Hugging Face fine-tuning guide: https://huggingface.co/docs/transformers/en/training

## GPU setup

Transformer training uses Hugging Face `Trainer`, which can use CUDA automatically when a CUDA-enabled PyTorch build is installed. Before running DistilBERT or RoBERTa, verify GPU access:

```powershell
python scripts/10_check_gpu.py
```

If the script reports `cuda_available: false` on a machine with an NVIDIA GPU, reinstall PyTorch with a CUDA wheel. See `docs/gpu_setup.md` for the full Windows setup.

