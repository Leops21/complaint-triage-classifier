# Repository Walkthrough

## Main scripts

| Script | Purpose |
|---|---|
| `00_profile_raw_data.py` | Inspect the raw CFPB CSV |
| `01_prepare_data.py` | Clean and filter complaint narratives |
| `02_bootstrap_labels.py` | Create weak labels and manual review queue |
| `03_make_splits.py` | Create train/validation/test splits |
| `04_train_baseline.py` | Train TF-IDF + Logistic Regression baseline |
| `05_train_transformer.py` | Fine-tune DistilBERT or RoBERTa |
| `06_evaluate.py` | Evaluate a trained transformer model |
| `07_predict.py` | Run inference on new text |
| `08_error_analysis.py` | Create error-analysis outputs |
| `09_cross_validate.py` | Run baseline cross-validation |

## Suggested final report sections

1. Introduction and business motivation
2. Dataset description
3. Label design and annotation guidelines
4. Preprocessing
5. Baseline model
6. Transformer fine-tuning setup
7. Evaluation metrics
8. Results
9. Confusion matrix analysis
10. Escalation false-negative analysis
11. Operational impact
12. Limitations and future work
