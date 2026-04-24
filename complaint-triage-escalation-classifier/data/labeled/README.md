# Labeled data

Generated files:

- `bootstrap_labels.csv`: weak-label output from `scripts/02_bootstrap_labels.py`
- `manual_review_queue.csv`: file to review and correct manually
- `verified_labels.csv`: final manually verified label file used for training

The final `verified_labels.csv` file must include:

```text
Complaint ID
clean_text
triage_label
escalation_label
```
