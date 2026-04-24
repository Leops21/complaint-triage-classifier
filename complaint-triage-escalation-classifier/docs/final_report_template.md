# Final Report Template

## 1. Introduction

Explain the business problem: complaint triage is slow, inconsistent, and expensive when done manually.

## 2. Dataset

Describe the CFPB Consumer Complaint Database, the CSV fields used, and why only rows with public complaint narratives were retained.

## 3. Labeling schema

Explain the six custom triage labels and the binary escalation label. Mention that weak labels were used only as a bootstrap step and that the final training file was manually verified.

## 4. Preprocessing

Describe narrative filtering, deduplication by `Complaint ID`, whitespace normalization, redaction normalization, and creation of `clean_text`.

## 5. Models

Include:

- TF-IDF + Logistic Regression baseline
- DistilBERT classifier
- RoBERTa-base classifier

## 6. Evaluation metrics

For triage, report macro precision, macro recall, macro F1, weighted F1, and confusion matrix.

For escalation, report precision/recall/F1 for `escalate`, PR-AUC, ROC-AUC, threshold sweep, and false-negative analysis.

## 7. Results

Insert the experiment table generated from `reports/metrics`.

## 8. Error analysis

Discuss most confused class pairs, weakest recall classes, performance by product, performance by narrative length, and costly escalation false negatives.

## 9. Operational impact

Explain how the classifier could reduce manual triage effort, reduce misrouting, and prioritize high-risk complaints.

## 10. Limitations and future work

Discuss label noise, CFPB representativeness limits, threshold tuning, active learning, hierarchical classification, and human-in-the-loop review.
