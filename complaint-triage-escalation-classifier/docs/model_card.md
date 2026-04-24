# Model Card: Customer Complaint Triage and Escalation Classifier

## Model overview

This project trains transformer-based sequence classifiers for consumer complaint narratives.

## Intended use

The model is intended to support operational triage by recommending:

1. a complaint routing category,
2. whether a complaint should be escalated for urgent human review.

It is not intended to make legal decisions, deny services, or replace human review for high-risk complaints.

## Input

A free-text consumer complaint narrative.

## Output

Task A:

- `fraud_identity_risk`
- `billing_payment_dispute`
- `account_access_management`
- `documentation_processing_verification`
- `customer_service_failure`
- `legal_compliance_regulatory`

Task B:

- `escalate`
- `do_not_escalate`

## Data

The model is designed for the CFPB Consumer Complaint Database CSV export. Only public narratives are used, and raw data is not stored in the repository.

## Limitations

- Labels are custom operational categories and require manual review for high-quality ground truth.
- CFPB narratives are not representative of all consumers or all companies.
- The model may learn artifacts from CFPB product/issue distributions.
- The escalation label is especially sensitive to false negatives.
- The model should be monitored for drift if complaint patterns change.

## Human oversight

Escalated complaints should always be reviewed by a trained specialist. Model predictions should be used as decision support, not as final decisions.

## Evaluation

The final report should include:

- classification reports,
- confusion matrices,
- training and validation curves,
- error-analysis tables,
- escalation false negative review,
- threshold tuning for escalation.
