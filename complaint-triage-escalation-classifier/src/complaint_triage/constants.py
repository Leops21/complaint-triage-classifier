"""Project-wide constants."""

from __future__ import annotations

RAW_REQUIRED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

TRIAGE_LABELS = [
    "fraud_identity_risk",
    "billing_payment_dispute",
    "account_access_management",
    "documentation_processing_verification",
    "customer_service_failure",
    "legal_compliance_regulatory",
]

TRIAGE_LABEL_DISPLAY_NAMES = {
    "fraud_identity_risk": "Fraud / scam / identity risk",
    "billing_payment_dispute": "Billing / payment dispute",
    "account_access_management": "Account access / login / account management",
    "documentation_processing_verification": "Documentation / processing / verification issue",
    "customer_service_failure": "Customer service failure",
    "legal_compliance_regulatory": "Legal / compliance / regulatory concern",
}

ESCALATION_LABELS = ["do_not_escalate", "escalate"]

TASK_TO_LABEL_COLUMN = {
    "triage": "triage_label",
    "escalation": "escalation_label",
}

TASK_TO_LABELS = {
    "triage": TRIAGE_LABELS,
    "escalation": ESCALATION_LABELS,
}
