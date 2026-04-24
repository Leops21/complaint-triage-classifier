from pathlib import Path

import pandas as pd

from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.weak_labeler import weak_triage_label
from complaint_triage.labeling.escalation_rules import weak_escalation_label


ROOT = Path(__file__).resolve().parents[1]


def make_schema():
    return LabelSchema.from_yaml(ROOT / "configs" / "label_schema.yaml")


def test_weak_labeler_detects_fraud():
    row = pd.Series(
        {
            "clean_text": "Someone opened a fake account in my name and this is identity theft.",
            "Product": "Credit reporting",
            "Sub-product": "Credit reporting",
            "Issue": "Fraud or scam",
            "Sub-issue": "",
            "Company public response": "",
            "Company response to consumer": "",
            "Tags": "",
        }
    )
    label, confidence, reason = weak_triage_label(row, make_schema())
    assert label == "fraud_identity_risk"
    assert confidence > 0.5
    assert "fraud_identity_risk" in reason


def test_escalation_detects_legal_threat():
    row = pd.Series(
        {
            "clean_text": "The collector threatened a lawsuit and legal action.",
            "Issue": "Took or threatened to take negative or legal action",
            "Sub-issue": "",
            "Tags": "",
            "Company response to consumer": "",
            "Company public response": "",
            "Timely response?": "Yes",
        }
    )
    label, score, reason = weak_escalation_label(row, make_schema())
    assert label == "escalate"
    assert score >= 2
    assert "legal action" in reason.lower() or "issue:" in reason.lower()
