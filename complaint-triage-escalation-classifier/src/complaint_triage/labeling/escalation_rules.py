"""Weak-label rules for escalation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.weak_labeler import _contains_phrase, _safe_lower


def escalation_score(row: pd.Series, schema: LabelSchema) -> tuple[int, list[str]]:
    """Compute a transparent escalation risk score and reasons."""
    spec = schema.escalation
    reasons: list[str] = []
    score = 0

    narrative_and_metadata = " | ".join(
        [
            _safe_lower(row.get("clean_text", "")),
            _safe_lower(row.get("Issue", "")),
            _safe_lower(row.get("Sub-issue", "")),
            _safe_lower(row.get("Tags", "")),
            _safe_lower(row.get("Company response to consumer", "")),
            _safe_lower(row.get("Company public response", "")),
            _safe_lower(row.get("Timely response?", "")),
        ]
    )

    for keyword in spec.get("high_risk_keywords", []):
        if _contains_phrase(narrative_and_metadata, keyword):
            score += 1
            reasons.append(f"keyword:{keyword}")

    issue_text = _safe_lower(row.get("Issue", ""))
    sub_issue_text = _safe_lower(row.get("Sub-issue", ""))
    for issue in spec.get("high_risk_issues", []):
        issue_l = issue.lower()
        if issue_l in issue_text or issue_l in sub_issue_text:
            score += 2
            reasons.append(f"issue:{issue}")

    response = _safe_lower(row.get("Company response to consumer", ""))
    for item in spec.get("high_risk_company_responses", []):
        if item.lower() in response:
            score += 1
            reasons.append(f"company_response:{item}")

    public_response = _safe_lower(row.get("Company public response", ""))
    for item in spec.get("high_risk_public_responses", []):
        if item.lower() in public_response:
            score += 1
            reasons.append(f"public_response:{item}")

    tags = _safe_lower(row.get("Tags", ""))
    for tag in spec.get("high_risk_tags", []):
        if tag.lower() in tags:
            score += 1
            reasons.append(f"tag:{tag}")

    timely = _safe_lower(row.get("Timely response?", ""))
    if timely == "no":
        score += 1
        reasons.append("timely_response:no")

    return score, reasons


def weak_escalation_label(row: pd.Series, schema: LabelSchema) -> tuple[str, int, str]:
    """Assign a candidate escalation label."""
    positive = schema.escalation.get("positive_label", "escalate")
    negative = schema.escalation.get("negative_label", "do_not_escalate")
    score, reasons = escalation_score(row, schema)

    label = positive if score >= 2 else negative
    return label, int(score), "; ".join(reasons) if reasons else "no_high_risk_rule_match"


def apply_weak_escalation_labels(df: pd.DataFrame, schema: LabelSchema) -> pd.DataFrame:
    """Add weak escalation labels to a DataFrame."""
    output = df.copy()
    results = output.apply(
        lambda row: weak_escalation_label(row, schema), axis=1, result_type="expand"
    )
    output["escalation_weak_label"] = results[0]
    output["escalation_score"] = results[1].astype(int)
    output["escalation_rule_reason"] = results[2]
    output["escalation_label"] = output["escalation_weak_label"]
    return output
