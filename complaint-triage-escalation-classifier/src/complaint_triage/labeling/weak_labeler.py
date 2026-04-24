"""Weak-label bootstrapping for triage categories."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from complaint_triage.constants import TRIAGE_LABELS
from complaint_triage.labeling.schema import LabelSchema


TEXT_COLUMNS_FOR_RULES = [
    "clean_text",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company public response",
    "Company response to consumer",
    "Tags",
]


def _safe_lower(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    phrase = phrase.strip().lower()
    if not phrase:
        return False

    # Word-boundary matching for normal phrases, substring fallback for phrases
    # with punctuation such as "can't" or "FCRA".
    escaped = re.escape(phrase)
    if re.search(rf"(?<!\w){escaped}(?!\w)", haystack):
        return True
    return phrase in haystack


def _row_text(row: pd.Series) -> str:
    values = [_safe_lower(row.get(col, "")) for col in TEXT_COLUMNS_FOR_RULES]
    return " | ".join(values)


def score_triage_labels(row: pd.Series, schema: LabelSchema) -> dict[str, float]:
    """Score each triage label using keyword and metadata rules."""
    full_text = _row_text(row)
    product_text = _safe_lower(row.get("Product", ""))
    sub_product_text = _safe_lower(row.get("Sub-product", ""))
    issue_text = _safe_lower(row.get("Issue", ""))
    sub_issue_text = _safe_lower(row.get("Sub-issue", ""))

    scores: dict[str, float] = defaultdict(float)

    for label_id, spec in schema.triage_labels.items():
        for keyword in spec.get("keywords", []):
            if _contains_phrase(full_text, keyword):
                scores[label_id] += 2.0

        for product in spec.get("products", []):
            product_l = product.lower()
            if product_l == product_text or product_l == sub_product_text:
                scores[label_id] += 1.0
            elif product_l in product_text or product_l in sub_product_text:
                scores[label_id] += 0.5

        for issue in spec.get("issues", []):
            issue_l = issue.lower()
            if issue_l == issue_text or issue_l == sub_issue_text:
                scores[label_id] += 2.0
            elif issue_l in issue_text or issue_l in sub_issue_text:
                scores[label_id] += 1.0

    return {label_id: float(scores.get(label_id, 0.0)) for label_id in schema.triage_label_ids}


def confidence_from_scores(scores: dict[str, float]) -> float:
    """Convert rule scores into a transparent confidence heuristic."""
    if not scores:
        return 0.0

    ordered = sorted(scores.values(), reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0

    if best <= 0:
        return 0.0

    margin = best - second
    # Smoothly increase confidence with stronger evidence and larger margin.
    confidence = 1.0 / (1.0 + math.exp(-(0.65 * best + 0.8 * margin - 2.0)))
    return round(float(min(max(confidence, 0.0), 0.98)), 4)


def weak_triage_label(row: pd.Series, schema: LabelSchema) -> tuple[str, float, str]:
    """Assign one candidate triage label."""
    scores = score_triage_labels(row, schema)
    if not scores:
        return TRIAGE_LABELS[0], 0.0, "no_rule_match_default"

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        # Default to documentation because no rule triggered; this row should
        # be manually reviewed.
        return "documentation_processing_verification", 0.0, "no_rule_match_default"

    confidence = confidence_from_scores(scores)
    reason = "; ".join(
        f"{label}={score:g}" for label, score in sorted(scores.items(), key=lambda x: -x[1])[:3]
    )
    return best_label, confidence, reason


def apply_weak_triage_labels(df: pd.DataFrame, schema: LabelSchema) -> pd.DataFrame:
    """Add weak triage labels to a DataFrame."""
    output = df.copy()
    results = output.apply(lambda row: weak_triage_label(row, schema), axis=1, result_type="expand")
    output["triage_weak_label"] = results[0]
    output["triage_confidence"] = results[1].astype(float)
    output["triage_rule_reason"] = results[2]
    output["triage_label"] = output["triage_weak_label"]
    return output
