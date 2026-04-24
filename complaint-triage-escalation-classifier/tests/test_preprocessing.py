import pandas as pd

from complaint_triage.preprocessing import normalize_text, prepare_complaints


def test_normalize_text_collapses_whitespace_and_redactions():
    text = "Hello\n\nXXXX    world!!!"
    result = normalize_text(text)
    assert result == "Hello [REDACTED] world!!"


def test_prepare_complaints_filters_empty_narratives():
    df = pd.DataFrame(
        {
            "Date received": ["2024-01-01", "2024-01-02"],
            "Product": ["Credit card", "Credit card"],
            "Sub-product": ["General-purpose credit card or charge card", "General-purpose credit card or charge card"],
            "Issue": ["Billing disputes", "Billing disputes"],
            "Sub-issue": ["", ""],
            "Consumer complaint narrative": ["This is a long enough complaint narrative about a billing issue.", ""],
            "Company public response": ["", ""],
            "Company": ["A", "B"],
            "State": ["CA", "CA"],
            "ZIP code": ["900", "900"],
            "Tags": ["", ""],
            "Consumer consent provided?": ["Consent provided", "Consent provided"],
            "Submitted via": ["Web", "Web"],
            "Date sent to company": ["2024-01-02", "2024-01-03"],
            "Company response to consumer": ["Closed with explanation", "Closed with explanation"],
            "Timely response?": ["Yes", "Yes"],
            "Consumer disputed?": ["", ""],
            "Complaint ID": ["1", "2"],
        }
    )
    out = prepare_complaints(df, min_text_chars=10)
    assert len(out) == 1
    assert out.loc[0, "Complaint ID"] == "1"
    assert "clean_text" in out.columns
