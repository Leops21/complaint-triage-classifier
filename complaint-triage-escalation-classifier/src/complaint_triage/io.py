"""I/O utilities for CFPB complaint data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from complaint_triage.constants import RAW_REQUIRED_COLUMNS


def read_complaints_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the CFPB complaints CSV robustly.

    The public CFPB export can be large, so this function keeps types flexible
    and tries common encodings.
    """
    path = Path(path)
    errors: list[str] = []

    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                path,
                on_bad_lines="skip",
                nrows=nrows,
                encoding=encoding,
                low_memory=False,
                dtype={"ZIP code": "string", "Complaint ID": "string"},
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Unable to read {path}. Encoding attempts failed: {errors}",
    )


def validate_columns(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    """Validate that required CFPB columns exist."""
    required_cols = list(required or RAW_REQUIRED_COLUMNS)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV with parent directory creation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Create a compact profile of a complaint DataFrame."""
    profile: dict = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
    }

    for col in ["Product", "Issue", "Company response to consumer", "Timely response?"]:
        if col in df.columns:
            profile[f"top_{col}"] = df[col].fillna("<MISSING>").value_counts().head(20).to_dict()

    if "Consumer complaint narrative" in df.columns:
        narrative = df["Consumer complaint narrative"].fillna("").astype(str)
        profile["non_empty_narratives"] = int(narrative.str.strip().ne("").sum())
        profile["narrative_char_len"] = {
            "mean": float(narrative.str.len().mean()),
            "median": float(narrative.str.len().median()),
            "p95": float(narrative.str.len().quantile(0.95)),
        }

    return profile
