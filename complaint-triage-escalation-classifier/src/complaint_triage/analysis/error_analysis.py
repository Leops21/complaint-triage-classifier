"""Error-analysis utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_prediction_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col.endswith("_prediction")]
    if not candidates:
        raise ValueError("No prediction column found. Expected a column ending with '_prediction'.")
    return candidates[0]


def _find_label_column(df: pd.DataFrame, prediction_column: str) -> str:
    task = prediction_column.replace("_prediction", "")
    label_col = f"{task}_label"
    if label_col in df.columns:
        return label_col
    if task == "triage" and "triage_label" in df.columns:
        return "triage_label"
    if task == "escalation" and "escalation_label" in df.columns:
        return "escalation_label"
    raise ValueError("Could not infer gold label column.")


def run_error_analysis(predictions_path: str | Path, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Create error-analysis CSV outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path, low_memory=False)
    pred_col = _find_prediction_column(df)
    label_col = _find_label_column(df, pred_col)

    df["is_correct"] = df[label_col].astype(str).eq(df[pred_col].astype(str))
    errors = df[~df["is_correct"]].copy()

    if "clean_text" in errors.columns:
        errors["text_char_len"] = errors["clean_text"].fillna("").astype(str).str.len()
        errors["text_word_len"] = errors["clean_text"].fillna("").astype(str).str.split().map(len)

    errors.to_csv(output_path / "all_errors.csv", index=False)

    confused_pairs = (
        errors.groupby([label_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confused_pairs.to_csv(output_path / "most_confused_pairs.csv", index=False)

    if "Product" in df.columns:
        by_product = (
            df.groupby("Product")["is_correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .sort_values(["accuracy", "count"], ascending=[True, False])
            .reset_index()
        )
        by_product.to_csv(output_path / "accuracy_by_product.csv", index=False)
    else:
        by_product = pd.DataFrame()

    if "text_word_len" in df.columns:
        working = df.copy()
    elif "clean_text" in df.columns:
        working = df.copy()
        working["text_word_len"] = working["clean_text"].fillna("").astype(str).str.split().map(len)
    else:
        working = pd.DataFrame()

    if not working.empty:
        working["length_bucket"] = pd.cut(
            working["text_word_len"],
            bins=[0, 50, 100, 200, 400, 10_000],
            labels=["0-50", "51-100", "101-200", "201-400", "401+"],
            include_lowest=True,
        )
        by_length = (
            working.groupby("length_bucket", observed=False)["is_correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .reset_index()
        )
        by_length.to_csv(output_path / "accuracy_by_length_bucket.csv", index=False)
    else:
        by_length = pd.DataFrame()

    return {
        "errors": errors,
        "confused_pairs": confused_pairs,
        "by_product": by_product,
        "by_length": by_length,
    }
