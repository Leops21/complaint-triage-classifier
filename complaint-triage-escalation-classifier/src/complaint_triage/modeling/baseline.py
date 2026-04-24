"""TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from complaint_triage.analysis.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.modeling.metrics import (
    binary_curve_metrics,
    confusion_matrix_df,
    threshold_sweep,
)


def make_baseline_pipeline() -> Pipeline:
    """Create a solid classical NLP baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=-1,
                    solver="saga",
                    penalty="l2",
                ),
            ),
        ]
    )


def train_baseline(
    task: str,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    text_column: str = "clean_text",
) -> dict:
    """Train and evaluate the baseline model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    # Combine train and validation for the final baseline fit after using the
    # split during development. This is common for a baseline once hyperparams
    # are fixed.
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    labels = TASK_TO_LABELS[task]
    label_column = TASK_TO_LABEL_COLUMN[task]

    model = make_baseline_pipeline()
    model.fit(fit_df[text_column].fillna("").astype(str), fit_df[label_column].astype(str))

    y_true = test_df[label_column].astype(str).to_numpy()
    y_pred = model.predict(test_df[text_column].fillna("").astype(str))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_path / "classification_report.csv")
    with (output_path / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_df = confusion_matrix_df(y_true, y_pred, labels=labels)
    cm_df.to_csv(output_path / "confusion_matrix.csv")
    plot_confusion_matrix(y_true, y_pred, labels=labels, output_path=output_path / "confusion_matrix.png")

    pred_df = test_df.copy()
    pred_df[f"{task}_prediction"] = y_pred
    pred_df[f"{task}_correct"] = pred_df[label_column].astype(str).eq(pred_df[f"{task}_prediction"])

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probs = model.predict_proba(test_df[text_column].fillna("").astype(str))
        class_order = list(model.named_steps["classifier"].classes_)
        for idx, label in enumerate(class_order):
            pred_df[f"prob_{label}"] = probs[:, idx]

        if task == "escalation" and "escalate" in class_order:
            positive_idx = class_order.index("escalate")
            positive_scores = probs[:, positive_idx]
            label2id = {label: idx for idx, label in enumerate(labels)}
            true_ids = np.array([label2id[label] for label in y_true])
            curve_metrics = binary_curve_metrics(true_ids, positive_scores)
            with (output_path / "binary_curve_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(curve_metrics, f, indent=2)
            threshold_sweep(true_ids, positive_scores).to_csv(
                output_path / "threshold_sweep.csv",
                index=False,
            )
            plot_pr_curve(true_ids, positive_scores, output_path / "pr_curve.png")
            plot_roc_curve(true_ids, positive_scores, output_path / "roc_curve.png")

    pred_df.to_csv(output_path / "predictions.csv", index=False)
    joblib.dump(model, output_path / "model.joblib")
    return report
