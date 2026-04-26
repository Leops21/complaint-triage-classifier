"""Evaluation utilities for trained transformer classifiers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from complaint_triage.analysis.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.modeling.datasets import dataframe_to_dataset
from complaint_triage.modeling.metrics import (
    binary_curve_metrics,
    confusion_matrix_df,
    make_classification_report,
    threshold_sweep,
)


def _trainer_processor_kwargs(tokenizer) -> dict:
    """Return the tokenizer/processor keyword supported by the installed Transformers version."""
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _load_label_maps(model_dir: str | Path, task: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load label maps from model output or defaults."""
    path = Path(model_dir) / "label_maps.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        return label2id, id2label

    labels = TASK_TO_LABELS[task]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def evaluate_transformer(
    task: str,
    model_dir: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """Evaluate a saved transformer model on a test CSV."""
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, low_memory=False)
    labels = TASK_TO_LABELS[task]
    label_column = TASK_TO_LABEL_COLUMN[task]
    label2id, id2label = _load_label_maps(model_path, task)

    dataset = dataframe_to_dataset(test_df, task=task, text_column=text_column)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 512)

    tokenized = dataset.map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        **_trainer_processor_kwargs(tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(tokenized)
    logits = predictions.predictions
    probabilities = softmax(logits, axis=1)
    pred_ids = np.argmax(probabilities, axis=1)

    y_true_labels = test_df[label_column].astype(str).to_numpy()
    y_pred_labels = np.array([id2label[int(idx)] for idx in pred_ids])

    report = make_classification_report(y_true_labels, y_pred_labels, labels=labels)
    with (output_path / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report).transpose().to_csv(output_path / "classification_report.csv")

    cm_df = confusion_matrix_df(y_true_labels, y_pred_labels, labels=labels)
    cm_df.to_csv(output_path / "confusion_matrix.csv")
    plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        output_path=output_path / "confusion_matrix.png",
    )

    pred_df = test_df.copy()
    pred_df[f"{task}_prediction"] = y_pred_labels
    pred_df[f"{task}_correct"] = pred_df[label_column].astype(str).eq(pred_df[f"{task}_prediction"])

    for idx, label in enumerate(labels):
        pred_df[f"prob_{label}"] = probabilities[:, idx]

    pred_df.to_csv(output_path / "predictions.csv", index=False)

    if task == "escalation":
        positive_id = label2id["escalate"]
        true_ids = np.array([label2id[label] for label in y_true_labels])
        positive_scores = probabilities[:, positive_id]

        curve_metrics = binary_curve_metrics(true_ids, positive_scores)
        with (output_path / "binary_curve_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(curve_metrics, f, indent=2)

        thresholds = threshold_sweep(true_ids, positive_scores)
        thresholds.to_csv(output_path / "threshold_sweep.csv", index=False)
        plot_pr_curve(true_ids, positive_scores, output_path / "pr_curve.png")
        plot_roc_curve(true_ids, positive_scores, output_path / "roc_curve.png")

    return pred_df
