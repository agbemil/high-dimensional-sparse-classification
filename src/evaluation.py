"""Evaluation and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .optimization import predict, predict_proba


def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-vs-rest sensitivity and specificity for each class."""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
    )

    sensitivity = np.zeros(n_classes, dtype=float)
    specificity = np.zeros(n_classes, dtype=float)

    total = cm.sum()

    for class_index in range(n_classes):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = total - tp - fn - fp

        sensitivity[class_index] = (
            tp / (tp + fn) if (tp + fn) > 0 else np.nan
        )
        specificity[class_index] = (
            tn / (tn + fp) if (tn + fp) > 0 else np.nan
        )

    return sensitivity, specificity


def sparsity_summary(
    weights: np.ndarray,
    exclude_bias: bool = True,
    zero_tolerance: float = 1e-12,
) -> tuple[int, int, float]:
    """Return nonzero count, total count, and zero percentage."""
    target = weights[1:, :] if exclude_bias else weights
    nonzero = int(np.sum(np.abs(target) > zero_tolerance))
    total = int(target.size)
    zero_pct = 100.0 * (1.0 - nonzero / total) if total else 0.0
    return nonzero, total, zero_pct


def evaluate_model(
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_encoded: np.ndarray,
    weights: np.ndarray,
    runtime_seconds: float,
    iterations: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate predictions and return a summary dictionary."""
    y_pred = predict(X_test, weights)
    probabilities = predict_proba(X_test, weights)

    n_classes = y_test_encoded.shape[1]
    sensitivity, specificity = sensitivity_specificity(
        y_test,
        y_pred,
        n_classes,
    )

    try:
        auc = roc_auc_score(
            y_test_encoded,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = np.nan

    nonzero, total, zero_pct = sparsity_summary(weights)

    summary = {
        "method": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "macro_roc_auc_ovr": auc,
        "runtime_seconds": runtime_seconds,
        "iterations": iterations,
        "nonzero_coefficients": nonzero,
        "total_coefficients": total,
        "zero_coefficients_percent": zero_pct,
        "mean_sensitivity": np.nanmean(sensitivity),
        "mean_specificity": np.nanmean(specificity),
    }

    return summary, y_pred, probabilities


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        ax=ax,
        cmap="Blues",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curves(
    y_true_encoded: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save one-vs-rest ROC curves."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for class_index, class_name in enumerate(classes):
        class_truth = y_true_encoded[:, class_index]
        if np.unique(class_truth).size < 2:
            continue

        fpr, tpr, _ = roc_curve(class_truth, probabilities[:, class_index])
        auc = roc_auc_score(class_truth, probabilities[:, class_index])
        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_convergence_comparison(
    fbs_history: list[float],
    subgradient_history: list[float],
    output_path: Path,
) -> None:
    """Save convergence curves for both optimization methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fbs_history, label="Forward-Backward Splitting")
    ax.plot(subgradient_history, label="Subgradient Descent")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regularized Objective")
    ax.set_title("Optimization Convergence Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_comparison_table(
    rows: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    """Save model-comparison metrics as CSV."""
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_path, index=False)
    return comparison
