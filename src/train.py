"""Run the sparse multiclass classification comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .evaluation import (
    evaluate_model,
    save_comparison_table,
    save_confusion_matrix,
    save_convergence_comparison,
    save_roc_curves,
)
from .optimization import (
    forward_backward_splitting,
    predict,
    subgradient_descent,
)
from .preprocessing import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def tune_fbs(
    X: np.ndarray,
    y_encoded: np.ndarray,
    y_labels: np.ndarray,
    lambdas: list[float],
    learning_rates: list[float],
    n_splits: int = 3,
    random_state: int = 42,
    max_iter: int = 300,
) -> dict:
    """Tune FBS hyperparameters with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_accuracy = -np.inf
    best_params = None

    for lambda_reg in lambdas:
        for learning_rate in learning_rates:
            fold_accuracies = []
            for train_index, valid_index in cv.split(X, y_labels):
                result = forward_backward_splitting(
                    X[train_index],
                    y_encoded[train_index],
                    lambda_reg=lambda_reg,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    tol=1e-6,
                )
                y_pred = predict(X[valid_index], result.weights)
                fold_accuracies.append(float(np.mean(y_pred == y_labels[valid_index])))

            mean_accuracy = float(np.mean(fold_accuracies))
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = {
                    "lambda_reg": lambda_reg,
                    "learning_rate": learning_rate,
                    "cv_accuracy": mean_accuracy,
                }

    return best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Forward-Backward Splitting and Subgradient Descent "
            "for L1-regularized multiclass logistic regression."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--label", type=str, default="label")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune FBS lambda and learning rate using 3-fold CV before fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    data = prepare_data(train_df, test_df, label_column=args.label, random_state=123)

    # Defaults reproduce the original project's chosen FBS settings.
    fbs_params = {"lambda_reg": 0.01, "learning_rate": 0.1, "cv_accuracy": np.nan}
    if args.tune:
        fbs_params = tune_fbs(
            data.X_train,
            data.y_train_encoded,
            data.y_train,
            lambdas=[0.001, 0.01, 0.1],
            learning_rates=[0.01, 0.05, 0.1],
        )

    print("FBS parameters:")
    print(fbs_params)

    fbs = forward_backward_splitting(
        data.X_train,
        data.y_train_encoded,
        lambda_reg=fbs_params["lambda_reg"],
        learning_rate=fbs_params["learning_rate"],
        max_iter=1000,
        tol=1e-6,
    )

    subgradient = subgradient_descent(
        data.X_train,
        data.y_train_encoded,
        lambda_reg=0.01,
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-6,
    )

    fbs_summary, fbs_pred, fbs_proba = evaluate_model(
        "Forward-Backward Splitting", data.X_test, data.y_test,
        data.y_test_encoded, fbs.weights, fbs.runtime_seconds, fbs.iterations,
    )
    sub_summary, sub_pred, sub_proba = evaluate_model(
        "Subgradient Descent", data.X_test, data.y_test,
        data.y_test_encoded, subgradient.weights,
        subgradient.runtime_seconds, subgradient.iterations,
    )

    comparison = save_comparison_table(
        [fbs_summary, sub_summary], RESULTS_DIR / "model_comparison.csv"
    )
    save_convergence_comparison(
        fbs.loss_history, subgradient.loss_history,
        RESULTS_DIR / "convergence_comparison.png",
    )
    save_roc_curves(
        data.y_test_encoded, fbs_proba, data.classes,
        "ROC Curves — Forward-Backward Splitting", RESULTS_DIR / "roc_fbs.png",
    )
    save_roc_curves(
        data.y_test_encoded, sub_proba, data.classes,
        "ROC Curves — Subgradient Descent", RESULTS_DIR / "roc_subgradient.png",
    )
    save_confusion_matrix(
        data.y_test, fbs_pred, data.classes,
        "Confusion Matrix — Forward-Backward Splitting",
        RESULTS_DIR / "confusion_matrix_fbs.png",
    )
    save_confusion_matrix(
        data.y_test, sub_pred, data.classes,
        "Confusion Matrix — Subgradient Descent",
        RESULTS_DIR / "confusion_matrix_subgradient.png",
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
