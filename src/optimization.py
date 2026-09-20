"""Optimization routines for sparse multiclass logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class OptimizationResult:
    weights: np.ndarray
    loss_history: list[float]
    iterations: int
    runtime_seconds: float


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(
    X: np.ndarray,
    y_encoded: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Mean multiclass cross-entropy loss."""
    probabilities = softmax(X @ weights)
    return float(
        -np.mean(
            np.sum(y_encoded * np.log(probabilities + epsilon), axis=1)
        )
    )


def l1_penalty(weights: np.ndarray, penalize_bias: bool = False) -> float:
    """L1 norm, optionally excluding the first row (bias/intercept)."""
    target = weights if penalize_bias else weights[1:, :]
    return float(np.sum(np.abs(target)))


def objective(
    X: np.ndarray,
    y_encoded: np.ndarray,
    weights: np.ndarray,
    lambda_reg: float,
    penalize_bias: bool = False,
) -> float:
    """L1-regularized multinomial logistic objective."""
    return (
        cross_entropy_loss(X, y_encoded, weights)
        + lambda_reg * l1_penalty(weights, penalize_bias)
    )


def logistic_gradient(
    X: np.ndarray,
    y_encoded: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Gradient of the smooth multinomial logistic loss."""
    probabilities = softmax(X @ weights)
    return X.T @ (probabilities - y_encoded) / X.shape[0]


def soft_threshold(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def forward_backward_splitting(
    X: np.ndarray,
    y_encoded: np.ndarray,
    lambda_reg: float,
    learning_rate: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    penalize_bias: bool = False,
) -> OptimizationResult:
    """Proximal-gradient optimization for L1-regularized logistic regression."""
    weights = np.zeros((X.shape[1], y_encoded.shape[1]), dtype=float)
    loss_history: list[float] = []

    start = perf_counter()

    for iteration in range(1, max_iter + 1):
        gradient = logistic_gradient(X, y_encoded, weights)
        tentative = weights - learning_rate * gradient

        new_weights = tentative.copy()
        if penalize_bias:
            new_weights = soft_threshold(tentative, learning_rate * lambda_reg)
        else:
            new_weights[1:, :] = soft_threshold(
                tentative[1:, :],
                learning_rate * lambda_reg,
            )

        current_loss = objective(
            X,
            y_encoded,
            new_weights,
            lambda_reg,
            penalize_bias,
        )
        loss_history.append(current_loss)

        # Keep the newest iterate before convergence is checked.
        weights = new_weights

        if (
            len(loss_history) > 1
            and abs(loss_history[-1] - loss_history[-2]) < tol
        ):
            break

    runtime = perf_counter() - start

    return OptimizationResult(
        weights=weights,
        loss_history=loss_history,
        iterations=iteration,
        runtime_seconds=runtime,
    )


def subgradient_descent(
    X: np.ndarray,
    y_encoded: np.ndarray,
    lambda_reg: float,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    penalize_bias: bool = False,
) -> OptimizationResult:
    """Subgradient descent for L1-regularized logistic regression."""
    weights = np.zeros((X.shape[1], y_encoded.shape[1]), dtype=float)
    loss_history: list[float] = []

    start = perf_counter()

    for iteration in range(1, max_iter + 1):
        grad = logistic_gradient(X, y_encoded, weights)

        penalty_subgrad = np.sign(weights)
        if not penalize_bias:
            penalty_subgrad[0, :] = 0.0

        weights = weights - learning_rate * (
            grad + lambda_reg * penalty_subgrad
        )

        current_loss = objective(
            X,
            y_encoded,
            weights,
            lambda_reg,
            penalize_bias,
        )
        loss_history.append(current_loss)

        if (
            len(loss_history) > 1
            and abs(loss_history[-1] - loss_history[-2]) < tol
        ):
            break

    runtime = perf_counter() - start

    return OptimizationResult(
        weights=weights,
        loss_history=loss_history,
        iterations=iteration,
        runtime_seconds=runtime,
    )


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return multiclass class probabilities."""
    return softmax(X @ weights)


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return predicted class indices."""
    return np.argmax(predict_proba(X, weights), axis=1)
