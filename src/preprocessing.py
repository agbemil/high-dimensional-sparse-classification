"""Data loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import resample


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_encoded: np.ndarray
    y_test_encoded: np.ndarray
    classes: np.ndarray


def balance_classes(
    df: pd.DataFrame,
    label_column: str,
    random_state: int = 123,
) -> pd.DataFrame:
    """Downsample every class to the size of the minority class."""
    class_counts = df[label_column].value_counts()
    min_class_count = int(class_counts.min())

    balanced_parts = []
    for class_label in class_counts.index:
        subset = df[df[label_column] == class_label]
        sampled = resample(
            subset,
            replace=False,
            n_samples=min_class_count,
            random_state=random_state,
        )
        balanced_parts.append(sampled)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with recent scikit-learn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "label",
    random_state: int = 123,
) -> PreparedData:
    """Balance training classes, encode labels, scale features, and add bias."""
    if label_column not in train_df.columns or label_column not in test_df.columns:
        raise ValueError(f"Both datasets must contain a '{label_column}' column.")

    train_balanced = balance_classes(train_df, label_column, random_state)

    y_train_raw = train_balanced[label_column].to_numpy()
    y_test_raw = test_df[label_column].to_numpy()

    X_train_df = train_balanced.drop(columns=[label_column])
    X_test_df = test_df.drop(columns=[label_column])

    if list(X_train_df.columns) != list(X_test_df.columns):
        raise ValueError(
            "Training and test predictors must have the same columns in the same order."
        )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    unknown_test = set(np.unique(y_test_raw)) - set(label_encoder.classes_)
    if unknown_test:
        raise ValueError(
            f"Test data contain classes not present in training data: {unknown_test}"
        )
    y_test = label_encoder.transform(y_test_raw)

    encoder = _make_one_hot_encoder()
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_encoded=y_train_encoded,
        y_test_encoded=y_test_encoded,
        classes=label_encoder.classes_,
    )
