"""Basic tests for optimization utilities."""

import numpy as np

from src.optimization import soft_threshold, softmax


def test_softmax_rows_sum_to_one():
    scores = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]])
    probabilities = softmax(scores)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_soft_threshold():
    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    thresholded = soft_threshold(values, 1.0)
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(thresholded, expected)
