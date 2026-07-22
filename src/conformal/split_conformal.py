"""
Split conformal prediction baseline for the mandatory core.

For each scalar morphometric measurement, calibration scores are absolute
errors between model-derived point predictions and ground-truth references.
The resulting interval is a fixed-width marginal interval around the point
prediction.
"""

import math

import numpy as np

from src.utils.config import TARGET_COVERAGE


def compute_nonconformity_scores(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return finite absolute-error scores ``|predicted - reference|``."""
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape:
        raise ValueError(f"predicted and reference must have same shape: {predicted.shape} != {reference.shape}")
    scores = np.abs(predicted - reference)
    return scores[np.isfinite(scores)]


def calibrate_split_conformal(cal_scores: np.ndarray, alpha: float = 1 - TARGET_COVERAGE) -> float:
    """
    Calibrate the split conformal half-width.

    Uses the standard finite-sample conformal quantile:
        k = ceil((n + 1) * (1 - alpha))

    If k exceeds n, the maximum calibration score is used. This is the usual
    conservative fallback when the requested coverage is high for the available
    calibration size.
    """
    scores = np.sort(np.asarray(cal_scores, dtype=float))
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("cal_scores must contain at least one finite score.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(scores[k - 1])


def predict_interval(point_prediction: float, q_hat: float) -> tuple:
    """Return the fixed-width interval around ``point_prediction``."""
    point_prediction = float(point_prediction)
    q_hat = float(q_hat)
    if not np.isfinite(point_prediction):
        raise ValueError(f"point_prediction must be finite, got {point_prediction}")
    if not np.isfinite(q_hat) or q_hat < 0:
        raise ValueError(f"q_hat must be finite and non-negative, got {q_hat}")
    return point_prediction - q_hat, point_prediction + q_hat


def empirical_coverage(intervals: list, true_values: list) -> float:
    """Return the fraction of true values covered by their intervals."""
    if len(intervals) != len(true_values):
        raise ValueError(f"intervals and true_values lengths differ: {len(intervals)} != {len(true_values)}")
    if not intervals:
        raise ValueError("intervals must be non-empty.")

    covered = []
    for interval, true_value in zip(intervals, true_values):
        lower, upper = interval
        true_value = float(true_value)
        if not np.isfinite(true_value):
            continue
        covered.append(float(lower) <= true_value <= float(upper))
    if not covered:
        raise ValueError("No finite true values were available for coverage.")
    return float(np.mean(covered))
