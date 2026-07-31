"""Joint split-conformal intervals for the two locked morphometric measures.

One score is calibrated per cell, not per correlated image variant. The score
is the maximum of the two residuals after calibration-only normalization.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.conformal.split_conformal import calibrate_split_conformal


def calibration_scales(errors_by_measurement: Mapping[str, np.ndarray], eps: float = 1e-8) -> dict[str, float]:
    """Return positive median-absolute-error scales from finite calibration errors."""
    scales: dict[str, float] = {}
    for measurement, errors in errors_by_measurement.items():
        finite = np.asarray(errors, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"No finite calibration errors for {measurement}.")
        scales[measurement] = max(float(np.median(finite)), float(eps))
    return scales


def joint_nonconformity_scores(
    errors_by_measurement: Mapping[str, np.ndarray], scales: Mapping[str, float]
) -> np.ndarray:
    """Return the per-cell maximum normalized score for jointly valid errors."""
    measurements = tuple(errors_by_measurement)
    if not measurements:
        raise ValueError("At least one measurement is required.")
    arrays = [np.asarray(errors_by_measurement[name], dtype=float) for name in measurements]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Joint error arrays must share the same shape.")
    normalized = []
    for name, errors in zip(measurements, arrays):
        scale = float(scales[name])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scale for {name} must be finite and positive.")
        normalized.append(errors / scale)
    return np.max(np.column_stack(normalized), axis=1)


def calibrate_joint_split_conformal(
    errors_by_measurement: Mapping[str, np.ndarray],
    alpha: float,
    scales: Mapping[str, float] | None = None,
) -> dict:
    """Calibrate a finite-sample joint threshold using jointly finite cells only.

    ``scales`` should come from data disjoint from calibration and test when a
    formal split-conformal interpretation is required. The fallback exists for
    exploratory diagnostics only.
    """
    arrays = {name: np.asarray(values, dtype=float) for name, values in errors_by_measurement.items()}
    if not arrays:
        raise ValueError("At least one measurement is required.")
    shape = next(iter(arrays.values())).shape
    if any(values.shape != shape for values in arrays.values()):
        raise ValueError("Joint error arrays must share the same shape.")
    valid = np.logical_and.reduce([np.isfinite(values) for values in arrays.values()])
    finite_errors = {name: values[valid] for name, values in arrays.items()}
    resolved_scales = dict(scales) if scales is not None else calibration_scales(finite_errors)
    scores = joint_nonconformity_scores(finite_errors, resolved_scales)
    return {
        "scales": resolved_scales,
        "q_hat_joint": calibrate_split_conformal(scores, alpha=alpha),
        "n_calibration_cells_total": int(valid.size),
        "n_calibration_cells_jointly_finite": int(valid.sum()),
    }


def joint_covered(errors_by_measurement: Mapping[str, np.ndarray], calibration: Mapping[str, object]) -> np.ndarray:
    """Return joint coverage flags; non-finite predictions count as uncovered."""
    arrays = {name: np.asarray(values, dtype=float) for name, values in errors_by_measurement.items()}
    scales = calibration["scales"]
    q_hat = float(calibration["q_hat_joint"])
    valid = np.logical_and.reduce([np.isfinite(values) for values in arrays.values()])
    covered = np.zeros(valid.shape, dtype=bool)
    if np.any(valid):
        scores = joint_nonconformity_scores({name: values[valid] for name, values in arrays.items()}, scales)
        covered[valid] = scores <= q_hat
    return covered
