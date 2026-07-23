"""Group-conditional (Mondrian) split conformal utilities.

Mondrian conformal keeps the split-conformal quantile formula unchanged, but
calibrates one threshold per predefined group. In this project the groups are
the degradation cells ``(degradation_type, severity_level)``.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.conformal.split_conformal import (
    calibrate_split_conformal,
    compute_nonconformity_scores,
    empirical_coverage,
    predict_interval,
)
from src.utils.config import MEASUREMENT_DOMAINS, TARGET_COVERAGE


GroupKey = Tuple[str, float]


def degradation_group_key(row: dict) -> GroupKey:
    """Return the Mondrian group key for an Experiment 2 row."""
    return row["degradation"], float(row["severity"])


def is_finite_measurement_row(row: dict, measurement: str) -> bool:
    """Return whether prediction, reference, and absolute error are finite."""
    try:
        values = (
            float(row[f"pred_{measurement}"]),
            float(row[f"gt_{measurement}"]),
            float(row[f"abs_error_{measurement}"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def bounded_interval(measurement: str, point_prediction: float, q_hat: float) -> tuple:
    """Return a conformal interval intersected with the measurement domain."""
    lower, upper = predict_interval(point_prediction, q_hat)
    domain_min, domain_max = MEASUREMENT_DOMAINS[measurement]
    if domain_min is not None:
        lower = max(float(domain_min), lower)
    if domain_max is not None:
        upper = min(float(domain_max), upper)
    return lower, upper


def group_rows(rows: Iterable[dict]) -> Dict[GroupKey, List[dict]]:
    """Group rows by ``(degradation_type, severity_level)``."""
    grouped: Dict[GroupKey, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[degradation_group_key(row)].append(row)
    return dict(grouped)


def calibrate_mondrian_qhats(
    calibration_rows: Iterable[dict],
    measurement: str,
    alpha: float = 1.0 - TARGET_COVERAGE,
) -> Dict[GroupKey, dict]:
    """Calibrate one split-conformal q_hat per degradation/severity group."""
    qhats = {}
    for group, rows in sorted(group_rows(calibration_rows).items()):
        finite_rows = [row for row in rows if is_finite_measurement_row(row, measurement)]
        predicted = np.array([row[f"pred_{measurement}"] for row in finite_rows], dtype=float)
        reference = np.array([row[f"gt_{measurement}"] for row in finite_rows], dtype=float)
        scores = compute_nonconformity_scores(predicted, reference)
        qhats[group] = {
            "q_hat": calibrate_split_conformal(scores, alpha=alpha),
            "n_calibration_total": len(rows),
            "n_calibration_scores_finite": int(scores.size),
        }
    return qhats


def evaluate_group_coverage(
    test_rows: Iterable[dict],
    measurement: str,
    q_hat: float,
    low_n_threshold: int = 30,
) -> dict:
    """Evaluate coverage and mean interval width for one test group."""
    finite_rows = [row for row in test_rows if is_finite_measurement_row(row, measurement)]
    intervals = [
        bounded_interval(measurement, float(row[f"pred_{measurement}"]), q_hat)
        for row in finite_rows
    ]
    true_values = [float(row[f"gt_{measurement}"]) for row in finite_rows]
    coverage = empirical_coverage(intervals, true_values) if finite_rows else float("nan")
    widths = [upper - lower for lower, upper in intervals]
    return {
        "n_test": len(finite_rows),
        "empirical_coverage": coverage,
        "coverage_gap": coverage - TARGET_COVERAGE if np.isfinite(coverage) else float("nan"),
        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
        "low_n_warning": len(finite_rows) < low_n_threshold,
    }
