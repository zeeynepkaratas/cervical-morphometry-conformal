"""Interval efficiency diagnostics, independent of model training packages."""

import numpy as np

from src.conformal.split_conformal import predict_interval
from src.utils.config import MEASUREMENTS, MEASUREMENT_DOMAINS


def joint_interval_efficiency(rows: list[dict], calibration: dict) -> dict:
    """Summarize domain-clipped widths on jointly finite evaluation rows.

    Failed predictions never receive a zero width. They are omitted here and
    remain failures in the separate failure-aware coverage calculation.
    """
    valid_rows = [row for row in rows if all(
        np.isfinite(float(row[f"{prefix}_{measurement}"]))
        for measurement in MEASUREMENTS for prefix in ("pred", "gt")
    )]
    result = {"n_joint_interval_rows": len(valid_rows)}
    widths_by_measurement = []
    for measurement in MEASUREMENTS:
        half_width = float(calibration["q_hat_joint"]) * float(calibration["scales"][measurement])
        domain_min, domain_max = MEASUREMENT_DOMAINS[measurement]
        widths = []
        full_domain = []
        for row in valid_rows:
            lower, upper = predict_interval(row[f"pred_{measurement}"], half_width)
            if domain_min is not None:
                lower = max(domain_min, lower)
            if domain_max is not None:
                upper = min(domain_max, upper)
            widths.append(max(0.0, upper - lower))
            if domain_min is not None and domain_max is not None:
                full_domain.append(lower <= domain_min and upper >= domain_max)
        array = np.asarray(widths)
        widths_by_measurement.append(array)
        result[f"joint_mean_width_{measurement}"] = float(array.mean()) if array.size else None
        result[f"joint_median_width_{measurement}"] = float(np.median(array)) if array.size else None
        if domain_min is not None and domain_max is not None:
            result[f"joint_full_domain_rate_{measurement}"] = float(np.mean(full_domain)) if full_domain else None
    result["joint_mean_rectangle_area"] = (
        float(np.prod(np.column_stack(widths_by_measurement), axis=1).mean()) if valid_rows else None
    )
    return result
