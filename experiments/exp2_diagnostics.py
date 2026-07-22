"""Diagnostic analyses for Experiment 2 marginal coverage outputs.

This script does not rerun inference and does not touch train/calibration/test
splits. It reads the existing ``exp2_marginal_rows.json`` and
``exp2_marginal_coverage_summary.json`` files and writes diagnostic tables.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp2_marginal_coverage import _bounded_interval
from src.conformal.split_conformal import empirical_coverage
from src.utils.config import MEASUREMENT_DOMAINS, MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


BONFERRONI_ALPHA = 0.05


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _score_is_finite(row: dict, measurement: str) -> bool:
    return (
        _is_finite(row.get(f"pred_{measurement}"))
        and _is_finite(row.get(f"gt_{measurement}"))
        and _is_finite(row.get(f"abs_error_{measurement}"))
    )


def _nonfinite_reason(row: dict, measurement: str) -> str:
    """Infer only reasons supported by stored Experiment 2 row fields."""
    if _score_is_finite(row, measurement):
        return "finite"
    if not _is_finite(row.get(f"gt_{measurement}")):
        return "gt_nan_upstream"
    if not _is_finite(row.get(f"pred_{measurement}")):
        if measurement == "circularity" and int(row.get("pred_nucleus_area", -1)) == 0:
            return "area_zero"
        if measurement == "nc_ratio" and int(row.get("pred_cytoplasm_area", -1)) == 0:
            return "cytoplasm_area_zero"
        return "nan_upstream"
    if not _is_finite(row.get(f"abs_error_{measurement}")):
        return "nan_upstream"
    return "unknown"


def _group_key(row: dict) -> tuple:
    return row["degradation"], float(row["severity"]), row["split"]


def build_nonfinite_breakdown(rows_by_split: Dict[str, List[dict]]) -> List[dict]:
    """Summarize finite/non-finite scores by measurement, degradation, severity, and split."""
    grouped: dict[tuple, list] = defaultdict(list)
    for split_rows in rows_by_split.values():
        for row in split_rows:
            for measurement in MEASUREMENTS:
                degradation, severity, split = _group_key(row)
                grouped[(measurement, degradation, severity, split)].append(row)

    output_rows = []
    for (measurement, degradation, severity, split), rows in sorted(grouped.items()):
        reasons = [_nonfinite_reason(row, measurement) for row in rows if not _score_is_finite(row, measurement)]
        reason_counts = defaultdict(int)
        for reason in reasons:
            reason_counts[reason] += 1
        if not reason_counts:
            nonfinite_reason = "none"
        else:
            nonfinite_reason = ";".join(f"{reason}:{count}" for reason, count in sorted(reason_counts.items()))
        output_rows.append(
            {
                "measurement": measurement,
                "degradation_type": degradation,
                "severity_level": severity,
                "split": split,
                "n_total": len(rows),
                "n_nonfinite": len(reasons),
                "nonfinite_reason": nonfinite_reason,
            }
        )
    return output_rows


def build_clipping_summary(test_rows: List[dict], coverage_summary: dict) -> dict:
    """Count test intervals that touch measurement support boundaries."""
    q_hat_by_measurement = {row["measurement"]: float(row["q_hat"]) for row in coverage_summary["coverage"]}
    summary = {}
    for measurement in MEASUREMENTS:
        domain_min, domain_max = MEASUREMENT_DOMAINS[measurement]
        finite_rows = [row for row in test_rows if _score_is_finite(row, measurement)]
        lower_clipped = 0
        upper_clipped = 0
        both_clipped = 0
        for row in finite_rows:
            lower, upper = _bounded_interval(measurement, float(row[f"pred_{measurement}"]), q_hat_by_measurement[measurement])
            clipped_lower = (
                domain_min is not None and math.isclose(lower, domain_min, rel_tol=0.0, abs_tol=1e-12)
            )
            clipped_upper = (
                domain_max is not None and math.isclose(upper, domain_max, rel_tol=0.0, abs_tol=1e-12)
            )
            lower_clipped += int(clipped_lower)
            upper_clipped += int(clipped_upper)
            both_clipped += int(clipped_lower and clipped_upper)
        clipped_either = lower_clipped + upper_clipped - both_clipped
        summary[measurement] = {
            "domain": [domain_min, domain_max],
            "n_test_variants_finite": len(finite_rows),
            "n_clipped_lower": lower_clipped,
            "n_clipped_upper": upper_clipped,
            "n_clipped_both": both_clipped,
            "pct_clipped_either": float(clipped_either / len(finite_rows)) if finite_rows else float("nan"),
        }
    return summary


def build_conditional_coverage(test_rows: List[dict], coverage_summary: dict, low_n_threshold: int = 30) -> List[dict]:
    """Compute empirical coverage separately for each degradation/severity cell."""
    q_hat_by_measurement = {row["measurement"]: float(row["q_hat"]) for row in coverage_summary["coverage"]}
    grouped: dict[tuple, list] = defaultdict(list)
    for row in test_rows:
        grouped[_group_key(row)].append(row)

    n_comparisons = len(MEASUREMENTS) * len(grouped)

    output_rows = []
    for measurement in MEASUREMENTS:
        for (degradation, severity, split), rows in sorted(grouped.items()):
            if split != "test":
                continue
            finite_rows = [row for row in rows if _score_is_finite(row, measurement)]
            intervals = [
                _bounded_interval(measurement, float(row[f"pred_{measurement}"]), q_hat_by_measurement[measurement])
                for row in finite_rows
            ]
            true_values = [float(row[f"gt_{measurement}"]) for row in finite_rows]
            coverage = empirical_coverage(intervals, true_values) if finite_rows else float("nan")
            if finite_rows and np.isfinite(coverage):
                standard_error = math.sqrt(TARGET_COVERAGE * (1.0 - TARGET_COVERAGE) / len(finite_rows))
                z_score = (coverage - TARGET_COVERAGE) / standard_error if standard_error > 0 else float("nan")
                p_value = math.erfc(abs(z_score) / math.sqrt(2.0)) if np.isfinite(z_score) else float("nan")
                bonferroni_pvalue = min(1.0, p_value * n_comparisons) if np.isfinite(p_value) else float("nan")
                bonferroni_significant = bool(bonferroni_pvalue < BONFERRONI_ALPHA)
            else:
                standard_error = float("nan")
                z_score = float("nan")
                p_value = float("nan")
                bonferroni_pvalue = float("nan")
                bonferroni_significant = False
            output_rows.append(
                {
                    "measurement": measurement,
                    "degradation_type": degradation,
                    "severity_level": severity,
                    "n_test": len(finite_rows),
                    "empirical_coverage": coverage,
                    "coverage_gap": coverage - TARGET_COVERAGE if np.isfinite(coverage) else float("nan"),
                    "standard_error_at_target": standard_error,
                    "z_score_vs_target": z_score,
                    "two_sided_pvalue_normal_approx": p_value,
                    "bonferroni_pvalue": bonferroni_pvalue,
                    "bonferroni_significant_0p05": bonferroni_significant,
                    "low_n_warning": len(finite_rows) < low_n_threshold,
                }
            )
    return output_rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(
    rows_path: Path = RESULTS_TABLES / "exp2_marginal_rows.json",
    summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_dir: Path = RESULTS_TABLES,
) -> dict:
    rows_by_split = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    coverage_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    calibration_rows = rows_by_split["calibration"]
    test_rows = rows_by_split["test"]

    nonfinite_breakdown = build_nonfinite_breakdown(rows_by_split)
    clipping_summary = build_clipping_summary(test_rows, coverage_summary)
    conditional_coverage = build_conditional_coverage(test_rows, coverage_summary)

    _write_csv(output_dir / "exp2_nonfinite_breakdown.csv", nonfinite_breakdown)
    _write_csv(output_dir / "exp2_conditional_coverage.csv", conditional_coverage)
    (output_dir / "exp2_clipping_summary.json").write_text(
        json.dumps(clipping_summary, indent=2),
        encoding="utf-8",
    )

    consistency = {
        "calibration_rows": len(calibration_rows),
        "test_rows": len(test_rows),
        "finite_counts": {
            split: {
                measurement: sum(_score_is_finite(row, measurement) for row in split_rows)
                for measurement in MEASUREMENTS
            }
            for split, split_rows in rows_by_split.items()
        },
        "nonfinite_breakdown_rows": len(nonfinite_breakdown),
        "conditional_coverage_rows": len(conditional_coverage),
    }
    return {
        "outputs": {
            "nonfinite_breakdown": str(output_dir / "exp2_nonfinite_breakdown.csv"),
            "clipping_summary": str(output_dir / "exp2_clipping_summary.json"),
            "conditional_coverage": str(output_dir / "exp2_conditional_coverage.csv"),
        },
        "consistency": consistency,
        "clipping_summary": clipping_summary,
    }


if __name__ == "__main__":
    print(json.dumps(run_diagnostics(), indent=2))
