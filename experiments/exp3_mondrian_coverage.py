"""Experiment 3: global versus Mondrian split conformal coverage.

This experiment reuses the already generated Experiment 2 calibration/test
rows. It does not run segmentation inference and does not touch train data.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.conformal.mondrian_conformal import (
    calibrate_mondrian_qhats,
    evaluate_group_coverage,
    group_rows,
)
from src.utils.config import MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


def _load_exp2_rows(rows_path: Path) -> tuple[List[dict], List[dict]]:
    rows_by_split = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    return rows_by_split["calibration"], rows_by_split["test"]


def _load_global_qhats(summary_path: Path) -> Dict[str, float]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return {row["measurement"]: float(row["q_hat"]) for row in summary["coverage"]}


def _comparison_row(
    method: str,
    measurement: str,
    degradation: str,
    severity: float,
    q_hat: float,
    n_calibration_total: int,
    n_calibration_scores_finite: int,
    test_rows: List[dict],
    low_n_threshold: int,
) -> dict:
    metrics = evaluate_group_coverage(test_rows, measurement, q_hat, low_n_threshold=low_n_threshold)
    return {
        "method": method,
        "measurement": measurement,
        "degradation_type": degradation,
        "severity_level": severity,
        "q_hat": q_hat,
        "n_calibration_total": n_calibration_total,
        "n_calibration_scores_finite": n_calibration_scores_finite,
        "n_test": metrics["n_test"],
        "empirical_coverage": metrics["empirical_coverage"],
        "coverage_gap": metrics["coverage_gap"],
        "mean_interval_width": metrics["mean_interval_width"],
        "low_n_warning": metrics["low_n_warning"],
    }


def build_global_vs_mondrian_rows(
    calibration_rows: List[dict],
    test_rows: List[dict],
    global_qhats: Dict[str, float],
    low_n_threshold: int = 30,
) -> List[dict]:
    """Return paired global/Mondrian coverage rows for each degradation cell."""
    calibration_groups = group_rows(calibration_rows)
    test_groups = group_rows(test_rows)
    output_rows = []

    for measurement in MEASUREMENTS:
        mondrian_qhats = calibrate_mondrian_qhats(calibration_rows, measurement)
        for group, group_test_rows in sorted(test_groups.items()):
            degradation, severity = group
            cal_info = mondrian_qhats[group]
            output_rows.append(
                _comparison_row(
                    "global",
                    measurement,
                    degradation,
                    severity,
                    global_qhats[measurement],
                    len(calibration_groups[group]),
                    cal_info["n_calibration_scores_finite"],
                    group_test_rows,
                    low_n_threshold,
                )
            )
            output_rows.append(
                _comparison_row(
                    "mondrian",
                    measurement,
                    degradation,
                    severity,
                    cal_info["q_hat"],
                    len(calibration_groups[group]),
                    cal_info["n_calibration_scores_finite"],
                    group_test_rows,
                    low_n_threshold,
                )
            )
    return output_rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row_lookup(rows: List[dict]) -> dict:
    return {
        (row["method"], row["measurement"], row["degradation_type"], float(row["severity_level"])): row
        for row in rows
    }


def _summarize_key_cells(rows: List[dict]) -> dict:
    lookup = _row_lookup(rows)
    key_cells = [
        ("circularity", "gaussian_noise", 30.0),
        ("circularity", "contrast_change", 1.5),
        ("nc_ratio", "contrast_change", 1.5),
    ]
    summary = []
    for measurement, degradation, severity in key_cells:
        global_row = lookup[("global", measurement, degradation, severity)]
        mondrian_row = lookup[("mondrian", measurement, degradation, severity)]
        summary.append(
            {
                "measurement": measurement,
                "degradation_type": degradation,
                "severity_level": severity,
                "global_coverage": global_row["empirical_coverage"],
                "mondrian_coverage": mondrian_row["empirical_coverage"],
                "coverage_delta": mondrian_row["empirical_coverage"] - global_row["empirical_coverage"],
                "global_mean_interval_width": global_row["mean_interval_width"],
                "mondrian_mean_interval_width": mondrian_row["mean_interval_width"],
                "width_delta": mondrian_row["mean_interval_width"] - global_row["mean_interval_width"],
            }
        )
    return {"key_cells": summary}


def run_experiment_3(
    rows_path: Path = RESULTS_TABLES / "exp2_marginal_rows.json",
    summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_path: Path = RESULTS_TABLES / "exp3_global_vs_mondrian.csv",
) -> dict:
    calibration_rows, test_rows = _load_exp2_rows(rows_path)
    global_qhats = _load_global_qhats(summary_path)
    comparison_rows = build_global_vs_mondrian_rows(calibration_rows, test_rows, global_qhats)
    _write_csv(output_path, comparison_rows)
    return {
        "output_path": str(output_path),
        "n_rows": len(comparison_rows),
        "target_coverage": TARGET_COVERAGE,
        **_summarize_key_cells(comparison_rows),
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment_3(), indent=2))
