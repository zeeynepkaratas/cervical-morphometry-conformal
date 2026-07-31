"""Evaluate strict Herlev joint-conformal thresholds on frozen Cx22 rows.

Cx22 is never used to estimate scales or thresholds. This is an exploratory
external stress test, not a finite-sample coverage guarantee for Cx22.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.conformal.joint_conformal import joint_covered
from src.utils.config import RESULTS_TABLES, TARGET_COVERAGE


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repeat_calibrations(strict_rows: list[dict]) -> list[dict]:
    calibrations = []
    for row in strict_rows:
        if row["analysis"] != "joint":
            continue
        calibrations.append(
            {
                "seed": int(row["seed"]),
                "q_hat_joint": float(row["q_hat_joint"]),
                "scales": {
                    "nc_ratio": float(row["scale_nc_ratio"]),
                    "circularity": float(row["scale_circularity"]),
                },
            }
        )
    if not calibrations:
        raise ValueError("No joint calibration rows were found.")
    return calibrations


def _coverage_row(cx22_rows: list[dict], calibration: dict, partition: str) -> dict:
    subset = cx22_rows if partition == "pooled" else [row for row in cx22_rows if row["source_partition"] == partition]
    errors = {
        "nc_ratio": np.asarray([row["abs_error_nc_ratio"] for row in subset], dtype=float),
        "circularity": np.asarray([row["abs_error_circularity"] for row in subset], dtype=float),
    }
    covered = joint_covered(errors, calibration)
    jointly_finite = np.logical_and.reduce([np.isfinite(values) for values in errors.values()])
    return {
        "seed": calibration["seed"],
        "partition": partition,
        "n_total": len(subset),
        "n_jointly_finite": int(jointly_finite.sum()),
        "joint_valid_prediction_rate": float(jointly_finite.mean()),
        "joint_finite_case_coverage": float(covered[jointly_finite].mean()) if np.any(jointly_finite) else float("nan"),
        "joint_failure_aware_coverage": float(covered.mean()),
    }


def _summary(rows: list[dict]) -> list[dict]:
    output = []
    for partition in sorted({row["partition"] for row in rows}):
        group = [row for row in rows if row["partition"] == partition]
        record = {"partition": partition, "n_repeats": len(group)}
        for key in ("joint_valid_prediction_rate", "joint_finite_case_coverage", "joint_failure_aware_coverage"):
            values = np.asarray([row[key] for row in group], dtype=float)
            record[f"mean_{key}"] = float(np.mean(values))
            record[f"min_{key}"] = float(np.min(values))
            record[f"max_{key}"] = float(np.max(values))
        output.append(record)
    return output


def run_cx22_joint_external(
    strict_repeats_path: Path = RESULTS_TABLES / "exp2_strict_cellwise_repeats.csv",
    cx22_rows_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_rows.csv",
    output_dir: Path = RESULTS_TABLES,
) -> dict:
    """Apply fixed strict-Herlev joint calibration to Cx22 rows."""
    calibrations = _repeat_calibrations(_read_csv(Path(strict_repeats_path)))
    cx22_rows = _read_csv(Path(cx22_rows_path))
    rows = [
        _coverage_row(cx22_rows, calibration, partition)
        for calibration in calibrations
        for partition in ("pooled", "Pair", "Multi-Train", "Multi-Test")
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "exp5_cx22_joint_external_repeats.csv"
    summary_path = output_dir / "exp5_cx22_joint_external_summary.json"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "analysis_scope": "Exploratory Cx22 external joint-coverage stress test using fixed strict Herlev calibration.",
        "interpretation_guardrail": "Cx22 does not contribute to calibration; this is not a formal Cx22 finite-sample guarantee.",
        "target_coverage": TARGET_COVERAGE,
        "n_cx22_images": len(cx22_rows),
        "summary_by_partition": _summary(rows),
        "repeat_rows_path": str(rows_path.relative_to(ROOT_DIR)),
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_cx22_joint_external(), indent=2))
