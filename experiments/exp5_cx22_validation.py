"""Cx22 external-evaluation compatibility check.

This script deliberately stops before model inference. It checks whether Cx22
is usable for real external validation in this workspace:

1. official label archives are present,
2. MATLAB v7.3 labels can be read in Python,
3. sample nucleus/cytoplasm instance masks are binary,
4. generated image data are already available.

If generated images are missing, the correct outcome is a documented no-go:
either generate Cx22 images from LLPC source images first, or use the Herlev
heavy-degradation domain-shift fallback already produced in Experiments 2-4.
"""

import json
import sys
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_cx22 import validate_cx22_compatibility
from src.utils.config import DATA_RAW_CX22, MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


HEAVY_SEVERITY_BY_DEGRADATION = {
    "gaussian_blur": 3.0,
    "gaussian_noise": 30.0,
    "contrast_change": 1.5,
    "low_resolution": 0.25,
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_herlev_heavy_shift_summary(exp4_rows_path: Path) -> list[dict]:
    """
    Aggregate the heaviest degradation cells as the planned fallback
    domain-shift stress test.

    This reuses existing Experiment 4 per-cell rows; it does not rerun
    inference or touch train/calibration/test splits.
    """
    rows = _read_csv(exp4_rows_path)
    output = []
    for measurement in MEASUREMENTS:
        for method in ["global", "mondrian", "cqr"]:
            selected = [
                row
                for row in rows
                if row["measurement"] == measurement
                and row["method"] == method
                and row["degradation_type"] in HEAVY_SEVERITY_BY_DEGRADATION
                and float(row["severity_level"]) == HEAVY_SEVERITY_BY_DEGRADATION[row["degradation_type"]]
            ]
            n_total = sum(int(row["n_test"]) for row in selected)
            covered = sum(float(row["empirical_coverage"]) * int(row["n_test"]) for row in selected)
            width_total = sum(float(row["mean_interval_width"]) * int(row["n_test"]) for row in selected)
            coverage = covered / n_total if n_total else float("nan")
            mean_width = width_total / n_total if n_total else float("nan")
            output.append(
                {
                    "stress_test": "herlev_heavy_degradation_fallback",
                    "method": method,
                    "measurement": measurement,
                    "n_cells": len(selected),
                    "n_test_variants": n_total,
                    "empirical_coverage": coverage,
                    "coverage_gap": coverage - TARGET_COVERAGE,
                    "mean_interval_width": mean_width,
                    "included_cells": ";".join(
                        f"{row['degradation_type']}={row['severity_level']}" for row in selected
                    ),
                }
            )
    return output


def run_experiment_5(
    raw_dir: Path = DATA_RAW_CX22,
    output_path: Path = RESULTS_TABLES / "exp5_cx22_compatibility_summary.json",
    exp4_rows_path: Path = RESULTS_TABLES / "exp4_cqr_coverage.csv",
    fallback_output_path: Path = RESULTS_TABLES / "exp5_herlev_heavy_shift_summary.csv",
) -> dict:
    """Run the Cx22 compatibility check and write a JSON report."""
    report = validate_cx22_compatibility(raw_dir, herlev_reference_stats={})
    fallback_rows = build_herlev_heavy_shift_summary(exp4_rows_path)
    _write_csv(fallback_output_path, fallback_rows)
    report["analysis_scope"] = "Cx22 input-availability compatibility check supporting the pooled ShiftEval evaluation."
    report["external_validation_ready"] = bool(report["compatible"])
    report["fallback_domain_shift_summary_path"] = str(fallback_output_path)
    report["fallback_domain_shift_rows"] = len(fallback_rows)
    report["next_step"] = (
        "run_full_cx22_external_validation"
        if report["compatible"]
        else "do_not_run_full_cx22_validation_until_generated_images_are_available"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment_5(), indent=2))
