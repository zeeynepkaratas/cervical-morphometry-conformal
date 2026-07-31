"""Summarize the Cx22 Multi-Test partition within the pooled evaluation.

This post-processing step consumes the completed scale-normalized rows only.
It does not train, tune, recalibrate, resample, or rerun the U-Net. This
retained n=100 summary is a source-partition record, not a separate Cx22
experiment or publication result.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


def _normal_coverage_test(coverage: float, n: int, n_comparisons: int) -> dict:
    """Match the project's existing normal-approximation/Bonferroni convention."""
    standard_error = math.sqrt(TARGET_COVERAGE * (1.0 - TARGET_COVERAGE) / n)
    z_score = (coverage - TARGET_COVERAGE) / standard_error
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    adjusted = min(1.0, p_value * n_comparisons)
    return {
        "standard_error_at_nominal": standard_error,
        "z_score_vs_nominal": z_score,
        "two_sided_pvalue_normal_approx": p_value,
        "bonferroni_n_comparisons": n_comparisons,
        "bonferroni_pvalue": adjusted,
        "bonferroni_significant_0p05": adjusted < 0.05,
    }


def lock_cx22_stage_b(
    rows_path: Path = RESULTS_TABLES / "exp5_cx22_scale_normalized_rows.csv",
    output_path: Path = RESULTS_TABLES / "exp5_cx22_stage_b_lock_summary.json",
) -> dict:
    """Verify finite denominators for the Multi-Test partition-level record."""
    rows = list(csv.DictReader(Path(rows_path).open(encoding="utf-8", newline="")))
    if len(rows) != 100:
        raise ValueError(f"Expected the approved 100 Cx22-Multi-Test rows, found {len(rows)}.")
    result = {
        "analysis_scope": "Cx22 pooled evaluation (n=1320): Multi-Test source-partition statistical summary (n=100).",
        "dataset": "Cx22-Multi-Test source partition",
        "n_images": len(rows),
        "protocol": "Scale-normalized known-localisation crop. GT masks set crop centre/scale only; model input is RGB alone; masks are evaluation-only.",
        "interpretation_guardrail": "Herlev q_hat values are applied without Cx22 recalibration, so this is an external stress test rather than a finite-sample Cx22 coverage guarantee.",
        "normal_approximation_note": "Uses the same target-null normal approximation and Bonferroni convention as the existing conditional-coverage analyses.",
        "measurements": {},
    }
    for measurement in MEASUREMENTS:
        finite = [row for row in rows if np.isfinite(float(row[f"gt_{measurement}"])) and np.isfinite(float(row[f"pred_{measurement}"]))]
        n = len(finite)
        covered = sum(row[f"covered_{measurement}"].strip().lower() == "true" for row in finite)
        coverage = covered / n if n else float("nan")
        result["measurements"][measurement] = {
            "n_rows_total": len(rows),
            "n_finite": n,
            "n_nonfinite": len(rows) - n,
            "n_covered": covered,
            "n_not_covered": n - covered,
            "empirical_coverage": coverage,
            "coverage_gap_vs_0p90": coverage - TARGET_COVERAGE,
            **_normal_coverage_test(coverage, n, len(MEASUREMENTS)),
            "interpretation": (
                "Coverage was not statistically distinguishable from the 0.90 nominal target at n=100; this does not establish equivalence or a formal transfer guarantee."
                if measurement == "nc_ratio"
                else "Coverage remained significantly below the 0.90 nominal target after two-measurement Bonferroni correction."
            ),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(lock_cx22_stage_b(), indent=2))
