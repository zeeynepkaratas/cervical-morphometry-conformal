"""Exploratory post-hoc diagnostics for Cx22 measurement-specific coverage.

This analysis uses completed ShiftEval rows only. It neither retrains a model
nor changes the pre-frozen manifest, and must be reported as explanatory rather
than confirmatory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp5_cx22_stage_a import _global_q_hats
from src.utils.config import MEASUREMENTS, RESULTS_TABLES


def _covered(value: str) -> bool:
    return value.strip().lower() == "true"


def run_cx22_mechanism_diagnostics(
    rows_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_rows.csv",
    herlev_summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_path: Path = RESULTS_TABLES / "exp5_cx22_mechanism_diagnostics.json",
) -> dict:
    """Quantify the post-hoc error-buffer explanation for the pooled cohort."""
    rows = list(csv.DictReader(Path(rows_path).open(encoding="utf-8", newline="")))
    q_hats = _global_q_hats(herlev_summary_path)
    summary = {
        "analysis_type": "exploratory_post_hoc_mechanism_diagnostic",
        "n_rows": len(rows),
        "interpretation_guardrail": "Correlations and buffer ratios explain completed results; they were not used to select data, tune the model, or alter conformal calibration.",
        "measurements": {},
    }
    for measurement in MEASUREMENTS:
        finite = [row for row in rows if np.isfinite(float(row[f"abs_error_{measurement}"]))]
        errors = np.asarray([float(row[f"abs_error_{measurement}"]) for row in finite], dtype=float)
        nucleus_dice = np.asarray([float(row["nucleus_dice"]) for row in finite], dtype=float)
        covered = np.asarray([_covered(row[f"covered_{measurement}"]) for row in finite], dtype=bool)
        q_hat = float(q_hats[measurement])
        threshold_covered = errors <= q_hat
        pearson = pearsonr(nucleus_dice, errors)
        spearman = spearmanr(nucleus_dice, errors)
        summary["measurements"][measurement] = {
            "n_finite": len(finite),
            "q_hat": q_hat,
            "mean_absolute_error": float(errors.mean()),
            "median_absolute_error": float(np.median(errors)),
            "error_mean_to_median_ratio": float(errors.mean() / np.median(errors)),
            "q_hat_to_mean_error_buffer_ratio": float(q_hat / errors.mean()),
            "q_hat_to_median_error_buffer_ratio": float(q_hat / np.median(errors)),
            "mean_observed_interval_width": float(np.mean([float(row[f"interval_width_{measurement}"]) for row in finite])),
            "nucleus_dice_vs_absolute_error": {
                "pearson_r": float(pearson.statistic), "pearson_pvalue": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_pvalue": float(spearman.pvalue),
            },
            "coverage_by_nucleus_dice": {
                "mean_nucleus_dice_covered": float(nucleus_dice[covered].mean()),
                "mean_nucleus_dice_not_covered": float(nucleus_dice[~covered].mean()),
            },
            "q_hat_threshold_check": {
                "n_abs_error_exceeds_q_hat": int(np.count_nonzero(~threshold_covered)),
                "pct_abs_error_exceeds_q_hat": float(100.0 * np.mean(~threshold_covered)),
                "coverage_from_abs_error_threshold": float(np.mean(threshold_covered)),
                "observed_interval_coverage": float(np.mean(covered)),
                "n_disagreements_threshold_vs_interval": int(np.count_nonzero(threshold_covered != covered)),
                "interpretation": "Zero disagreements means domain clipping did not alter coverage relative to abs_error <= q_hat.",
            },
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_cx22_mechanism_diagnostics(), indent=2))
