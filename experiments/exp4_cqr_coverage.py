"""Experiment 4: CQR efficiency versus global and Mondrian intervals.

CQR is evaluated after Mondrian split conformal has removed statistically
detectable conditional undercoverage. The question here is interval efficiency:
can adaptive CQR intervals preserve coverage while narrowing widths?
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp2_marginal_coverage import _collect_rows, _load_model
from src.conformal.cqr import conformalize_cqr, cqr_intervals, train_quantile_regressor
from src.conformal.mondrian_conformal import (
    bounded_interval,
    calibrate_mondrian_qhats,
    group_rows,
    is_finite_measurement_row,
)
from src.conformal.split_conformal import empirical_coverage
from src.data_prep.load_herlev import list_herlev_images
from src.utils.config import (
    CQR_QR_VAL_FRACTION,
    DATA_RAW_HERLEV,
    DEGRADATION_SEVERITY_LEVELS,
    MEASUREMENTS,
    RANDOM_SEED,
    RESULTS_TABLES,
    TARGET_COVERAGE,
)


KEY_CELLS = [
    ("circularity", "gaussian_noise", 30.0),
    ("circularity", "contrast_change", 1.5),
    ("nc_ratio", "contrast_change", 1.5),
]
BONFERRONI_ALPHA = 0.05


def _load_exp2_rows(rows_path: Path) -> tuple[List[dict], List[dict]]:
    rows_by_split = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    return rows_by_split["calibration"], rows_by_split["test"]


def _load_global_qhats(summary_path: Path) -> Dict[str, dict]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return {row["measurement"]: row for row in summary["coverage"]}


def _derive_qr_split(split: dict, seed: int = RANDOM_SEED) -> dict:
    """Derive QR train/validation IDs only from train_full."""
    train_full = list(split["train_full"])
    rng = np.random.default_rng(seed)
    shuffled = train_full.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * CQR_QR_VAL_FRACTION)))
    qr_val = sorted(shuffled[:n_val])
    qr_train = sorted(shuffled[n_val:])
    return {"qr_train": qr_train, "qr_val": qr_val}


def _split_overlap_report(split: dict, qr_split: dict) -> dict:
    keys = ["train_full", "calibration", "test", "qr_train", "qr_val"]
    combined = {**split, **qr_split}
    overlaps = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(set(combined[left]) & set(combined[right]))
    return overlaps


def _collect_qr_rows(
    split_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    device: str | None,
) -> tuple[List[dict], List[dict], dict]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    qr_split = _derive_qr_split(split)
    image_by_stem = {path.stem: path for path in list_herlev_images(raw_dir)}
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(checkpoint_path, device_obj)
    qr_train_rows = _collect_rows("qr_train", qr_split["qr_train"], image_by_stem, model, device_obj)
    qr_val_rows = _collect_rows("qr_val", qr_split["qr_val"], image_by_stem, model, device_obj)
    return qr_train_rows, qr_val_rows, {
        "qr_train_clean_images": len(qr_split["qr_train"]),
        "qr_val_clean_images": len(qr_split["qr_val"]),
        "qr_train_variants": len(qr_train_rows),
        "qr_val_variants": len(qr_val_rows),
        "overlaps": _split_overlap_report(split, qr_split),
    }


def _coverage_stats(intervals: List[tuple], references: List[float]) -> dict:
    widths = [upper - lower for lower, upper in intervals]
    coverage = empirical_coverage(intervals, references) if intervals else float("nan")
    return {
        "n_test": len(references),
        "empirical_coverage": coverage,
        "coverage_gap": coverage - TARGET_COVERAGE if np.isfinite(coverage) else float("nan"),
        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
        "median_interval_width": float(np.median(widths)) if widths else float("nan"),
    }


def _significance_fields(coverage: float, n_test: int, n_comparisons: int | None) -> dict:
    if n_test == 0 or not np.isfinite(coverage):
        return {
            "standard_error_at_target": float("nan"),
            "z_score_vs_target": float("nan"),
            "two_sided_pvalue_normal_approx": float("nan"),
            "bonferroni_pvalue": float("nan"),
            "bonferroni_significant_0p05": False,
        }
    standard_error = math.sqrt(TARGET_COVERAGE * (1.0 - TARGET_COVERAGE) / n_test)
    z_score = (coverage - TARGET_COVERAGE) / standard_error if standard_error > 0 else float("nan")
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0)) if np.isfinite(z_score) else float("nan")
    bonferroni_pvalue = min(1.0, p_value * n_comparisons) if n_comparisons else float("nan")
    return {
        "standard_error_at_target": standard_error,
        "z_score_vs_target": z_score,
        "two_sided_pvalue_normal_approx": p_value,
        "bonferroni_pvalue": bonferroni_pvalue,
        "bonferroni_significant_0p05": bool(np.isfinite(bonferroni_pvalue) and bonferroni_pvalue < BONFERRONI_ALPHA),
    }


def _evaluate_point_centered(
    method: str,
    measurement: str,
    rows: List[dict],
    q_hat: float,
    n_calibration_total: int,
    n_calibration_scores_finite: int,
    degradation: str,
    severity: float | str,
    n_comparisons: int | None,
) -> dict:
    finite_rows = [row for row in rows if is_finite_measurement_row(row, measurement)]
    intervals = [bounded_interval(measurement, float(row[f"pred_{measurement}"]), q_hat) for row in finite_rows]
    references = [float(row[f"gt_{measurement}"]) for row in finite_rows]
    stats = _coverage_stats(intervals, references)
    return {
        "method": method,
        "measurement": measurement,
        "degradation_type": degradation,
        "severity_level": severity,
        "q_hat": q_hat,
        "n_calibration_total": n_calibration_total,
        "n_calibration_scores_finite": n_calibration_scores_finite,
        **stats,
        **_significance_fields(stats["empirical_coverage"], stats["n_test"], n_comparisons),
        "width_ratio_vs_mondrian": float("nan"),
        "low_n_warning": stats["n_test"] < 30,
    }


def _evaluate_mondrian_marginal(
    measurement: str,
    test_groups: Dict[tuple, List[dict]],
    mondrian_qhats: dict,
) -> dict:
    intervals = []
    references = []
    n_test = 0
    for group, group_test_rows in sorted(test_groups.items()):
        q_hat = mondrian_qhats[group]["q_hat"]
        finite_rows = [row for row in group_test_rows if is_finite_measurement_row(row, measurement)]
        intervals.extend(
            [bounded_interval(measurement, float(row[f"pred_{measurement}"]), q_hat) for row in finite_rows]
        )
        references.extend([float(row[f"gt_{measurement}"]) for row in finite_rows])
        n_test += len(finite_rows)
    stats = _coverage_stats(intervals, references)
    return {
        "method": "mondrian",
        "measurement": measurement,
        "degradation_type": "all",
        "severity_level": "all",
        "q_hat": float("nan"),
        "n_calibration_total": sum(row["n_calibration_total"] for row in mondrian_qhats.values()),
        "n_calibration_scores_finite": sum(row["n_calibration_scores_finite"] for row in mondrian_qhats.values()),
        **stats,
        **_significance_fields(stats["empirical_coverage"], n_test, None),
        "width_ratio_vs_mondrian": 1.0,
        "low_n_warning": stats["n_test"] < 30,
    }


def _evaluate_cqr(
    measurement: str,
    rows: List[dict],
    cqr_model,
    n_calibration_total: int,
    n_calibration_scores_finite: int,
    degradation: str,
    severity: float | str,
    n_comparisons: int | None,
) -> dict:
    intervals, references, _ = cqr_intervals(cqr_model, rows)
    stats = _coverage_stats(intervals, references)
    return {
        "method": "cqr",
        "measurement": measurement,
        "degradation_type": degradation,
        "severity_level": severity,
        "q_hat": cqr_model.q_hat,
        "n_calibration_total": n_calibration_total,
        "n_calibration_scores_finite": n_calibration_scores_finite,
        **stats,
        **_significance_fields(stats["empirical_coverage"], stats["n_test"], n_comparisons),
        "width_ratio_vs_mondrian": float("nan"),
        "low_n_warning": stats["n_test"] < 30,
    }


def _set_width_ratios(rows: List[dict]) -> None:
    lookup = {
        (row["method"], row["measurement"], row["degradation_type"], row["severity_level"]): row
        for row in rows
    }
    for row in rows:
        if row["method"] != "cqr":
            continue
        key = ("mondrian", row["measurement"], row["degradation_type"], row["severity_level"])
        mondrian_row = lookup.get(key)
        if mondrian_row and float(mondrian_row["mean_interval_width"]) > 0:
            row["width_ratio_vs_mondrian"] = row["mean_interval_width"] / mondrian_row["mean_interval_width"]


def build_exp4_rows(
    calibration_rows: List[dict],
    test_rows: List[dict],
    qr_train_rows: List[dict],
    qr_val_rows: List[dict],
    global_qhats: Dict[str, dict],
) -> tuple[List[dict], dict]:
    calibration_groups = group_rows(calibration_rows)
    test_groups = group_rows(test_rows)
    n_conditional_comparisons = len(MEASUREMENTS) * len(test_groups)
    output_rows = []
    cqr_training = {}

    for measurement in MEASUREMENTS:
        cqr_model = train_quantile_regressor(qr_train_rows, qr_val_rows, measurement)
        cqr_model = conformalize_cqr(cqr_model, calibration_rows)
        cqr_training[measurement] = {
            "train_loss": cqr_model.train_loss,
            "val_loss": cqr_model.val_loss,
            "epochs_trained": cqr_model.epochs_trained,
            "q_hat": cqr_model.q_hat,
        }
        mondrian_qhats = calibrate_mondrian_qhats(calibration_rows, measurement)
        global_info = global_qhats[measurement]

        output_rows.append(
            _evaluate_point_centered(
                "global",
                measurement,
                test_rows,
                float(global_info["q_hat"]),
                int(global_info["n_calibration_variants_total"]),
                int(global_info["n_calibration_scores_finite"]),
                "all",
                "all",
                None,
            )
        )
        output_rows.append(_evaluate_mondrian_marginal(measurement, test_groups, mondrian_qhats))
        output_rows.append(
            _evaluate_cqr(
                measurement,
                test_rows,
                cqr_model,
                len(calibration_rows),
                sum(is_finite_measurement_row(row, measurement) for row in calibration_rows),
                "all",
                "all",
                None,
            )
        )

        for group, group_test_rows in sorted(test_groups.items()):
            degradation, severity = group
            cal_info = mondrian_qhats[group]
            output_rows.append(
                _evaluate_point_centered(
                    "global",
                    measurement,
                    group_test_rows,
                    float(global_info["q_hat"]),
                    len(calibration_groups[group]),
                    cal_info["n_calibration_scores_finite"],
                    degradation,
                    severity,
                    n_conditional_comparisons,
                )
            )
            output_rows.append(
                _evaluate_point_centered(
                    "mondrian",
                    measurement,
                    group_test_rows,
                    cal_info["q_hat"],
                    len(calibration_groups[group]),
                    cal_info["n_calibration_scores_finite"],
                    degradation,
                    severity,
                    n_conditional_comparisons,
                )
            )
            output_rows.append(
                _evaluate_cqr(
                    measurement,
                    group_test_rows,
                    cqr_model,
                    len(calibration_rows),
                    sum(is_finite_measurement_row(row, measurement) for row in calibration_rows),
                    degradation,
                    severity,
                    n_conditional_comparisons,
                )
            )

    _set_width_ratios(output_rows)
    return output_rows, cqr_training


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _row_lookup(rows: Iterable[dict]) -> dict:
    return {
        (row["method"], row["measurement"], row["degradation_type"], str(row["severity_level"])): row
        for row in rows
    }


def _key_cell_summary(rows: List[dict]) -> List[dict]:
    lookup = _row_lookup(rows)
    summary = []
    for measurement, degradation, severity in KEY_CELLS:
        item = {
            "measurement": measurement,
            "degradation_type": degradation,
            "severity_level": severity,
        }
        for method in ["global", "mondrian", "cqr"]:
            row = lookup[(method, measurement, degradation, str(severity))]
            item[f"{method}_coverage"] = row["empirical_coverage"]
            item[f"{method}_mean_width"] = row["mean_interval_width"]
            item[f"{method}_median_width"] = row["median_interval_width"]
        item["cqr_width_ratio_vs_mondrian"] = item["cqr_mean_width"] / item["mondrian_mean_width"]
        summary.append(item)
    return summary


def _method_marginal_summary(rows: List[dict]) -> List[dict]:
    return [
        row for row in rows if row["degradation_type"] == "all" and row["severity_level"] == "all"
    ]


def _cqr_significance_summary(rows: List[dict]) -> dict:
    cqr_conditional = [row for row in rows if row["method"] == "cqr" and row["degradation_type"] != "all"]
    significant = [row for row in cqr_conditional if row["bonferroni_significant_0p05"]]
    worst_gap = min(cqr_conditional, key=lambda row: row["coverage_gap"])
    return {
        "n_cqr_conditional_cells": len(cqr_conditional),
        "n_bonferroni_significant_undercoverage_cells": len(
            [row for row in significant if row["coverage_gap"] < 0]
        ),
        "worst_cqr_gap_cell": {
            "measurement": worst_gap["measurement"],
            "degradation_type": worst_gap["degradation_type"],
            "severity_level": worst_gap["severity_level"],
            "empirical_coverage": worst_gap["empirical_coverage"],
            "coverage_gap": worst_gap["coverage_gap"],
            "bonferroni_pvalue": worst_gap["bonferroni_pvalue"],
        },
    }


def _interpretation(rows: List[dict]) -> str:
    lookup = _row_lookup(rows)
    cqr_marginal = [row for row in rows if row["method"] == "cqr" and row["degradation_type"] == "all"]
    cqr_conditional_bad = [
        row
        for row in rows
        if row["method"] == "cqr"
        and row["degradation_type"] != "all"
        and row["bonferroni_significant_0p05"]
        and row["coverage_gap"] < 0
    ]
    key_width_ratios = []
    for measurement, degradation, severity in KEY_CELLS:
        cqr_row = lookup[("cqr", measurement, degradation, str(severity))]
        key_width_ratios.append(float(cqr_row["width_ratio_vs_mondrian"]))
    if cqr_conditional_bad:
        return "CQR narrows/adapts some intervals but introduces Bonferroni-significant conditional undercoverage."
    if all(row["empirical_coverage"] >= TARGET_COVERAGE - 0.03 for row in cqr_marginal) and np.mean(key_width_ratios) < 1.0:
        return "CQR preserves practical coverage while narrowing key-cell intervals relative to Mondrian."
    return "CQR should be reported as an efficiency tradeoff, not a strict improvement over Mondrian."


def run_experiment_4(
    rows_path: Path = RESULTS_TABLES / "exp2_marginal_rows.json",
    summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    split_path: Path = RESULTS_TABLES / "herlev_group_split.json",
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    raw_dir: Path = DATA_RAW_HERLEV,
    output_dir: Path = RESULTS_TABLES,
    device: str | None = None,
) -> dict:
    calibration_rows, test_rows = _load_exp2_rows(rows_path)
    global_qhats = _load_global_qhats(summary_path)
    qr_train_rows, qr_val_rows, qr_split_report = _collect_qr_rows(split_path, raw_dir, checkpoint_path, device)
    rows, cqr_training = build_exp4_rows(calibration_rows, test_rows, qr_train_rows, qr_val_rows, global_qhats)

    csv_path = output_dir / "exp4_cqr_coverage.csv"
    summary_path_out = output_dir / "exp4_cqr_vs_mondrian_summary.json"
    _write_csv(csv_path, rows)
    result = {
        "output_path": str(csv_path),
        "summary_path": str(summary_path_out),
        "target_coverage": TARGET_COVERAGE,
        "n_rows": len(rows),
        "qr_split": qr_split_report,
        "cqr_training": cqr_training,
        "marginal_summary": _method_marginal_summary(rows),
        "key_cells": _key_cell_summary(rows),
        "cqr_significance": _cqr_significance_summary(rows),
        "interpretation": _interpretation(rows),
    }
    summary_path_out.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_experiment_4(), indent=2))
