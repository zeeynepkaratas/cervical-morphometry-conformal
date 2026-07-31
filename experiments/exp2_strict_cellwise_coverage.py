"""Strict cell-wise re-analysis of frozen Experiment 2 inference rows.

Each repeat selects one degradation variant per original cell. Therefore,
calibration and test contain 183 and 184 cells rather than 2,196 and 2,208
correlated variants. Non-finite predictions are failure-aware uncovered cases.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.conformal.joint_conformal import calibration_scales, calibrate_joint_split_conformal, joint_covered
from src.conformal.split_conformal import calibrate_split_conformal
from src.data_prep.group_split import sample_one_variant_per_cell
from src.utils.config import MEASUREMENTS, RANDOM_SEED, RESULTS_TABLES, TARGET_COVERAGE


STRICT_REPEAT_SEEDS = tuple(range(RANDOM_SEED, RANDOM_SEED + 5))


def _load_rows(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["calibration"], payload["test"]


def _load_scale_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _select_one_variant_per_cell(rows: Iterable[dict], seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["cell_id"], []).append(row)
    selected_ids = sample_one_variant_per_cell(
        {cell_id: [row["variant_id"] for row in cell_rows] for cell_id, cell_rows in grouped.items()}, seed
    )
    selected = []
    for cell_id, cell_rows in sorted(grouped.items()):
        matches = [row for row in cell_rows if row["variant_id"] == selected_ids[cell_id]]
        if len(matches) != 1:
            raise AssertionError(f"Expected exactly one selected row for {cell_id}.")
        selected.append(matches[0])
    return selected


def _errors(rows: list[dict], measurement: str) -> np.ndarray:
    return np.asarray([row[f"abs_error_{measurement}"] for row in rows], dtype=float)


def _marginal_result(calibration_rows: list[dict], test_rows: list[dict], measurement: str) -> dict:
    cal_errors = _errors(calibration_rows, measurement)
    cal_finite = np.isfinite(cal_errors)
    q_hat = calibrate_split_conformal(cal_errors[cal_finite], alpha=1.0 - TARGET_COVERAGE)
    test_errors = _errors(test_rows, measurement)
    test_finite = np.isfinite(test_errors)
    covered = np.zeros(test_errors.shape, dtype=bool)
    covered[test_finite] = test_errors[test_finite] <= q_hat
    return {
        "measurement": measurement,
        "q_hat": q_hat,
        "n_calibration_cells_total": len(calibration_rows),
        "n_calibration_cells_finite": int(cal_finite.sum()),
        "n_test_cells_total": len(test_rows),
        "n_test_cells_finite": int(test_finite.sum()),
        "valid_prediction_rate": float(test_finite.mean()),
        "finite_case_coverage": float(covered[test_finite].mean()) if np.any(test_finite) else float("nan"),
        "failure_aware_coverage": float(covered.mean()),
    }


def _joint_result(scale_rows: list[dict], calibration_rows: list[dict], test_rows: list[dict]) -> dict:
    scale_errors = {measurement: _errors(scale_rows, measurement) for measurement in MEASUREMENTS}
    scales = calibration_scales(scale_errors)
    cal_errors = {measurement: _errors(calibration_rows, measurement) for measurement in MEASUREMENTS}
    test_errors = {measurement: _errors(test_rows, measurement) for measurement in MEASUREMENTS}
    calibration = calibrate_joint_split_conformal(cal_errors, alpha=1.0 - TARGET_COVERAGE, scales=scales)
    covered = joint_covered(test_errors, calibration)
    finite = np.logical_and.reduce([np.isfinite(values) for values in test_errors.values()])
    return {
        "q_hat_joint": calibration["q_hat_joint"],
        "scale_nc_ratio": calibration["scales"]["nc_ratio"],
        "scale_circularity": calibration["scales"]["circularity"],
        "n_calibration_cells_total": calibration["n_calibration_cells_total"],
        "n_calibration_cells_jointly_finite": calibration["n_calibration_cells_jointly_finite"],
        "n_test_cells_total": len(test_rows),
        "n_test_cells_jointly_finite": int(finite.sum()),
        "joint_valid_prediction_rate": float(finite.mean()),
        "joint_finite_case_coverage": float(covered[finite].mean()) if np.any(finite) else float("nan"),
        "joint_failure_aware_coverage": float(covered.mean()),
    }


def _aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, members in sorted(groups.items()):
        summary = dict(zip(keys, group))
        for key, value in members[0].items():
            if key not in keys and isinstance(value, (int, float)):
                values = np.asarray([row[key] for row in members], dtype=float)
                summary[f"mean_{key}"] = float(np.nanmean(values))
                summary[f"min_{key}"] = float(np.nanmin(values))
                summary[f"max_{key}"] = float(np.nanmax(values))
        output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_strict_cellwise_analysis(
    rows_path: Path = RESULTS_TABLES / "exp2_marginal_rows.json",
    scale_rows_path: Path = RESULTS_TABLES / "exp2_scale_split_rows.json",
    output_dir: Path = RESULTS_TABLES,
    seeds: tuple[int, ...] = STRICT_REPEAT_SEEDS,
) -> dict:
    """Run five deterministic, one-variant-per-cell marginal and joint analyses."""
    calibration_rows, test_rows = _load_rows(Path(rows_path))
    all_scale_rows = _load_scale_rows(Path(scale_rows_path))
    repeat_rows = []
    for seed in seeds:
        selected_calibration = _select_one_variant_per_cell(calibration_rows, seed)
        selected_test = _select_one_variant_per_cell(test_rows, seed)
        selected_scale = _select_one_variant_per_cell(all_scale_rows, seed)
        for measurement in MEASUREMENTS:
            repeat_rows.append({"seed": seed, "analysis": "marginal", **_marginal_result(selected_calibration, selected_test, measurement)})
        repeat_rows.append({
            "seed": seed,
            "analysis": "joint",
            "measurement": "nc_ratio_and_circularity",
            **_joint_result(selected_scale, selected_calibration, selected_test),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_path = output_dir / "exp2_strict_cellwise_repeats.csv"
    summary_path = output_dir / "exp2_strict_cellwise_summary.json"
    _write_csv(repeat_path, repeat_rows)
    summary = {
        "analysis_scope": "Repeated one-variant-per-original-cell strict re-analysis of frozen Experiment 2 rows with scales fixed on the disjoint U-Net validation split.",
        "interpretation_guardrail": "This removes within-cell variant pseudo-replication but does not make the historically inspected Herlev test split newly untouched.",
        "target_coverage": TARGET_COVERAGE,
        "repeat_seeds": list(seeds),
        "n_calibration_cells_per_repeat": len({row['cell_id'] for row in calibration_rows}),
        "n_test_cells_per_repeat": len({row['cell_id'] for row in test_rows}),
        "n_scale_cells_per_repeat": len({row['cell_id'] for row in all_scale_rows}),
        "marginal_summary": _aggregate([row for row in repeat_rows if row["analysis"] == "marginal"], ["analysis", "measurement"]),
        "joint_summary": _aggregate([row for row in repeat_rows if row["analysis"] == "joint"], ["analysis", "measurement"]),
        "repeat_rows_path": str(repeat_path.relative_to(ROOT_DIR)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_strict_cellwise_analysis(), indent=2))
