"""Evaluate the frozen Herlev model on the pre-frozen Cx22-ShiftEval manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp1_dice_correlation import dice_score, foreground_dice, masks_to_target
from experiments.exp2_marginal_coverage import _bounded_interval, _load_model, _predict_target, _safe_measurements
from experiments.exp5_cx22_bbox_qc import _build_generated_canvas, _load_ccedd_json_image, _match_instances, _read_cx22_names, _read_mat_dataset_from_archive
from experiments.exp5_cx22_shifteval_manifest import select_target_instance
from experiments.exp5_cx22_stage_a import _global_q_hats, _measurement_summary, _write_rows
from src.data_prep.cx22_bbox_crop import build_scale_normalized_instance_crop
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.utils.config import DATA_RAW_CX22, MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _summary(rows: list[dict], q_hats: dict[str, float]) -> dict:
    return {
        "n_images": len(rows),
        "mean_foreground_dice": _mean(row["foreground_dice"] for row in rows),
        "median_foreground_dice": float(np.median([float(row["foreground_dice"]) for row in rows])),
        "mean_cytoplasm_dice": _mean(row["cytoplasm_dice"] for row in rows),
        "mean_nucleus_dice": _mean(row["nucleus_dice"] for row in rows),
        "measurements": {measurement: _measurement_summary(rows, measurement, q_hats[measurement]) for measurement in MEASUREMENTS},
    }


def _coverage_test(coverage: float, n: int, n_comparisons: int) -> dict:
    se = math.sqrt(TARGET_COVERAGE * (1.0 - TARGET_COVERAGE) / n)
    z = (coverage - TARGET_COVERAGE) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    adjusted = min(1.0, p * n_comparisons)
    return {"z_score_vs_nominal": z, "two_sided_pvalue_normal_approx": p, "bonferroni_pvalue": adjusted, "bonferroni_significant_0p05": adjusted < 0.05}


def _stratified(rows: list[dict], axis: str, q_hats: dict[str, float], n_comparisons: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[axis])].append(row)
    output = []
    for label, group_rows in sorted(grouped.items()):
        for measurement in MEASUREMENTS:
            finite = [row for row in group_rows if np.isfinite(float(row[f"gt_{measurement}"])) and np.isfinite(float(row[f"pred_{measurement}"]))]
            coverage = float(np.mean([bool(row[f"covered_{measurement}"]) for row in finite])) if finite else float("nan")
            output.append({
                "axis": axis, "stratum": label, "measurement": measurement,
                "n_images": len(group_rows), "n_finite": len(finite), "empirical_coverage": coverage,
                "coverage_gap_vs_0p90": coverage - TARGET_COVERAGE,
                "mean_absolute_error": _mean(row[f"abs_error_{measurement}"] for row in finite),
                "mean_foreground_dice": _mean(row["foreground_dice"] for row in group_rows),
                **_coverage_test(coverage, len(finite), n_comparisons),
            })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _normalise_loaded_rows(rows: list[dict]) -> list[dict]:
    """Restore CSV booleans before deterministic post-inference report rebuilding."""
    for row in rows:
        for measurement in MEASUREMENTS:
            key = f"covered_{measurement}"
            if key in row:
                row[key] = str(row[key]).strip().lower() == "true"
    return rows


def _two_proportion_test(rows: list[dict], measurement: str) -> dict:
    """Exploratory crowded-vs-not-crowded coverage contrast, adjusted over two measurements."""
    groups = {label: [row for row in rows if row["crowding_group"] == label] for label in ("crowded", "not_crowded")}
    counts = {}
    for label, group_rows in groups.items():
        finite = [row for row in group_rows if np.isfinite(float(row[f"gt_{measurement}"])) and np.isfinite(float(row[f"pred_{measurement}"]))]
        covered = sum(bool(row[f"covered_{measurement}"]) for row in finite)
        counts[label] = (covered, len(finite))
    covered_a, n_a = counts["crowded"]
    covered_b, n_b = counts["not_crowded"]
    p_a, p_b = covered_a / n_a, covered_b / n_b
    pooled = (covered_a + covered_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    z = (p_a - p_b) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "measurement": measurement,
        "comparison": "crowded_vs_not_crowded",
        "crowded_covered": covered_a, "crowded_n": n_a, "crowded_coverage": p_a,
        "not_crowded_covered": covered_b, "not_crowded_n": n_b, "not_crowded_coverage": p_b,
        "coverage_difference_crowded_minus_not_crowded": p_a - p_b,
        "z_score": z, "two_sided_pvalue": p,
        "bonferroni_n_comparisons": len(MEASUREMENTS),
        "bonferroni_pvalue": min(1.0, p * len(MEASUREMENTS)),
        "bonferroni_significant_0p05": min(1.0, p * len(MEASUREMENTS)) < 0.05,
        "interpretation": "Exploratory between-group contrast; it is distinct from within-stratum tests against the 0.90 nominal target.",
    }


def rebuild_cx22_shifteval_reports(
    rows_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_rows.csv",
    manifest_summary_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest_summary.json",
    herlev_conformal_summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_summary_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_summary.json",
    output_strata_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_strata.csv",
) -> dict:
    """Rebuild reports from completed rows without rerunning U-Net inference."""
    rows_path = Path(rows_path)
    rows = _normalise_loaded_rows(list(csv.DictReader(rows_path.open(encoding="utf-8", newline=""))))
    frozen = json.loads(Path(manifest_summary_path).read_text(encoding="utf-8"))
    q_hats = _global_q_hats(herlev_conformal_summary_path)
    axes = ("scale_shift_tertile", "nucleus_occupancy_tertile", "crowding_group")
    n_strata_tests = sum(len({row[axis] for row in rows}) for axis in axes) * len(MEASUREMENTS)
    strata = [item for axis in axes for item in _stratified(rows, axis, q_hats, n_strata_tests)]
    _write_csv(Path(output_strata_path), strata)
    by_partition = {partition: _summary([row for row in rows if row["source_partition"] == partition], q_hats) for partition in ("Pair", "Multi-Train", "Multi-Test")}
    summary = {
        "analysis_scope": "Cx22 pooled evaluation (n=1320) with official source-partition breakdowns, including Multi-Test (n=100).",
        "protocol_notice": "All selection and shift labels come from the committed pre-inference, outcome-blind manifest. GT sets crop geometry only; RGB alone is model input. This is one Cx22-derived exploratory evaluation, not a third independent dataset or a deployment protocol.",
        "manifest_sha256_verified": frozen["manifest_sha256"], "n_images": len(rows),
        "model_training_or_finetuning_on_cx22": False, "model_selection_or_calibration_on_cx22": False,
        "conformal_interpretation": "Herlev q_hat values are external stress-test radii only; finite-sample Cx22 coverage is not claimed.",
        "pooled": _summary(rows, q_hats), "by_source_partition": by_partition,
        "stratification": {"axes_analysed_separately": list(axes), "n_within_stratum_tests_for_bonferroni": n_strata_tests, "correction_family": "3 scale strata x 2 measurements + 3 occupancy strata x 2 measurements + 2 crowding strata x 2 measurements = 16", "rows_path": str(rows_path.resolve().relative_to(ROOT_DIR)), "strata_path": str(Path(output_strata_path).resolve().relative_to(ROOT_DIR))},
        "crowding_between_group_tests": [_two_proportion_test(rows, measurement) for measurement in MEASUREMENTS],
    }
    Path(output_summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_cx22_shifteval(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    raw_dir: Path = DATA_RAW_CX22,
    manifest_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest.csv",
    manifest_summary_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest_summary.json",
    herlev_conformal_summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_rows_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_rows.csv",
    output_summary_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_summary.json",
    output_strata_path: Path = RESULTS_TABLES / "exp5_cx22_shifteval_strata.csv",
    device: str | None = None,
) -> dict:
    """Run inference only after verifying the committed outcome-blind manifest hash."""
    raw_dir, manifest_path = Path(raw_dir), Path(manifest_path)
    frozen = json.loads(Path(manifest_summary_path).read_text(encoding="utf-8"))
    observed_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_hash != frozen["manifest_sha256"]:
        raise ValueError("Cx22 ShiftEval manifest hash differs from the frozen pre-inference manifest.")
    manifest_rows = [row for row in csv.DictReader(manifest_path.open(encoding="utf-8", newline="")) if row["inclusion_status"] == "included"]
    if len(manifest_rows) != int(frozen["n_included"]):
        raise ValueError("Manifest row count differs from frozen manifest summary.")
    q_hats = _global_q_hats(herlev_conformal_summary_path)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(Path(checkpoint_path), device_obj)
    ccedd_path = raw_dir / "CCEDD.zip"
    grouped_manifest: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        grouped_manifest[row["archive"]].append(row)

    output_rows: list[dict] = []
    processed = 0
    for archive_name, archive_rows in grouped_manifest.items():
        archive_path = raw_dir / archive_name
        names = _read_cx22_names(archive_path)
        roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
        roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
        nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
        cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
        try:
            for manifest_row in archive_rows:
                processed += 1
                if processed == 1 or processed % 100 == 0 or processed == len(manifest_rows):
                    print(f"Cx22 ShiftEval: {processed}/{len(manifest_rows)} images")
                index = int(manifest_row["source_index"])
                nuclei = _instance_masks_from_mat(nucleus_temp, "nuc_ins", index)
                cytoplasms = _instance_masks_from_mat(cytoplasm_temp, "cyto_ins", index)
                cyto_index, nucleus_index, nucleus, cytoplasm = select_target_instance(_match_instances(nuclei, cytoplasms))
                if (cyto_index, nucleus_index) != (int(manifest_row["cytoplasm_instance_index"]), int(manifest_row["nucleus_instance_index"])):
                    raise ValueError(f"Manifest target mismatch for {manifest_row['sample_id']}.")
                image = np.asarray(_build_generated_canvas(_load_ccedd_json_image(ccedd_path, names[index]), roi_xy[index], roi_wh[index]), dtype=np.uint8)
                crop = build_scale_normalized_instance_crop(image, nucleus, cytoplasm, float(manifest_row["target_whole_cell_fraction"]))
                prediction = _predict_target(model, crop.rgb, crop.rgb.shape[:2], device_obj)
                pred_nucleus, pred_cytoplasm = prediction == 2, prediction == 1
                gt_values, pred_values = _safe_measurements(crop.nucleus_mask, crop.cytoplasm_mask), _safe_measurements(pred_nucleus, pred_cytoplasm)
                row = {**manifest_row, "protocol": "scale_normalized_known_localisation", "reference_scope": "largest matched target instance", "ground_truth_role": "Crop centre and scale only; never model input or prediction guidance.",
                    "selected_cytoplasm_instance_index_verified": cyto_index, "selected_nucleus_instance_index_verified": nucleus_index,
                    "foreground_dice": foreground_dice(prediction, masks_to_target(crop.nucleus_mask, crop.cytoplasm_mask)),
                    "cytoplasm_dice": dice_score(pred_cytoplasm, crop.cytoplasm_mask), "nucleus_dice": dice_score(pred_nucleus, crop.nucleus_mask),
                    "gt_cytoplasm_area": int(crop.cytoplasm_mask.sum()), "pred_cytoplasm_area": int(pred_cytoplasm.sum()),
                    "gt_nucleus_area": int(crop.nucleus_mask.sum()), "pred_nucleus_area": int(pred_nucleus.sum())}
                for measurement in MEASUREMENTS:
                    point, truth = float(pred_values[measurement]), float(gt_values[measurement])
                    row[f"gt_{measurement}"], row[f"pred_{measurement}"] = truth, point
                    row[f"abs_error_{measurement}"] = abs(point - truth) if np.isfinite(point) and np.isfinite(truth) else float("nan")
                    if np.isfinite(point) and np.isfinite(truth):
                        lower, upper = _bounded_interval(measurement, point, q_hats[measurement])
                        row[f"interval_lower_{measurement}"], row[f"interval_upper_{measurement}"], row[f"interval_width_{measurement}"] = lower, upper, upper - lower
                        row[f"covered_{measurement}"] = bool(lower <= truth <= upper)
                    else:
                        row[f"interval_lower_{measurement}"] = row[f"interval_upper_{measurement}"] = row[f"interval_width_{measurement}"] = float("nan")
                        row[f"covered_{measurement}"] = False
                output_rows.append(row)
        finally:
            nucleus_temp.unlink(missing_ok=True); cytoplasm_temp.unlink(missing_ok=True)

    _write_rows(Path(output_rows_path), output_rows)
    report = rebuild_cx22_shifteval_reports(output_rows_path, manifest_summary_path, herlev_conformal_summary_path, output_summary_path, output_strata_path)
    report["checkpoint_path"] = str(checkpoint_path)
    Path(output_summary_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_cx22_shifteval(), indent=2))
