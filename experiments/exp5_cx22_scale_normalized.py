"""Controlled, label-aware Cx22 scale-normalized instance-crop evaluation.

Ground truth establishes a fixed target-instance crop centre and crop scale
only. The frozen Herlev U-Net receives the resulting RGB crop alone; masks are
used afterwards solely as evaluation references. This is consequently a
controlled known-localisation analysis, not a deployment-time pipeline.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp1_dice_correlation import dice_score, foreground_dice, masks_to_target
from experiments.exp2_marginal_coverage import _bounded_interval, _load_model, _predict_target, _safe_measurements
from experiments.exp5_cx22_bbox_qc import (
    ARCHIVE_NAME,
    _build_generated_canvas,
    _load_ccedd_json_image,
    _match_instances,
    _read_cx22_names,
    _read_mat_dataset_from_archive,
)
from experiments.exp5_cx22_stage_a import _global_q_hats, _measurement_summary, _write_rows
from src.data_prep.cx22_bbox_crop import build_scale_normalized_instance_crop
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.segmentation.train_unet import build_segmentation_target
from src.utils.config import DATA_RAW_CX22, MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


def _mean(values: Iterable[float]) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _crowding_for_crop(nucleus_masks: list[np.ndarray], crop_box: tuple[int, int, int, int]) -> tuple[int, bool]:
    left, top, right, bottom = crop_box
    count = sum(bool(np.asarray(mask, dtype=bool)[max(0, top):bottom, max(0, left):right].any()) for mask in nucleus_masks)
    return int(count), bool(count > 1)


def _occupancy(nucleus: np.ndarray, cytoplasm: np.ndarray) -> dict[str, float]:
    target = build_segmentation_target(nucleus, cytoplasm).numpy()
    total = float(target.size)
    return {
        "nucleus_fraction_at_128": float(np.count_nonzero(target == 2) / total),
        "cytoplasm_fraction_at_128": float(np.count_nonzero(target == 1) / total),
        "whole_cell_fraction_at_128": float(np.count_nonzero(target > 0) / total),
    }


def _native_crop_summary(rows: list[dict]) -> dict:
    sizes = np.asarray([float(row["native_crop_side_px"]) for row in rows])
    return {
        "min": float(sizes.min()), "p05": float(np.percentile(sizes, 5)),
        "median": float(np.median(sizes)), "p95": float(np.percentile(sizes, 95)),
        "max": float(sizes.max()), "n_below_50px": int(np.count_nonzero(sizes < 50)),
    }


def _protocol_summary(rows: list[dict], q_hats: dict[str, float]) -> dict:
    crowded = [row for row in rows if row["crowding_flag"]]
    not_crowded = [row for row in rows if not row["crowding_flag"]]
    return {
        "segmentation": {
            "mean_foreground_dice": _mean(row["foreground_dice"] for row in rows),
            "median_foreground_dice": float(np.median([row["foreground_dice"] for row in rows])),
            "mean_cytoplasm_dice": _mean(row["cytoplasm_dice"] for row in rows),
            "mean_nucleus_dice": _mean(row["nucleus_dice"] for row in rows),
        },
        "measurements": {m: _measurement_summary(rows, m, q_hats[m]) for m in MEASUREMENTS},
        "crowding": {
            "n_crowded": len(crowded), "pct_crowded": 100.0 * len(crowded) / len(rows),
            "mean_foreground_dice_crowded": _mean(row["foreground_dice"] for row in crowded),
            "mean_foreground_dice_not_crowded": _mean(row["foreground_dice"] for row in not_crowded),
            "subgroups": {
                "crowded": {"n_images": len(crowded), "measurements": {m: _measurement_summary(crowded, m, q_hats[m]) for m in MEASUREMENTS}},
                "not_crowded": {"n_images": len(not_crowded), "measurements": {m: _measurement_summary(not_crowded, m, q_hats[m]) for m in MEASUREMENTS}},
            },
        },
        "occupancy_at_model_input": {key: float(np.median([row[key] for row in rows])) for key in ("nucleus_fraction_at_128", "cytoplasm_fraction_at_128", "whole_cell_fraction_at_128")},
        "native_crop_side_px": _native_crop_summary(rows),
    }


def run_scale_normalized_cx22(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    raw_dir: Path = DATA_RAW_CX22,
    herlev_conformal_summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    scale_diagnostic_path: Path = RESULTS_TABLES / "exp5_cx22_scale_diagnostic.json",
    raw_stage_a_rows_path: Path = RESULTS_TABLES / "exp5_cx22_stage_a_rows.csv",
    raw_stage_a_summary_path: Path = RESULTS_TABLES / "exp5_cx22_stage_a_summary.json",
    output_rows_path: Path = RESULTS_TABLES / "exp5_cx22_scale_normalized_rows.csv",
    output_summary_path: Path = RESULTS_TABLES / "exp5_cx22_scale_normalized_summary.json",
    device: str | None = None,
) -> dict:
    """Evaluate all Cx22-Multi-Test target instances under fixed GT crop geometry."""
    raw_dir, checkpoint_path = Path(raw_dir), Path(checkpoint_path)
    archive_path, ccedd_path = raw_dir / ARCHIVE_NAME, raw_dir / "CCEDD.zip"
    if not archive_path.exists() or not ccedd_path.exists():
        raise FileNotFoundError("Requires Cx22-Multi-Test.zip and CCEDD.zip under data/raw/cx22.")
    diagnostic = json.loads(Path(scale_diagnostic_path).read_text(encoding="utf-8"))
    target_fraction = float(diagnostic["herlev_calibration_plus_test"]["whole_cell_fraction"]["median"])
    q_hats = _global_q_hats(herlev_conformal_summary_path)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(checkpoint_path, device_obj)

    names = _read_cx22_names(archive_path)
    roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
    roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
    nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
    rows: list[dict] = []
    try:
        for sample_index, image_name in enumerate(names):
            if sample_index == 0 or (sample_index + 1) % 25 == 0 or sample_index + 1 == len(names):
                print(f"Cx22 scale-normalized: {sample_index + 1}/{len(names)} images")
            image = np.asarray(_build_generated_canvas(_load_ccedd_json_image(ccedd_path, image_name), roi_xy[sample_index], roi_wh[sample_index]), dtype=np.uint8)
            nuclei = _instance_masks_from_mat(nucleus_temp, "nuc_ins", sample_index)
            cytoplasms = _instance_masks_from_mat(cytoplasm_temp, "cyto_ins", sample_index)
            matches = _match_instances(nuclei, cytoplasms)
            if not matches:
                raise ValueError(f"No nucleus/cytoplasm match for Cx22 sample {image_name}.")
            cyto_index, nucleus_index, nucleus, cytoplasm = max(matches, key=lambda item: int(np.count_nonzero(item[3])))
            crop = build_scale_normalized_instance_crop(image, nucleus, cytoplasm, target_fraction)
            prediction = _predict_target(model, crop.rgb, crop.rgb.shape[:2], device_obj)
            pred_nucleus, pred_cytoplasm = prediction == 2, prediction == 1
            gt_values, pred_values = _safe_measurements(crop.nucleus_mask, crop.cytoplasm_mask), _safe_measurements(pred_nucleus, pred_cytoplasm)
            candidates, crowding = _crowding_for_crop(nuclei, crop.crop_box_xyxy)
            row = {
                "protocol": "scale_normalized", "sample_id": f"Cx22-Multi-Test:{sample_index + 1:06d}",
                "source_image_name": image_name, "sample_index": sample_index,
                "reference_scope": "largest matched target instance", "ground_truth_role": "Crop centre and scale only; never model input or prediction guidance.",
                "cytoplasm_instance_index": int(cyto_index), "nucleus_instance_index": int(nucleus_index),
                "crop_box_xyxy": json.dumps(crop.crop_box_xyxy), "native_crop_side_px": int(crop.rgb.shape[0]),
                "target_whole_cell_fraction": target_fraction, **_occupancy(crop.nucleus_mask, crop.cytoplasm_mask),
                "foreground_dice": foreground_dice(prediction, masks_to_target(crop.nucleus_mask, crop.cytoplasm_mask)),
                "cytoplasm_dice": dice_score(pred_cytoplasm, crop.cytoplasm_mask), "nucleus_dice": dice_score(pred_nucleus, crop.nucleus_mask),
                "gt_cytoplasm_area": int(crop.cytoplasm_mask.sum()), "pred_cytoplasm_area": int(pred_cytoplasm.sum()),
                "gt_nucleus_area": int(crop.nucleus_mask.sum()), "pred_nucleus_area": int(pred_nucleus.sum()),
                "n_candidate_nuclei_in_crop": candidates, "crowding_flag": crowding,
            }
            for measurement in MEASUREMENTS:
                point, truth = float(pred_values[measurement]), float(gt_values[measurement])
                row[f"gt_{measurement}"], row[f"pred_{measurement}"] = truth, point
                row[f"abs_error_{measurement}"] = abs(point - truth) if np.isfinite(point) and np.isfinite(truth) else float("nan")
                if np.isfinite(point) and np.isfinite(truth):
                    lower, upper = _bounded_interval(measurement, point, q_hats[measurement])
                    row[f"interval_lower_{measurement}"], row[f"interval_upper_{measurement}"] = lower, upper
                    row[f"interval_width_{measurement}"], row[f"covered_{measurement}"] = upper - lower, bool(lower <= truth <= upper)
                else:
                    row[f"interval_lower_{measurement}"] = row[f"interval_upper_{measurement}"] = row[f"interval_width_{measurement}"] = float("nan")
                    row[f"covered_{measurement}"] = False
            rows.append(row)
    finally:
        nucleus_temp.unlink(missing_ok=True)
        cytoplasm_temp.unlink(missing_ok=True)

    _write_rows(output_rows_path, rows)
    raw_rows = list(csv.DictReader(Path(raw_stage_a_rows_path).open(encoding="utf-8", newline="")))
    for row in raw_rows:
        row["protocol"] = "raw_bbox"
        row["reference_scope"] = "full-frame semantic union"
        row["ground_truth_role"] = "Evaluation-only; no localization crop used."
    comparison_rows = raw_rows + rows
    comparison_path = Path(output_rows_path).with_name("exp5_cx22_raw_vs_scale_normalized_rows.csv")
    fields = list(dict.fromkeys(key for row in comparison_rows for key in row))
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(comparison_rows)
    summary = {
        "analysis_scope": "Auxiliary protocol diagnostic for the Cx22 Multi-Test source partition; canonical Cx22 results are the pooled ShiftEval evaluation (n=1320) with partition-level breakdown.",
        "protocol_notice": "GT masks determine fixed crop centre/scale only. The U-Net receives RGB crops alone; masks are evaluation-only. The normalized protocol evaluates one known target instance, whereas raw_bbox uses full-frame semantic union; this is not a deployment-time or pure scale-only comparison.",
        "target_whole_cell_fraction_from_herlev": target_fraction,
        "n_images": len(rows), "target_instance_policy": "largest matched cytoplasm instance", "model_training_or_finetuning_on_cx22": False,
        "conformal_interpretation": "Herlev q_hat values are external stress-test radii only; finite-sample Cx22 coverage is not claimed.",
        "protocols": {
            "raw_bbox": json.loads(Path(raw_stage_a_summary_path).read_text(encoding="utf-8")),
            "scale_normalized": _protocol_summary(rows, q_hats),
        },
        "rows_path": str(output_rows_path), "comparison_rows_path": str(comparison_path),
    }
    Path(output_summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_scale_normalized_cx22(), indent=2))
