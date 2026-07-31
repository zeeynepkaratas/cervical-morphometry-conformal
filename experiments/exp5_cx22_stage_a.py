"""Auxiliary raw-bbox protocol diagnostic for the Cx22 Multi-Test partition.

The Herlev U-Net is frozen. Cx22 images are never used for model training,
model selection, conformal calibration, or hyperparameter tuning. Herlev's
global split-conformal radii are applied only as an external stress test, not
as a new finite-sample coverage guarantee under the distribution shift.

This retained helper is not a separate reported Cx22 analysis. The canonical
external evaluation is the pooled Cx22 ShiftEval cohort (n=1320), with
Multi-Test reported as one official source-partition breakdown.

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
    _build_generated_canvas,
    _load_ccedd_json_image,
    _match_instances,
    _read_cx22_names,
    _read_mat_dataset_from_archive,
    _union_bbox,
)
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.utils.config import DATA_RAW_CX22, MEASUREMENTS, RESULTS_TABLES, TARGET_COVERAGE


ARCHIVE_NAME = "Cx22-Multi-Test.zip"


def _or_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("Cannot union an empty Cx22 mask list.")
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= np.asarray(mask, dtype=bool)
    return union


def _crowding_metadata(nucleus_masks: list[np.ndarray], cytoplasm_masks: list[np.ndarray]) -> dict:
    """Describe ambiguity around the largest matched cytoplasm instance."""
    matches = _match_instances(nucleus_masks, cytoplasm_masks)
    if not matches:
        return {
            "selected_cytoplasm_instance_index": None,
            "selected_nucleus_instance_index": None,
            "n_candidate_nuclei_in_crop": 0,
            "crowding_flag": False,
        }

    cyto_index, nucleus_index, nucleus, cytoplasm = max(
        matches,
        key=lambda item: int(np.count_nonzero(item[3])),
    )
    left, top, right, bottom = _union_bbox(cytoplasm, nucleus)
    candidates = sum(
        bool(np.asarray(mask, dtype=bool)[top:bottom, left:right].any()) for mask in nucleus_masks
    )
    return {
        "selected_cytoplasm_instance_index": int(cyto_index),
        "selected_nucleus_instance_index": int(nucleus_index),
        "n_candidate_nuclei_in_crop": int(candidates),
        "crowding_flag": bool(candidates > 1),
    }


def _global_q_hats(summary_path: Path) -> dict[str, float]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    q_hats = {row["measurement"]: float(row["q_hat"]) for row in payload["coverage"]}
    missing = set(MEASUREMENTS) - set(q_hats)
    if missing:
        raise ValueError(f"Experiment 2 summary is missing q_hat values for: {sorted(missing)}")
    return q_hats


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No Cx22 Multi-Test raw-bbox diagnostic rows were produced.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _measurement_summary(rows: list[dict], measurement: str, q_hat: float) -> dict:
    finite = [
        row
        for row in rows
        if np.isfinite(float(row[f"gt_{measurement}"])) and np.isfinite(float(row[f"pred_{measurement}"]))
    ]
    covered = [bool(row[f"covered_{measurement}"]) for row in finite]
    widths = [float(row[f"interval_width_{measurement}"]) for row in finite]
    return {
        "q_hat_from_herlev_calibration": q_hat,
        "n_finite": len(finite),
        "mean_absolute_error": _mean(abs(float(row[f"pred_{measurement}"]) - float(row[f"gt_{measurement}"])) for row in finite),
        "empirical_coverage_with_herlev_q_hat": float(np.mean(covered)) if covered else float("nan"),
        "coverage_gap_vs_0p90": (float(np.mean(covered)) - TARGET_COVERAGE) if covered else float("nan"),
        "mean_interval_width": _mean(widths),
    }


def run_cx22_stage_a(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    raw_dir: Path = DATA_RAW_CX22,
    herlev_conformal_summary_path: Path = RESULTS_TABLES / "exp2_marginal_coverage_summary.json",
    output_rows_path: Path = RESULTS_TABLES / "exp5_cx22_stage_a_rows.csv",
    output_summary_path: Path = RESULTS_TABLES / "exp5_cx22_stage_a_summary.json",
    device: str | None = None,
) -> dict:
    """Evaluate the frozen Herlev U-Net on the Multi-Test partition diagnostic."""
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / ARCHIVE_NAME
    ccedd_path = raw_dir / "CCEDD.zip"
    if not archive_path.exists() or not ccedd_path.exists():
        raise FileNotFoundError("The Multi-Test partition diagnostic requires Cx22-Multi-Test.zip and CCEDD.zip in data/raw/cx22.")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    q_hats = _global_q_hats(herlev_conformal_summary_path)
    model = _load_model(Path(checkpoint_path), device_obj)

    names = _read_cx22_names(archive_path)
    roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
    roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
    if not (len(names) == len(roi_wh) == len(roi_xy)):
        raise ValueError("Cx22 source names and ROI arrays have inconsistent lengths.")

    nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
    rows: list[dict] = []
    try:
        for sample_index, image_name in enumerate(names):
            if sample_index == 0 or (sample_index + 1) % 25 == 0 or sample_index + 1 == len(names):
                print(f"Cx22-Multi-Test: {sample_index + 1}/{len(names)} images")

            image = np.asarray(
                _build_generated_canvas(
                    _load_ccedd_json_image(ccedd_path, image_name),
                    roi_xy[sample_index],
                    roi_wh[sample_index],
                ),
                dtype=np.uint8,
            )
            nucleus_instances = _instance_masks_from_mat(nucleus_temp, "nuc_ins", sample_index)
            cytoplasm_instances = _instance_masks_from_mat(cytoplasm_temp, "cyto_ins", sample_index)
            gt_nucleus = _or_masks(nucleus_instances)
            gt_cytoplasm = np.logical_and(_or_masks(cytoplasm_instances), ~gt_nucleus)
            prediction = _predict_target(model, image, image.shape[:2], device_obj)
            pred_nucleus = prediction == 2
            pred_cytoplasm = prediction == 1
            gt_values = _safe_measurements(gt_nucleus, gt_cytoplasm)
            pred_values = _safe_measurements(pred_nucleus, pred_cytoplasm)
            row = {
                "sample_id": f"Cx22-Multi-Test:{sample_index + 1:06d}",
                "source_image_name": image_name,
                "sample_index": sample_index,
                "foreground_dice": foreground_dice(prediction, masks_to_target(gt_nucleus, gt_cytoplasm)),
                "cytoplasm_dice": dice_score(pred_cytoplasm, gt_cytoplasm),
                "nucleus_dice": dice_score(pred_nucleus, gt_nucleus),
                "gt_cytoplasm_area": int(np.count_nonzero(gt_cytoplasm)),
                "pred_cytoplasm_area": int(np.count_nonzero(pred_cytoplasm)),
                "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
                "pred_nucleus_area": int(np.count_nonzero(pred_nucleus)),
                "n_nucleus_instances": len(nucleus_instances),
                "n_cytoplasm_instances": len(cytoplasm_instances),
                **_crowding_metadata(nucleus_instances, cytoplasm_instances),
            }
            for measurement in MEASUREMENTS:
                point = float(pred_values[measurement])
                truth = float(gt_values[measurement])
                row[f"gt_{measurement}"] = truth
                row[f"pred_{measurement}"] = point
                row[f"abs_error_{measurement}"] = abs(point - truth) if np.isfinite(point) and np.isfinite(truth) else float("nan")
                if np.isfinite(point) and np.isfinite(truth):
                    lower, upper = _bounded_interval(measurement, point, q_hats[measurement])
                    row[f"interval_lower_{measurement}"] = lower
                    row[f"interval_upper_{measurement}"] = upper
                    row[f"interval_width_{measurement}"] = upper - lower
                    row[f"covered_{measurement}"] = bool(lower <= truth <= upper)
                else:
                    row[f"interval_lower_{measurement}"] = float("nan")
                    row[f"interval_upper_{measurement}"] = float("nan")
                    row[f"interval_width_{measurement}"] = float("nan")
                    row[f"covered_{measurement}"] = False
            rows.append(row)
    finally:
        nucleus_temp.unlink(missing_ok=True)
        cytoplasm_temp.unlink(missing_ok=True)

    _write_rows(Path(output_rows_path), rows)
    crowded = [row for row in rows if row["crowding_flag"]]
    summary = {
        "analysis_scope": "Auxiliary raw-bbox protocol diagnostic for the Cx22 Multi-Test source partition; not a separate reported Cx22 analysis.",
        "protocol": {
            "dataset_split": "Cx22-Multi-Test source partition within the pooled Cx22 ShiftEval cohort",
            "n_images": len(rows),
            "model_training_or_finetuning_on_cx22": False,
            "model_selection_or_calibration_on_cx22": False,
            "checkpoint_path": str(checkpoint_path),
            "conformal_interpretation": (
                "Herlev global q_hat values are applied without recalibration as an external distribution-shift stress test; "
                "finite-sample split-conformal coverage is not claimed for Cx22."
            ),
        },
        "segmentation": {
            "mean_foreground_dice": _mean(row["foreground_dice"] for row in rows),
            "median_foreground_dice": float(np.median([row["foreground_dice"] for row in rows])),
            "mean_cytoplasm_dice": _mean(row["cytoplasm_dice"] for row in rows),
            "mean_nucleus_dice": _mean(row["nucleus_dice"] for row in rows),
        },
        "measurements": {measurement: _measurement_summary(rows, measurement, q_hats[measurement]) for measurement in MEASUREMENTS},
        "crowding": {
            "definition": "More than one ground-truth nucleus instance intersects the margin-expanded crop around the largest matched cytoplasm instance.",
            "n_crowded": len(crowded),
            "pct_crowded": 100.0 * len(crowded) / len(rows),
            "mean_foreground_dice_crowded": _mean(row["foreground_dice"] for row in crowded),
            "mean_foreground_dice_not_crowded": _mean(row["foreground_dice"] for row in rows if not row["crowding_flag"]),
            "subgroups": {
                "crowded": {
                    "n_images": len(crowded),
                    "measurements": {
                        measurement: _measurement_summary(crowded, measurement, q_hats[measurement])
                        for measurement in MEASUREMENTS
                    },
                },
                "not_crowded": {
                    "n_images": len(rows) - len(crowded),
                    "measurements": {
                        measurement: _measurement_summary(
                            [row for row in rows if not row["crowding_flag"]],
                            measurement,
                            q_hats[measurement],
                        )
                        for measurement in MEASUREMENTS
                    },
                },
            },
        },
        "rows_path": str(output_rows_path),
    }
    Path(output_summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_cx22_stage_a(), indent=2))
