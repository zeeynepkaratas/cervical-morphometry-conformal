"""Diagnose Cx22 Multi-Test crop geometry relative to Herlev model inputs.

This is a read-only diagnostic. It compares ground-truth occupancy after the
exact aspect-ratio-preserving 128x128 preprocessing used by the Herlev U-Net.
It does not alter Cx22 crops, model weights, or conformal calibration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp5_cx22_bbox_qc import _read_cx22_names
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.segmentation.train_unet import INPUT_SIZE, build_segmentation_target
from src.utils.config import DATA_RAW_CX22, DATA_RAW_HERLEV, RESULTS_TABLES


CX22_ARCHIVE = "Cx22-Multi-Test.zip"


def _union(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("Cannot compute occupancy from an empty mask list.")
    result = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        result |= np.asarray(mask, dtype=bool)
    return result


def _occupancy(target: np.ndarray) -> dict[str, float]:
    total = float(target.size)
    cytoplasm = float(np.count_nonzero(target == 1)) / total
    nucleus = float(np.count_nonzero(target == 2)) / total
    return {
        "cytoplasm_fraction": cytoplasm,
        "nucleus_fraction": nucleus,
        "whole_cell_fraction": cytoplasm + nucleus,
    }


def _distribution(rows: list[dict]) -> dict:
    output = {"n_images": len(rows)}
    for key in ("cytoplasm_fraction", "nucleus_fraction", "whole_cell_fraction"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        output[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return output


def _ratio(cx22: float, herlev: float) -> float:
    return float(cx22 / herlev) if herlev else float("nan")


def run_scale_diagnostic(
    herlev_split_path: Path = RESULTS_TABLES / "herlev_group_split.json",
    herlev_raw_dir: Path = DATA_RAW_HERLEV,
    cx22_raw_dir: Path = DATA_RAW_CX22,
    output_path: Path = RESULTS_TABLES / "exp5_cx22_scale_diagnostic.json",
) -> dict:
    """Compare Herlev calibration/test and the Cx22 Multi-Test partition input occupancy."""
    split = json.loads(Path(herlev_split_path).read_text(encoding="utf-8"))
    herlev_ids = set(split["calibration"]) | set(split["test"])
    herlev_rows = []
    for image_path in list_herlev_images(herlev_raw_dir):
        if image_path.stem not in herlev_ids:
            continue
        _, nucleus, cytoplasm = load_image_and_masks(image_path)
        target = build_segmentation_target(nucleus, cytoplasm).numpy()
        herlev_rows.append({"sample_id": image_path.stem, **_occupancy(target)})
    if len(herlev_rows) != len(herlev_ids):
        raise ValueError(f"Expected {len(herlev_ids)} Herlev calibration/test images, found {len(herlev_rows)}.")

    archive_path = Path(cx22_raw_dir) / CX22_ARCHIVE
    names = _read_cx22_names(archive_path)
    nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
    cx22_rows = []
    try:
        for index, image_name in enumerate(names):
            nucleus = _union(_instance_masks_from_mat(nucleus_temp, "nuc_ins", index))
            cytoplasm = np.logical_and(_union(_instance_masks_from_mat(cytoplasm_temp, "cyto_ins", index)), ~nucleus)
            target = build_segmentation_target(nucleus, cytoplasm).numpy()
            cx22_rows.append({"sample_id": f"Cx22-Multi-Test:{index + 1:06d}", "source_image_name": image_name, **_occupancy(target)})
    finally:
        nucleus_temp.unlink(missing_ok=True)
        cytoplasm_temp.unlink(missing_ok=True)

    herlev_summary = _distribution(herlev_rows)
    cx22_summary = _distribution(cx22_rows)
    comparison = {}
    for mask_name in ("cytoplasm_fraction", "nucleus_fraction", "whole_cell_fraction"):
        comparison[mask_name] = {
            "median_cx22_over_herlev": _ratio(cx22_summary[mask_name]["median"], herlev_summary[mask_name]["median"]),
            "mean_cx22_over_herlev": _ratio(cx22_summary[mask_name]["mean"], herlev_summary[mask_name]["mean"]),
        }
    report = {
        "purpose": "Protocol diagnostic for the Cx22 Multi-Test source partition; it supports the pooled Cx22 ShiftEval evaluation and is not a separate analysis.",
        "preprocessing": "Exact build_segmentation_target() 128x128 isotropic resize plus padding used by the Herlev U-Net.",
        "herlev_calibration_plus_test": herlev_summary,
        "cx22_multi_test": cx22_summary,
        "cx22_over_herlev": comparison,
        "interpretation_rule": (
            "Ratios far below 1 indicate Cx22 annotated cells occupy less of the model input than Herlev; "
            "ratios near 1 do not support a simple foreground-scale explanation."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_scale_diagnostic(), indent=2))
