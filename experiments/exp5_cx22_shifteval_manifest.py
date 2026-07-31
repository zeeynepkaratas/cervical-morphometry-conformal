"""Build the outcome-blind manifest for the Cx22-derived ShiftEval protocol.

Selection is deliberately determined before U-Net inference. Every image in
the three official Cx22 archives is considered. An image is included only if
the official label masks contain at least one nucleus/cytoplasm pair matching
by >=50% nucleus containment; the largest matched cytoplasm-only instance is
then selected deterministically. No Dice, prediction, error, or coverage field
is read while constructing this manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp5_cx22_bbox_qc import _match_instances, _read_cx22_names
from src.data_prep.cx22_bbox_crop import build_scale_normalized_instance_crop
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.segmentation.train_unet import build_segmentation_target
from src.utils.config import DATA_RAW_CX22, RESULTS_TABLES


ARCHIVE_NAMES = ("Cx22-Pair.zip", "Cx22-Multi-Train.zip", "Cx22-Multi-Test.zip")


def _tertile_labels(values: list[float]) -> tuple[float, float, list[str]]:
    """Assign deterministic pooled low/middle/high labels from pre-inference geometry."""
    low_cut, high_cut = (float(value) for value in np.quantile(np.asarray(values, dtype=float), [1 / 3, 2 / 3]))
    labels = ["low" if value <= low_cut else "middle" if value <= high_cut else "high" for value in values]
    return low_cut, high_cut, labels


def _crop_crowding(nucleus_masks: list[np.ndarray], box: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = box
    return sum(bool(np.asarray(mask, dtype=bool)[max(0, top):bottom, max(0, left):right].any()) for mask in nucleus_masks)


def select_target_instance(matches: list[tuple[int, int, np.ndarray, np.ndarray]]) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Select largest cytoplasm-only target; ties use smallest cyto then nucleus index."""
    if not matches:
        raise ValueError("Cannot select a target from an empty match list.")
    return max(
        matches,
        key=lambda item: (int(np.count_nonzero(item[3])), -int(item[0]), -int(item[1])),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cx22_shifteval_manifest(
    raw_dir: Path = DATA_RAW_CX22,
    scale_diagnostic_path: Path = RESULTS_TABLES / "exp5_cx22_scale_diagnostic.json",
    output_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest.csv",
    summary_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest_summary.json",
) -> dict:
    """Construct an all-archives, outcome-blind Cx22 ShiftEval manifest."""
    raw_dir = Path(raw_dir)
    target_fraction = float(json.loads(Path(scale_diagnostic_path).read_text(encoding="utf-8"))["herlev_calibration_plus_test"]["whole_cell_fraction"]["median"])
    included, excluded = [], []
    for archive_name in ARCHIVE_NAMES:
        archive_path = raw_dir / archive_name
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing official Cx22 archive: {archive_path}")
        partition = archive_path.stem.replace("Cx22-", "")
        names = _read_cx22_names(archive_path)
        nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
        cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
        try:
            for index, image_name in enumerate(names):
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(names):
                    print(f"ShiftEval manifest {partition}: {index + 1}/{len(names)}")
                nuclei = _instance_masks_from_mat(nucleus_temp, "nuc_ins", index)
                cytoplasms = _instance_masks_from_mat(cytoplasm_temp, "cyto_ins", index)
                matches = _match_instances(nuclei, cytoplasms)
                base = {"archive": archive_name, "source_partition": partition, "source_image_name": image_name, "source_index": index, "sample_id": f"{archive_path.stem}:{index + 1:06d}"}
                if not matches:
                    excluded.append({**base, "inclusion_status": "excluded", "exclusion_reason": "no_matched_nucleus_cytoplasm_instance"})
                    continue
                cyto_index, nucleus_index, nucleus, cytoplasm = select_target_instance(matches)
                # Official generated canvases and masks share the fixed 512x512 frame.
                # Crop geometry depends only on the target masks, not RGB values.
                crop = build_scale_normalized_instance_crop(np.zeros((512, 512, 3), dtype=np.uint8), nucleus, cytoplasm, target_fraction)
                raw_target = build_segmentation_target(nucleus, cytoplasm).numpy()
                whole_area = int(np.count_nonzero(nucleus | cytoplasm))
                nucleus_area = int(np.count_nonzero(nucleus))
                candidate_nuclei = _crop_crowding(nuclei, crop.crop_box_xyxy)
                included.append({
                    **base, "inclusion_status": "included", "exclusion_reason": "",
                    "selection_rule": "largest_matched_cytoplasm_instance",
                    "cytoplasm_instance_index": int(cyto_index), "nucleus_instance_index": int(nucleus_index),
                    "nucleus_area_px": nucleus_area, "whole_cell_area_px": whole_area,
                    "raw_whole_cell_fraction_at_128": float(np.count_nonzero(raw_target > 0) / raw_target.size),
                    "nucleus_to_whole_cell_fraction": float(nucleus_area / whole_area),
                    "n_candidate_nuclei_in_normalized_crop": int(candidate_nuclei),
                    "crowding_group": "crowded" if candidate_nuclei > 1 else "not_crowded",
                    "normalized_crop_side_px": int(crop.rgb.shape[0]),
                    "target_whole_cell_fraction": target_fraction,
                    "outcome_blind": True,
                })
        finally:
            nucleus_temp.unlink(missing_ok=True)
            cytoplasm_temp.unlink(missing_ok=True)

    if not included:
        raise ValueError("No eligible Cx22 samples found.")
    scale_low, scale_high, scale_labels = _tertile_labels([row["raw_whole_cell_fraction_at_128"] for row in included])
    occupancy_low, occupancy_high, occupancy_labels = _tertile_labels([row["nucleus_to_whole_cell_fraction"] for row in included])
    for row, scale_label, occupancy_label in zip(included, scale_labels, occupancy_labels):
        row["scale_shift_tertile"] = scale_label
        row["nucleus_occupancy_tertile"] = occupancy_label

    rows = included + excluded
    output_path = Path(output_path)
    _write_csv(output_path, rows)
    manifest_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary = {
        "name": "Cx22-ShiftEval",
        "description": "Cx22-derived structured external evaluation protocol; not an independent third dataset.",
        "selection_before_model_outcomes": True,
        "candidate_population": "All official Cx22 Pair, Multi-Train, and Multi-Test images (N=1320).",
        "inclusion_rule": "At least one nucleus/cytoplasm pair with >=50% nucleus containment; choose largest matched cytoplasm-only instance deterministically.",
        "tie_break_rule": "If cytoplasm-only areas tie, choose the smallest cytoplasm instance index; if still tied, choose the smallest nucleus instance index.",
        "ground_truth_role": "Defines target-instance geometry and pre-inference shift metadata. RGB crop alone is model input; masks remain evaluation-only after crop construction.",
        "n_candidate_images": len(rows), "n_included": len(included), "n_excluded": len(excluded),
        "partition_counts_included": dict(Counter(row["source_partition"] for row in included)),
        "partition_counts_excluded": dict(Counter(row["source_partition"] for row in excluded)),
        "exclusion_reasons": dict(Counter(row["exclusion_reason"] for row in excluded)),
        "crowding_count_distribution": {str(key): value for key, value in sorted(Counter(int(row["n_candidate_nuclei_in_normalized_crop"]) for row in included).items())},
        "crowding_interpretation": "The binary crowded label means at least one additional nucleus intersects the fixed scale-normalized crop. Its prevalence describes Cx22 cell density; the integer candidate count is retained for sensitivity analyses.",
        "predefined_shift_axes": {
            "scale": {"variable": "raw_whole_cell_fraction_at_128", "labels": ["low", "middle", "high"], "pooled_tertile_cutpoints": [scale_low, scale_high]},
            "nucleus_occupancy": {"variable": "nucleus_to_whole_cell_fraction", "labels": ["low", "middle", "high"], "pooled_tertile_cutpoints": [occupancy_low, occupancy_high]},
            "crowding": {"variable": "n_candidate_nuclei_in_normalized_crop", "labels": ["not_crowded", "crowded"], "threshold": ">1"},
        },
        "analysis_rule": "Each shift axis is analysed separately; outcome stratification is prohibited and axes are not cross-tabulated into small cells.",
        "manifest_sha256": manifest_sha256,
        "manifest_generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(output_path.resolve().relative_to(ROOT_DIR)),
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_cx22_shifteval_manifest(), indent=2))
