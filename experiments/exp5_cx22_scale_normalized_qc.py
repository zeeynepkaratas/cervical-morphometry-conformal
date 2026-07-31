"""Visual QC for the controlled label-aware, scale-normalized Cx22 crop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp5_cx22_bbox_qc import (
    ARCHIVE_NAME,
    _build_generated_canvas,
    _load_ccedd_json_image,
    _match_instances,
    _overlay,
    _read_cx22_names,
    _read_mat_dataset_from_archive,
)
from src.data_prep.cx22_bbox_crop import build_scale_normalized_instance_crop
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.segmentation.train_unet import build_segmentation_target
from src.utils.config import DATA_RAW_CX22, RESULTS_FIGURES


QC_DIRECTORY = RESULTS_FIGURES / "cx22_scale_normalized_qc"


def _target_fraction(scale_report_path: Path) -> float:
    report = json.loads(Path(scale_report_path).read_text(encoding="utf-8"))
    return float(report["herlev_calibration_plus_test"]["whole_cell_fraction"]["median"])


def generate_scale_normalized_qc_grid(
    raw_dir: Path = DATA_RAW_CX22,
    output_path: Path = QC_DIRECTORY / "cx22_scale_normalized_qc.png",
    scale_report_path: Path = RESULTS_FIGURES.parent / "tables" / "exp5_cx22_scale_diagnostic.json",
    n_samples: int = 10,
) -> dict:
    """Render the first ten scale-normalized instance crops for manual approval."""
    raw_dir, output_path = Path(raw_dir), Path(output_path)
    archive_path, ccedd_path = raw_dir / ARCHIVE_NAME, raw_dir / "CCEDD.zip"
    if not archive_path.exists() or not ccedd_path.exists():
        raise FileNotFoundError("Cx22 QC requires Cx22-Multi-Test.zip and CCEDD.zip.")
    target_fraction = _target_fraction(scale_report_path)
    names = _read_cx22_names(archive_path)
    roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
    roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
    nucleus_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cytoplasm_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")

    panels, records = [], []
    try:
        for index, image_name in enumerate(names):
            full = _build_generated_canvas(_load_ccedd_json_image(ccedd_path, image_name), roi_xy[index], roi_wh[index])
            nucleus_masks = _instance_masks_from_mat(nucleus_temp, "nuc_ins", index)
            cytoplasm_masks = _instance_masks_from_mat(cytoplasm_temp, "cyto_ins", index)
            matches = _match_instances(nucleus_masks, cytoplasm_masks)
            if not matches:
                continue
            cyto_index, nucleus_index, nucleus, cytoplasm = max(matches, key=lambda item: int(item[3].sum()))
            crop = build_scale_normalized_instance_crop(np.asarray(full), nucleus, cytoplasm, target_fraction)
            input_target = build_segmentation_target(crop.nucleus_mask, crop.cytoplasm_mask).numpy()
            observed_fraction = float(np.count_nonzero(input_target > 0) / input_target.size)

            full_with_box = full.copy()
            draw = ImageDraw.Draw(full_with_box)
            left, top, right, bottom = crop.crop_box_xyxy
            draw.rectangle((max(0, left), max(0, top), min(full.width, right), min(full.height, bottom)), outline=(255, 220, 0), width=4)
            raw_crop = Image.fromarray(crop.rgb)
            overlay_crop = _overlay(raw_crop, crop.nucleus_mask, crop.cytoplasm_mask)
            panels.append((full_with_box, raw_crop, overlay_crop, f"{image_name} c{cyto_index + 1}/n{nucleus_index + 1} | occ={observed_fraction:.3f}"))
            records.append(
                {
                    "source_image_name": image_name,
                    "sample_index": index,
                    "cytoplasm_instance_index": cyto_index,
                    "nucleus_instance_index": nucleus_index,
                    "crop_box_xyxy": crop.crop_box_xyxy,
                    "target_whole_cell_fraction": target_fraction,
                    "observed_whole_cell_fraction_at_128": observed_fraction,
                    "crop_size": crop.rgb.shape[:2],
                    "ground_truth_role": "Crop centre and scale only; never model input or prediction guidance.",
                }
            )
            if len(panels) == n_samples:
                break
    finally:
        nucleus_temp.unlink(missing_ok=True)
        cytoplasm_temp.unlink(missing_ok=True)
    if not panels:
        raise ValueError("No Cx22 scale-normalized QC panels were produced.")

    cell_width, cell_height, label_height = 220, 220, 30
    sheet = Image.new("RGB", (cell_width * 3, (cell_height + label_height) * len(panels)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for row, (full, raw, overlay, label) in enumerate(panels):
        y = row * (cell_height + label_height)
        sheet.paste(full.resize((cell_width, cell_height), Image.Resampling.BILINEAR), (0, y))
        sheet.paste(raw.resize((cell_width, cell_height), Image.Resampling.BILINEAR), (cell_width, y))
        sheet.paste(overlay.resize((cell_width, cell_height), Image.Resampling.BILINEAR), (cell_width * 2, y))
        draw.text((5, y + cell_height + 5), f"{label} | full+box | RGB only to model | GT overlay", fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": "Controlled label-aware instance crop; GT only determines crop centre/scale. RGB crop alone is model input.",
                "target_whole_cell_fraction": target_fraction,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"output_path": str(output_path), "metadata_path": str(metadata_path), "n_panels": len(panels)}


if __name__ == "__main__":
    print(json.dumps(generate_scale_normalized_qc_grid(), indent=2))
