"""Visual QC across all source partitions of the Cx22-ShiftEval manifest."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp5_cx22_bbox_qc import _build_generated_canvas, _load_ccedd_json_image, _match_instances, _overlay, _read_cx22_names, _read_mat_dataset_from_archive
from experiments.exp5_cx22_shifteval_manifest import select_target_instance
from src.data_prep.cx22_bbox_crop import build_scale_normalized_instance_crop
from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.utils.config import DATA_RAW_CX22, RESULTS_FIGURES, RESULTS_TABLES


def _select_qc_rows(rows: list[dict]) -> list[dict]:
    """One deterministic low/middle/high-scale example per official partition."""
    selected = []
    for partition in ("Pair", "Multi-Train", "Multi-Test"):
        for tertile in ("low", "middle", "high"):
            matches = [row for row in rows if row["source_partition"] == partition and row["scale_shift_tertile"] == tertile]
            if not matches:
                raise ValueError(f"No {partition}/{tertile} manifest row.")
            selected.append(min(matches, key=lambda row: int(row["source_index"])))
    return selected


def generate_cx22_shifteval_qc(
    raw_dir: Path = DATA_RAW_CX22,
    manifest_path: Path = RESULTS_TABLES / "cx22_shifteval_manifest.csv",
    output_path: Path = RESULTS_FIGURES / "cx22_shifteval_qc.png",
) -> dict:
    """Render full+crop-box, RGB-only crop, and evaluation overlay for all partitions."""
    raw_dir = Path(raw_dir)
    rows = [row for row in csv.DictReader(Path(manifest_path).open(encoding="utf-8", newline="")) if row["inclusion_status"] == "included"]
    selected = _select_qc_rows(rows)
    target_fraction = float(selected[0]["target_whole_cell_fraction"])
    ccedd_path = raw_dir / "CCEDD.zip"
    panels, metadata = [], []
    by_archive: dict[str, list[dict]] = {}
    for row in selected:
        by_archive.setdefault(row["archive"], []).append(row)
    for archive_name, archive_rows in by_archive.items():
        archive_path = raw_dir / archive_name
        names = _read_cx22_names(archive_path)
        roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
        roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
        nuc_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
        cyto_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
        try:
            for row in archive_rows:
                index = int(row["source_index"])
                image = _build_generated_canvas(_load_ccedd_json_image(ccedd_path, names[index]), roi_xy[index], roi_wh[index])
                nuclei = _instance_masks_from_mat(nuc_temp, "nuc_ins", index)
                cytoplasms = _instance_masks_from_mat(cyto_temp, "cyto_ins", index)
                cyto_index, nucleus_index, nucleus, cytoplasm = select_target_instance(_match_instances(nuclei, cytoplasms))
                crop = build_scale_normalized_instance_crop(np.asarray(image), nucleus, cytoplasm, target_fraction)
                crop_image = Image.fromarray(crop.rgb)
                full = image.copy()
                ImageDraw.Draw(full).rectangle(crop.crop_box_xyxy, outline=(255, 220, 0), width=4)
                panels.append((full, crop_image, _overlay(crop_image, crop.nucleus_mask, crop.cytoplasm_mask), row))
                metadata.append({"sample_id": row["sample_id"], "partition": row["source_partition"], "scale_tertile": row["scale_shift_tertile"], "cytoplasm_instance_index": cyto_index, "nucleus_instance_index": nucleus_index, "ground_truth_role": "Crop centre and scale only; never model input or prediction guidance."})
        finally:
            nuc_temp.unlink(missing_ok=True)
            cyto_temp.unlink(missing_ok=True)

    cell_w, cell_h, label_h = 220, 220, 26
    sheet = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * len(panels)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for row_index, (full, crop, overlay, row) in enumerate(panels):
        y = row_index * (cell_h + label_h)
        for column, panel in enumerate((full, crop, overlay)):
            sheet.paste(panel.resize((cell_w, cell_h), Image.Resampling.BILINEAR), (column * cell_w, y))
        draw.text((5, y + cell_h + 5), f"{row['source_partition']} | {row['scale_shift_tertile']} scale | {row['sample_id']} | full+box | RGB | GT overlay", fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"selection": "First source-index row in each partition x pre-inference scale tertile.", "ground_truth_role": "Crop centre and scale only; never model input or prediction guidance.", "records": metadata}, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), "metadata_path": str(metadata_path), "n_panels": len(panels)}


if __name__ == "__main__":
    print(json.dumps(generate_cx22_shifteval_qc(), indent=2))
