"""Cx22 bbox-crop visual QC before any external-validation claim.

This is a gate, not a coverage experiment. It ports the official Cx22 image
construction step only far enough to create visual crop/overlay examples from
Cx22-Multi-Test. Full inference must wait until these examples are inspected.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_cx22 import _extract_member_to_temp, _instance_masks_from_mat
from src.utils.config import DATA_RAW_CX22, RESULTS_FIGURES


CANVAS_SIZE = 512
ARCHIVE_NAME = "Cx22-Multi-Test.zip"


def _read_mat_dataset_from_archive(archive_path: Path, member_suffix: str, key: str) -> np.ndarray:
    temp = _extract_member_to_temp(archive_path, ".mat", member_suffix)
    try:
        with h5py.File(temp, "r") as handle:
            data = np.asarray(handle[key])
    finally:
        temp.unlink(missing_ok=True)
    return data.T if data.ndim == 2 and data.shape[0] in {2, 4} else data


def _read_cx22_names(archive_path: Path) -> list[str]:
    """Decode MATLAB v7.3 string arrays used by Cx22 ImageDataNames.mat."""
    temp = _extract_member_to_temp(archive_path, ".mat", "generator/ImageDataNames.mat")
    try:
        with h5py.File(temp, "r") as handle:
            packed = np.asarray(handle["#refs#"]["c"]).ravel()
    finally:
        temp.unlink(missing_ok=True)

    count = int(packed[2])
    lengths = packed[4 : 4 + count].astype(int)
    payload = packed[4 + count :]
    raw_bytes = b"".join(int(value).to_bytes(8, "little") for value in payload)
    chars = raw_bytes.decode("utf-16le", errors="ignore")
    names = []
    cursor = 0
    for length in lengths:
        names.append(chars[cursor : cursor + int(length)])
        cursor += int(length)
    return names


def _load_ccedd_json_image(ccedd_zip_path: Path, image_name: str) -> Image.Image:
    with zipfile.ZipFile(ccedd_zip_path) as outer:
        nested_bytes = outer.read("CCEDD/json.zip")
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
        payload = json.loads(nested.read(f"json/{image_name}.json"))
    image_bytes = base64.b64decode(payload["imageData"])
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _build_generated_canvas(source: Image.Image, roi_xy: np.ndarray, roi_wh: np.ndarray) -> Image.Image:
    """Python port of Cx22 generator/ImageDataGenerator.m for one sample."""
    width, height = int(roi_wh[0]), int(roi_wh[1])
    x, y = int(roi_xy[0]), int(roi_xy[1])
    right = min(x + width, source.width)
    bottom = min(y + height, source.height)
    crop = source.crop((x, y, right, bottom))

    if height >= CANVAS_SIZE or width >= CANVAS_SIZE:
        if height >= width:
            scale = CANVAS_SIZE / height
            new_height = CANVAS_SIZE
            new_width = max(1, int(round(scale * crop.width)))
        else:
            scale = CANVAS_SIZE / width
            new_height = max(1, int(round(scale * crop.height)))
            new_width = CANVAS_SIZE
        crop = crop.resize((new_width, new_height), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(crop, (0, 0))
    return canvas


def _union_bbox(cyto_mask: np.ndarray, nuc_mask: np.ndarray, margin: float = 0.20) -> tuple[int, int, int, int]:
    mask = np.logical_or(cyto_mask, nuc_mask)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("empty Cx22 instance mask")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    width, height = right - left, bottom - top
    side = int(math.ceil(max(width, height) * (1.0 + margin)))
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(CANVAS_SIZE, left + side)
    bottom = min(CANVAS_SIZE, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return left, top, right, bottom


def _overlay(image: Image.Image, nucleus: np.ndarray, cytoplasm: np.ndarray, alpha: float = 0.45) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    arr[cytoplasm] = (1 - alpha) * arr[cytoplasm] + alpha * np.array([0, 80, 255], dtype=np.float32)
    arr[nucleus] = (1 - alpha) * arr[nucleus] + alpha * np.array([255, 0, 0], dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _match_instances(nuc_masks: list[np.ndarray], cyto_masks: list[np.ndarray]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Match Cx22 nucleus and cytoplasm instances by containment, not array index.

    Cx22 stores instance-level masks. In practice, relying on identical nucleus
    and cytoplasm order is brittle, so the visual QC gate pairs each cytoplasm
    with the nucleus whose pixels lie most inside it.
    """
    matches = []
    used_nuclei: set[int] = set()
    for cyto_index, cyto_mask in enumerate(cyto_masks):
        cyto_mask = cyto_mask.astype(bool)
        best = None
        for nuc_index, nuc_mask in enumerate(nuc_masks):
            if nuc_index in used_nuclei:
                continue
            nuc_mask = nuc_mask.astype(bool)
            nuc_area = int(np.count_nonzero(nuc_mask))
            if nuc_area == 0:
                continue
            inside_fraction = float(np.count_nonzero(np.logical_and(nuc_mask, cyto_mask))) / nuc_area
            if best is None or inside_fraction > best[0]:
                best = (inside_fraction, nuc_index, nuc_mask)
        if best is None or best[0] < 0.50:
            continue
        _, nuc_index, nuc_mask = best
        used_nuclei.add(nuc_index)
        cytoplasm_only = np.logical_and(cyto_mask, ~nuc_mask)
        if cytoplasm_only.any():
            matches.append((cyto_index, nuc_index, nuc_mask, cytoplasm_only))
    return matches


def generate_cx22_bbox_qc_grid(
    raw_dir: Path = DATA_RAW_CX22,
    output_path: Path = RESULTS_FIGURES / "cx22_bbox_crop_qc.png",
    n_samples: int = 10,
) -> dict:
    """Create a side-by-side raw/overlay grid for Cx22-Multi-Test bbox crops."""
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / ARCHIVE_NAME
    ccedd_zip_path = raw_dir / "CCEDD.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing Cx22 label archive: {archive_path}")
    if not ccedd_zip_path.exists():
        raise FileNotFoundError(f"Missing CCEDD source archive: {ccedd_zip_path}")

    names = _read_cx22_names(archive_path)
    roi_wh = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_W_H.mat", "ROIs_W_H")
    roi_xy = _read_mat_dataset_from_archive(archive_path, "generator/ROIs_x_y.mat", "ROIs_x_y")
    nuc_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cyto_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")

    try:
        panels = []
        records = []
        for sample_index, image_name in enumerate(names):
            source = _load_ccedd_json_image(ccedd_zip_path, image_name)
            generated = _build_generated_canvas(source, roi_xy[sample_index], roi_wh[sample_index])
            nuc_masks = _instance_masks_from_mat(nuc_temp, "nuc_ins", sample_index)
            cyto_masks = _instance_masks_from_mat(cyto_temp, "cyto_ins", sample_index)
            matched_instances = _match_instances(nuc_masks, cyto_masks)
            candidates = []
            for cyto_index, nuc_index, nuc_mask, cyto_mask in matched_instances:
                candidates.append((int(np.count_nonzero(cyto_mask)), cyto_index, nuc_index, nuc_mask, cyto_mask))
            if not candidates:
                continue
            _, cyto_index, nuc_index, nuc_mask, cyto_mask = max(candidates, key=lambda item: item[0])
            box = _union_bbox(cyto_mask, nuc_mask)
            raw_crop = generated.crop(box)
            nuc_crop = nuc_mask[box[1] : box[3], box[0] : box[2]]
            cyto_crop = cyto_mask[box[1] : box[3], box[0] : box[2]]
            overlay_crop = _overlay(raw_crop, nuc_crop, cyto_crop)
            full_with_box = generated.copy()
            box_draw = ImageDraw.Draw(full_with_box)
            box_draw.rectangle(box, outline=(255, 220, 0), width=4)
            panels.append((full_with_box, raw_crop, overlay_crop, f"{image_name} c{cyto_index + 1}/n{nuc_index + 1}"))
            records.append(
                {
                    "image_name": image_name,
                    "sample_index": sample_index,
                    "cytoplasm_instance_index": cyto_index,
                    "nucleus_instance_index": nuc_index,
                    "crop_box": box,
                    "nucleus_area": int(np.count_nonzero(nuc_crop)),
                    "cytoplasm_area": int(np.count_nonzero(cyto_crop)),
                }
            )
            if len(panels) >= n_samples:
                break
    finally:
        nuc_temp.unlink(missing_ok=True)
        cyto_temp.unlink(missing_ok=True)

    if not panels:
        raise ValueError("No Cx22 bbox crop panels were generated.")

    cell_w, cell_h, label_h = 220, 220, 24
    sheet = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * len(panels)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for row, (full_image, raw_crop, overlay_crop, label) in enumerate(panels):
        y = row * (cell_h + label_h)
        full_resized = full_image.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        raw_resized = raw_crop.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        overlay_resized = overlay_crop.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        sheet.paste(full_resized, (0, y))
        sheet.paste(raw_resized, (cell_w, y))
        sheet.paste(overlay_resized, (cell_w * 2, y))
        draw.text(
            (6, y + cell_h + 4),
            f"{label} | full+box | raw crop | overlay crop",
            fill=(0, 0, 0),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return {
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "n_panels": len(panels),
        "archive": ARCHIVE_NAME,
    }


if __name__ == "__main__":
    print(json.dumps(generate_cx22_bbox_qc_grid(), indent=2))
