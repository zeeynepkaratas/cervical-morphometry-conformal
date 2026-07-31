"""Label-aware Cx22 instance crops for controlled scale diagnostics.

This module deliberately separates two roles for ground-truth masks:

* The target instance's mask may determine crop centre and crop side.
* After the crop is fixed, the RGB crop alone is passed to the segmenter.

Ground-truth pixels are never appended to the model input, used to alter model
predictions, or used in morphometric prediction. They remain evaluation-only
after crop construction. The protocol therefore requires known localisation
and must be reported as a controlled, label-aware external analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ScaleNormalizedCrop:
    """One square, label-aware crop and its evaluation-only target masks."""

    rgb: np.ndarray
    nucleus_mask: np.ndarray
    cytoplasm_mask: np.ndarray
    crop_box_xyxy: tuple[int, int, int, int]
    target_whole_cell_fraction: float
    source_whole_cell_area: int


def _square_crop_box(mask: np.ndarray, target_whole_cell_fraction: float) -> tuple[int, int, int, int]:
    """Choose a square crop whose target-cell occupancy approaches the target."""
    if not 0.0 < target_whole_cell_fraction <= 1.0:
        raise ValueError("target_whole_cell_fraction must be in (0, 1].")
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Target whole-cell mask is empty; a crop cannot be centred.")

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bbox_side = max(right - left, bottom - top)
    desired_side = math.sqrt(float(mask.sum()) / target_whole_cell_fraction)
    side = max(bbox_side, int(math.ceil(desired_side)))

    centre_x = (left + right) / 2.0
    centre_y = (top + bottom) / 2.0
    crop_left = int(math.floor(centre_x - side / 2.0))
    crop_top = int(math.floor(centre_y - side / 2.0))
    return crop_left, crop_top, crop_left + side, crop_top + side


def _crop_array_with_padding(array: np.ndarray, box: tuple[int, int, int, int], fill: int | tuple[int, int, int]) -> np.ndarray:
    """Crop an array with explicit padding, preserving the requested square size."""
    left, top, right, bottom = box
    height, width = bottom - top, right - left
    if array.ndim == 2:
        result = np.full((height, width), fill, dtype=array.dtype)
    elif array.ndim == 3:
        result = np.full((height, width, array.shape[2]), fill, dtype=array.dtype)
    else:
        raise ValueError(f"Expected 2-D mask or 3-D RGB image, got {array.shape}.")

    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(array.shape[1], right), min(array.shape[0], bottom)
    if src_left >= src_right or src_top >= src_bottom:
        return result
    dst_left, dst_top = src_left - left, src_top - top
    dst_right, dst_bottom = dst_left + (src_right - src_left), dst_top + (src_bottom - src_top)
    result[dst_top:dst_bottom, dst_left:dst_right] = array[src_top:src_bottom, src_left:src_right]
    return result


def build_scale_normalized_instance_crop(
    image: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    target_whole_cell_fraction: float,
) -> ScaleNormalizedCrop:
    """Build a square Cx22 crop centred on one known target-cell instance.

    ``nucleus_mask`` and ``cytoplasm_mask`` define crop geometry only. Once
    this function returns, callers must provide only ``result.rgb`` to the
    model. The masks are evaluation references and are never model features.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    nucleus = np.asarray(nucleus_mask, dtype=bool)
    cytoplasm = np.asarray(cytoplasm_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {rgb.shape}.")
    if nucleus.shape != rgb.shape[:2] or cytoplasm.shape != rgb.shape[:2]:
        raise ValueError("Image and target masks must share the same HxW shape.")
    if np.logical_and(nucleus, cytoplasm).any():
        raise ValueError("Target nucleus and cytoplasm masks must be pixel-exclusive.")

    whole_cell = np.logical_or(nucleus, cytoplasm)
    box = _square_crop_box(whole_cell, target_whole_cell_fraction)
    return ScaleNormalizedCrop(
        rgb=_crop_array_with_padding(rgb, box, fill=(255, 255, 255)),
        nucleus_mask=_crop_array_with_padding(nucleus, box, fill=False).astype(bool),
        cytoplasm_mask=_crop_array_with_padding(cytoplasm, box, fill=False).astype(bool),
        crop_box_xyxy=box,
        target_whole_cell_fraction=float(target_whole_cell_fraction),
        source_whole_cell_area=int(whole_cell.sum()),
    )
