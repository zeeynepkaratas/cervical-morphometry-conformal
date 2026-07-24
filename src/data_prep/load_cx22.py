"""Cx22 compatibility checks for the optional external-validation layer.

Cx22 is handled as a go/no-go strengthening step. The first question is not
model performance; it is whether Python can obtain aligned image + nucleus /
cytoplasm mask pairs without requiring a MATLAB-only generation step.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np


CX22_ARCHIVES = ["Cx22-Pair.zip", "Cx22-Multi-Train.zip", "Cx22-Multi-Test.zip"]
REQUIRED_LABEL_MEMBERS = [
    "CellNum.mat",
    "OverlapRatio.mat",
    "nuc/nuc_ins.mat",
    "cyto/cyto_ins.mat",
    "nuc/nuc_ins_bbox.mat",
    "cyto/cyto_ins_bbox.mat",
    "generator/ImageDataGenerator.m",
    "generator/ImageDataNames.mat",
    "generator/ROIs_x_y.mat",
    "generator/ROIs_W_H.mat",
]


def list_cx22_images(raw_dir: Path) -> List[Path]:
    """
    List generated Cx22 image files, if the optional image generation step has
    already been completed.

    The official Cx22 label archives do not directly contain PNG/BMP images.
    They contain MATLAB v7.3 label files plus a MATLAB generator that creates
    ``ImageDataSet.mat`` after the separate LLPC source images are downloaded.
    """
    raw_dir = Path(raw_dir)
    images = []
    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        images.extend(raw_dir.rglob(pattern))
    return sorted(images)


def load_image_and_masks(image_path: Path) -> Tuple:
    """
    Cx22 image loading is intentionally not implemented before Cx22-0 passes.

    Use ``validate_cx22_compatibility`` first. It reports whether generated
    image data are present. Full image+mask loading should only be implemented
    after that report says external validation can proceed.
    """
    raise NotImplementedError("Cx22 image+mask loading requires generated ImageDataSet.mat or extracted images.")


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("Cx22 MATLAB v7.3 .mat files require h5py for Python-only probing.") from exc
    return h5py


def _archive_members(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as zf:
        return [info.filename for info in zf.infolist()]


def _strip_archive_root(member: str, root_name: str) -> str:
    prefix = root_name.rstrip("/") + "/"
    return member[len(prefix) :] if member.startswith(prefix) else member


def _required_member_status(archive_path: Path) -> dict:
    root_name = archive_path.stem
    members = _archive_members(archive_path)
    normalized = {_strip_archive_root(member, root_name) for member in members}
    missing = [member for member in REQUIRED_LABEL_MEMBERS if member not in normalized]
    has_generated_images = any(
        Path(member).name == "ImageDataSet.mat" or Path(member).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        for member in members
    )
    return {
        "archive": archive_path.name,
        "n_members": len(members),
        "missing_required_members": missing,
        "has_generated_image_data": has_generated_images,
    }


def _extract_member_to_temp(archive_path: Path, suffix: str, contains: str) -> Path:
    with zipfile.ZipFile(archive_path) as zf:
        candidates = [info for info in zf.infolist() if info.filename.endswith(contains)]
        if not candidates:
            raise FileNotFoundError(f"{contains} not found in {archive_path}")
        candidate = candidates[0]
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(zf.read(candidate))
        temp.close()
        return Path(temp.name)


def _sample_instance_stats_from_mat(mat_path: Path, key: str) -> dict:
    h5py = _import_h5py()
    with h5py.File(mat_path, "r") as handle:
        top = handle[key]
        first_cell = handle[top[0, 0]]
        if first_cell.dtype != object:
            mask = np.asarray(first_cell)
            n_instances = 1
        else:
            n_instances = int(first_cell.shape[1] if len(first_cell.shape) > 1 else first_cell.shape[0])
            mask = np.asarray(handle[first_cell[0, 0]])
        return {
            "top_shape": list(top.shape),
            "first_cell_shape": list(first_cell.shape),
            "first_cell_instance_count": n_instances,
            "sample_mask_shape": list(mask.shape),
            "sample_mask_dtype": str(mask.dtype),
            "sample_mask_unique_values": sorted(int(value) for value in np.unique(mask).tolist()),
            "sample_mask_area": int(np.count_nonzero(mask)),
        }


def _probe_archive_labels(archive_path: Path) -> dict:
    nuc_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cyto_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
    try:
        return {
            "archive": archive_path.name,
            "nucleus": _sample_instance_stats_from_mat(nuc_temp, "nuc_ins"),
            "cytoplasm": _sample_instance_stats_from_mat(cyto_temp, "cyto_ins"),
        }
    finally:
        nuc_temp.unlink(missing_ok=True)
        cyto_temp.unlink(missing_ok=True)


def validate_cx22_compatibility(raw_dir: Path, herlev_reference_stats: dict) -> dict:
    """
    Run the Cx22-0 compatibility gate.

    The gate answers whether Cx22 can proceed to real external validation in
    the current workspace. It does not train models or compute coverage.
    """
    raw_dir = Path(raw_dir)
    issues = []
    archives = [raw_dir / name for name in CX22_ARCHIVES if (raw_dir / name).exists()]
    extracted_roots = [raw_dir / Path(name).stem for name in CX22_ARCHIVES if (raw_dir / Path(name).stem).exists()]
    images = list_cx22_images(raw_dir)

    if not raw_dir.exists():
        issues.append(f"raw_dir does not exist: {raw_dir}")
    if not archives and not extracted_roots:
        issues.append("No Cx22 label archives or extracted label folders found in data/raw/cx22.")

    h5py_available = True
    try:
        _import_h5py()
    except ImportError:
        h5py_available = False
        issues.append("h5py is not installed; MATLAB v7.3 .mat labels cannot be probed in Python.")

    archive_status = []
    label_probe = []
    for archive in archives:
        status = _required_member_status(archive)
        archive_status.append(status)
        if status["missing_required_members"]:
            issues.append(f"{archive.name} is missing required members: {status['missing_required_members']}")
        if h5py_available:
            try:
                label_probe.append(_probe_archive_labels(archive))
            except Exception as exc:  # pragma: no cover - diagnostic path
                issues.append(f"{archive.name} label probe failed: {type(exc).__name__}: {exc}")

    generated_image_data_present = bool(images) or any(
        status.get("has_generated_image_data", False) for status in archive_status
    )
    if not generated_image_data_present:
        issues.append(
            "No generated images/ImageDataSet.mat found. Official Cx22 README requires LLPC source images "
            "plus ImageDataGenerator.m before image+label pairs are available."
        )

    labels_python_readable = bool(label_probe) and all(
        probe["nucleus"]["sample_mask_unique_values"] == [0, 1]
        and probe["cytoplasm"]["sample_mask_unique_values"] == [0, 1]
        for probe in label_probe
    )
    compatible = bool(labels_python_readable and generated_image_data_present)
    recommendation = "proceed" if compatible else "fallback_to_herlev_shift_or_generate_cx22_images_first"

    return {
        "compatible": compatible,
        "recommendation": recommendation,
        "issues": issues,
        "raw_dir": str(raw_dir),
        "archives_found": [archive.name for archive in archives],
        "extracted_roots_found": [root.name for root in extracted_roots],
        "n_image_files_found": len(images),
        "h5py_available": h5py_available,
        "labels_python_readable": labels_python_readable,
        "generated_image_data_present": generated_image_data_present,
        "archive_status": archive_status,
        "label_probe": label_probe,
        "herlev_reference_stats_keys": sorted(herlev_reference_stats.keys()) if herlev_reference_stats else [],
        "terms_of_use": "non-commercial research and educational purposes (per official Cx22 README)",
    }
