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
from PIL import Image


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
GENERATED_DATASET_NAME = "ImageDataSet.mat"


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


def list_cx22_samples(raw_dir: Path) -> List[str]:
    """
    List Cx22 sample ids available for full external validation.

    Full validation requires the official generated ``ImageDataSet.mat`` file.
    The three Cx22 label archives are concatenated in the fixed official order:
    Pair, Multi-Train, Multi-Test. Sample ids are stable strings such as
    ``Cx22-Pair:000001``.
    """
    raw_dir = Path(raw_dir)
    dataset_path = _find_generated_dataset(raw_dir)
    if dataset_path is None:
        return []
    n_images = _count_generated_images(dataset_path)
    archive_counts = _archive_cell_counts(raw_dir)
    samples: list[str] = []
    offset = 0
    for archive_name in CX22_ARCHIVES:
        count = archive_counts.get(archive_name, 0)
        for index in range(count):
            if offset + index < n_images:
                samples.append(_sample_id(archive_name, index))
        offset += count
    return samples


def load_image_and_masks(image_path: Path) -> Tuple:
    """
    Load one generated Cx22 sample and its union nucleus/cytoplasm masks.

    ``image_path`` may be either a concrete generated image file or a synthetic
    Cx22 sample id encoded as ``data/raw/cx22/Cx22-Pair__000001.cx22``. For the
    generated ``ImageDataSet.mat`` route, masks are read from the matching Cx22
    label archive. Instance masks inside an image are unioned before returning
    because the Herlev U-Net predicts semantic classes, not Cx22 instance ids.
    """
    image_path = Path(image_path)
    if image_path.suffix == ".cx22":
        raw_dir = _find_cx22_root(image_path)
        archive_name, index = _parse_sample_path(image_path)
        dataset_index = _global_sample_index(raw_dir, archive_name, index)
        image = _load_generated_image(_find_generated_dataset_or_raise(raw_dir), dataset_index)
        nucleus, cytoplasm = _load_union_masks(raw_dir / archive_name, index)
        return image, nucleus, cytoplasm

    image = np.asarray(Image.open(image_path).convert("RGB"))
    raise NotImplementedError(
        "Direct image-file Cx22 loading needs a naming convention that maps files "
        "back to Cx22 label archive indices. Use generated .cx22 sample ids."
    )


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("Cx22 MATLAB v7.3 .mat files require h5py for Python-only probing.") from exc
    return h5py


def _find_cx22_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name.lower() == "cx22":
            return parent
    return path.parent


def _sample_id(archive_name: str, index: int) -> str:
    return f"{Path(archive_name).stem}:{index + 1:06d}"


def _sample_path(raw_dir: Path, archive_name: str, index: int) -> Path:
    return Path(raw_dir) / f"{Path(archive_name).stem}__{index + 1:06d}.cx22"


def _parse_sample_path(path: Path) -> tuple[str, int]:
    stem = path.stem
    archive_stem, index_text = stem.rsplit("__", 1)
    archive_name = f"{archive_stem}.zip"
    return archive_name, int(index_text) - 1


def _find_generated_dataset(raw_dir: Path) -> Path | None:
    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.rglob(GENERATED_DATASET_NAME)) if raw_dir.exists() else []
    return candidates[0] if candidates else None


def _find_generated_dataset_or_raise(raw_dir: Path) -> Path:
    dataset_path = _find_generated_dataset(raw_dir)
    if dataset_path is None:
        raise FileNotFoundError(
            f"{GENERATED_DATASET_NAME} not found under {raw_dir}. Generate it from LLPC/CCEDD source images first."
        )
    return dataset_path


def _deref(handle, ref):
    return handle[ref]


def _count_generated_images(dataset_path: Path) -> int:
    h5py = _import_h5py()
    with h5py.File(dataset_path, "r") as handle:
        dataset = handle["ImageDataSet"]
        return int(max(dataset.shape))


def _normalise_generated_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"Expected generated Cx22 RGB image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (2, 1, 0))
    elif image.shape[-1] == 3:
        image = image
    elif image.shape[1] == 3:
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Could not infer channel axis for generated Cx22 image shape {image.shape}")
    return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)


def _load_generated_image(dataset_path: Path, index: int) -> np.ndarray:
    h5py = _import_h5py()
    with h5py.File(dataset_path, "r") as handle:
        dataset = handle["ImageDataSet"]
        ref = dataset[index, 0] if dataset.shape[0] >= dataset.shape[1] else dataset[0, index]
        return _normalise_generated_image(np.asarray(_deref(handle, ref)))


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


def _archive_cell_count(archive_path: Path) -> int:
    h5py = _import_h5py()
    temp = _extract_member_to_temp(archive_path, ".mat", "CellNum.mat")
    try:
        with h5py.File(temp, "r") as handle:
            dataset = handle["CellNum"]
            return int(max(dataset.shape))
    finally:
        temp.unlink(missing_ok=True)


def _archive_cell_counts(raw_dir: Path) -> dict[str, int]:
    counts = {}
    for archive_name in CX22_ARCHIVES:
        archive_path = Path(raw_dir) / archive_name
        if archive_path.exists():
            counts[archive_name] = _archive_cell_count(archive_path)
    return counts


def _global_sample_index(raw_dir: Path, archive_name: str, index: int) -> int:
    offset = 0
    counts = _archive_cell_counts(raw_dir)
    for name in CX22_ARCHIVES:
        if name == archive_name:
            return offset + index
        offset += counts.get(name, 0)
    raise KeyError(f"Unknown Cx22 archive: {archive_name}")


def _instance_masks_from_mat(mat_path: Path, key: str, index: int) -> list[np.ndarray]:
    h5py = _import_h5py()
    with h5py.File(mat_path, "r") as handle:
        top = handle[key]
        cell_ref = top[index, 0] if top.shape[0] >= top.shape[1] else top[0, index]
        cell = handle[cell_ref]
        if cell.dtype != object:
            return [np.asarray(cell) > 0]
        n_instances = int(cell.shape[1] if len(cell.shape) > 1 else cell.shape[0])
        masks = []
        for instance_index in range(n_instances):
            ref = cell[0, instance_index] if cell.shape[0] <= cell.shape[1] else cell[instance_index, 0]
            masks.append(np.asarray(handle[ref]) > 0)
        return masks


def _union_instance_masks(mat_path: Path, key: str, index: int) -> np.ndarray:
    masks = _instance_masks_from_mat(mat_path, key, index)
    if not masks:
        raise ValueError(f"No Cx22 masks found for {mat_path.name} index {index}")
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    return union


def _load_union_masks(archive_path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    nuc_temp = _extract_member_to_temp(archive_path, ".mat", "nuc/nuc_ins.mat")
    cyto_temp = _extract_member_to_temp(archive_path, ".mat", "cyto/cyto_ins.mat")
    try:
        nucleus = _union_instance_masks(nuc_temp, "nuc_ins", index)
        cytoplasm = _union_instance_masks(cyto_temp, "cyto_ins", index)
    finally:
        nuc_temp.unlink(missing_ok=True)
        cyto_temp.unlink(missing_ok=True)
    if nucleus.shape != cytoplasm.shape:
        raise ValueError(f"Cx22 nucleus/cytoplasm shape mismatch: {nucleus.shape} vs {cytoplasm.shape}")
    cytoplasm = np.logical_and(cytoplasm, ~nucleus)
    return nucleus, cytoplasm


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
    generated_dataset_path = _find_generated_dataset(raw_dir)

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

    generated_image_data_present = generated_dataset_path is not None or bool(images) or any(
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
    archive_counts = _archive_cell_counts(raw_dir) if h5py_available and archives else {}
    generated_image_count = _count_generated_images(generated_dataset_path) if generated_dataset_path else 0
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
        "generated_dataset_path": str(generated_dataset_path) if generated_dataset_path else None,
        "generated_image_count": generated_image_count,
        "archive_cell_counts": archive_counts,
        "n_listable_samples": len(list_cx22_samples(raw_dir)) if generated_dataset_path else 0,
        "h5py_available": h5py_available,
        "labels_python_readable": labels_python_readable,
        "generated_image_data_present": generated_image_data_present,
        "archive_status": archive_status,
        "label_probe": label_probe,
        "herlev_reference_stats_keys": sorted(herlev_reference_stats.keys()) if herlev_reference_stats else [],
        "terms_of_use": "non-commercial research and educational purposes (per official Cx22 README)",
    }
