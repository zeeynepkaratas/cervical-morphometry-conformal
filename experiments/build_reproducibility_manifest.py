"""Write a portable provenance manifest for the completed experiment bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import DATA_RAW_CX22, DATA_RAW_HERLEV, RESULTS_TABLES


PACKAGES = ("torch", "numpy", "Pillow", "opencv-python", "scipy", "matplotlib", "h5py")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(path: Path) -> dict:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + _sha256_file(file).encode("ascii") + b"\n")
    return {"path": str(path.relative_to(ROOT_DIR)), "n_files": len(files), "aggregate_sha256": digest.hexdigest()}


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> tuple[dict[str, str | None], list[str]]:
    """Record only versions discoverable in the interpreter that runs this script."""
    versions: dict[str, str | None] = {}
    unavailable: list[str] = []
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            unavailable.append(package)
    return versions, unavailable


def build_manifest(output_path: Path = RESULTS_TABLES / "reproducibility_manifest.json") -> dict:
    """Record direct package versions, source-data digests, checkpoint hash and Git HEAD."""
    checkpoint = ROOT_DIR / "results" / "unet_best_trainval.pt"
    locked_requirements = ROOT_DIR / "requirements-lock.txt"
    package_versions, unavailable_packages = _package_versions()
    result = {
        "git_head_before_manifest_commit": _git_head(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "direct_package_versions": package_versions,
        "packages_unavailable_in_manifest_runtime": unavailable_packages,
        "requirements_lock": {
            "path": str(locked_requirements.relative_to(ROOT_DIR)),
            "sha256": _sha256_file(locked_requirements) if locked_requirements.exists() else None,
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT_DIR)),
            "exists": checkpoint.exists(),
            "sha256": _sha256_file(checkpoint) if checkpoint.exists() else None,
        },
        "raw_data": {
            "herlev": _tree_digest(DATA_RAW_HERLEV),
            "cx22": _tree_digest(DATA_RAW_CX22),
        },
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=2))
