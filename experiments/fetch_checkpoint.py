"""Fetch the frozen U-Net checkpoint and verify its recorded SHA-256.

No PyTorch installation is needed to download or verify the file.
"""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from src.utils.config import ROOT_DIR


CHECKPOINT_SHA256 = "91e8b3430ea8425634485bbfdd210c2b559a10f71a906498db1c4bdd089a835f"
CHECKPOINT_URL = "https://github.com/zeeynepkaratas/cervical-morphometry-conformal/releases/download/reproducibility-v1/unet_best_trainval.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(destination: Path, source: Path | None = None) -> Path:
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != CHECKPOINT_SHA256:
            raise ValueError("Existing checkpoint hash differs; refusing to overwrite it.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download/copy into a temporary file and publish only after validation.
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / "checkpoint.pt"
        if source is None:
            with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        else:
            shutil.copyfile(Path(source), temporary)
        if sha256(temporary) != CHECKPOINT_SHA256:
            raise ValueError("Downloaded checkpoint SHA-256 does not match the frozen model.")
        temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional local copy of the exact frozen checkpoint")
    parser.add_argument("--destination", type=Path, default=ROOT_DIR / "results" / "unet_best_trainval.pt")
    args = parser.parse_args()
    print(fetch_checkpoint(args.destination, args.source))


if __name__ == "__main__":
    main()
