"""Train U-Net using the committed, disjoint train/validation cell IDs."""

import argparse
import json
from pathlib import Path

from src.utils.config import DATA_RAW_HERLEV, RESULTS_TABLES, ROOT_DIR


def load_training_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = [split[name] for name in ("unet_train", "unet_val", "calibration", "test")]
    all_ids = [cell_id for group in groups for cell_id in group]
    if any(not group for group in groups) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Train, validation, calibration, and test IDs must be nonempty and disjoint.")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_HERLEV)
    parser.add_argument("--split", type=Path, default=RESULTS_TABLES / "herlev_group_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "retraining")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    split = load_training_split(args.split)
    if args.epochs < 1 or not args.raw_dir.is_dir():
        parser.error("Provide an existing Herlev directory and a positive epoch count.")
    from src.segmentation.train_unet import train_unet
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_unet(
        train_ids=split["unet_train"], val_ids=split["unet_val"], epochs=args.epochs,
        raw_dir=args.raw_dir, device=args.device, batch_size=8, base_channels=32,
        checkpoint_path=args.output_dir / "unet_best_trainval.pt",
        log_path=args.output_dir / "unet_trainval_training_log.json",
    )


if __name__ == "__main__":
    main()
