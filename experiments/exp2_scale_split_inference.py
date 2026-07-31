"""Generate frozen degradation rows for the disjoint U-Net validation split.

These rows are used only to define fixed normalization scales for strict joint
conformal scoring. They do not contribute to q_hat calibration or test
coverage, and no model weights are updated here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp2_marginal_coverage import _collect_rows, _load_model
from src.data_prep.load_herlev import list_herlev_images
from src.utils.config import DATA_RAW_HERLEV, RESULTS_TABLES


def run_scale_split_inference(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    split_path: Path = RESULTS_TABLES / "herlev_group_split.json",
    raw_dir: Path = DATA_RAW_HERLEV,
    output_path: Path = RESULTS_TABLES / "exp2_scale_split_rows.json",
    device: str | None = None,
) -> dict:
    """Run frozen inference for only the disjoint 55-cell U-Net validation split."""
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    validation_ids = split["unet_val"]
    image_by_stem = {path.stem: path for path in list_herlev_images(Path(raw_dir))}
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(Path(checkpoint_path), device_obj)
    rows = _collect_rows("scale", validation_ids, image_by_stem, model, device_obj)
    result = {
        "analysis_scope": "Disjoint U-Net validation split used only for strict joint-score normalization scales.",
        "checkpoint_path": str(checkpoint_path),
        "n_clean_images": len(validation_ids),
        "n_variants": len(rows),
        "rows": rows,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_scale_split_inference(), indent=2))
