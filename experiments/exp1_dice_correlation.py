"""Experiment 1: relationship between Dice and morphometric error."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scipy.stats import spearmanr
except ModuleNotFoundError:  # pragma: no cover
    spearmanr = None

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.measurements.morphometry import compute_circularity, compute_nc_ratio, relative_error
from src.segmentation.train_unet import INPUT_SIZE, N_CLASSES, preprocess_rgb_image
from src.segmentation.unet_model import UNet
from src.utils.config import DATA_RAW_HERLEV, RANDOM_SEED, RESULTS_TABLES


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice = 2*|A intersection B| / (|A|+|B|)."""
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_mask, gt_mask).sum() / denom)


def foreground_dice(pred_target: np.ndarray, gt_target: np.ndarray) -> float:
    """Mean Dice over cytoplasm and nucleus classes."""
    return float(np.mean([dice_score(pred_target == cls, gt_target == cls) for cls in (1, 2)]))


def restore_prediction_to_original_size(
    pred_target: np.ndarray,
    original_shape: Tuple[int, int],
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Undo aspect-ratio-preserving resize + padding for a predicted class map."""
    original_height, original_width = original_shape
    target_width, target_height = input_size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    cropped = pred_target[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width]
    restored = Image.fromarray(cropped.astype(np.uint8), mode="L").resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )
    return np.asarray(restored, dtype=np.uint8)


def masks_to_target(nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> np.ndarray:
    """Build original-resolution 0/1/2 target from pixel-exclusive GT masks."""
    target = np.zeros(nucleus_mask.shape, dtype=np.uint8)
    target[np.logical_and(cytoplasm_mask, ~nucleus_mask)] = 1
    target[nucleus_mask] = 2
    return target


def compute_dice_bins(dice_scores: np.ndarray, bin_width: float = 0.02) -> np.ndarray:
    """Assign Dice values to fixed-width bins."""
    dice_scores = np.asarray(dice_scores, dtype=float)
    return np.floor(dice_scores / bin_width).astype(int) * bin_width


def summarize_error_by_bin(dice_bins: np.ndarray, errors: np.ndarray) -> dict:
    """Summarize error distribution per Dice bin."""
    summary = {}
    for bin_start in sorted(set(dice_bins.tolist())):
        values = np.asarray(errors)[dice_bins == bin_start]
        q1, q3 = np.percentile(values, [25, 75])
        key = f"{bin_start:.2f}-{bin_start + 0.02:.2f}"
        summary[key] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "iqr": float(q3 - q1),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def find_discordant_pairs(
    dice_scores: np.ndarray,
    errors: np.ndarray,
    ids: List[str] | None = None,
    dice_tol: float = 0.02,
    error_tol: float = 0.10,
    max_pairs: int = 25,
) -> list:
    """
    Find pairs with similar Dice but substantially different relative error.

    Discordant pair:
        |Dice_i - Dice_j| <= dice_tol and |RE_i - RE_j| >= error_tol.
    """
    pairs = []
    ids = ids or [str(i) for i in range(len(dice_scores))]
    for i in range(len(dice_scores)):
        for j in range(i + 1, len(dice_scores)):
            dice_diff = abs(float(dice_scores[i]) - float(dice_scores[j]))
            error_diff = abs(float(errors[i]) - float(errors[j]))
            if dice_diff <= dice_tol and error_diff >= error_tol:
                pairs.append(
                    {
                        "id_i": ids[i],
                        "id_j": ids[j],
                        "dice_i": float(dice_scores[i]),
                        "dice_j": float(dice_scores[j]),
                        "error_i": float(errors[i]),
                        "error_j": float(errors[j]),
                        "dice_diff": dice_diff,
                        "error_diff": error_diff,
                    }
                )
    return sorted(pairs, key=lambda row: row["error_diff"], reverse=True)[:max_pairs]


def compute_dice_error_correlation(dice_scores: np.ndarray, errors: np.ndarray) -> Dict[str, float]:
    """Return Spearman correlation between Dice and relative error."""
    dice_scores = np.asarray(dice_scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if spearmanr is not None:
        result = spearmanr(dice_scores, errors)
        return {"rho": float(result.statistic), "pvalue": float(result.pvalue)}

    dice_ranks = np.argsort(np.argsort(dice_scores)).astype(float)
    error_ranks = np.argsort(np.argsort(errors)).astype(float)
    rho = float(np.corrcoef(dice_ranks, error_ranks)[0, 1])
    return {"rho": rho, "pvalue": float("nan")}


def checkpoint_decision(high_dice_error_fraction: float) -> str:
    """Return an early checkpoint recommendation."""
    if high_dice_error_fraction >= 0.15:
        return "pattern_supported_continue"
    return "pattern_weak_review_scope"


def run_experiment(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    split_path: Path = Path("results/tables/herlev_group_split.json"),
    raw_dir: Path = DATA_RAW_HERLEV,
    output_dir: Path = RESULTS_TABLES,
    device: str | None = None,
) -> dict:
    """Run Experiment 1 on the untouched test split only."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    test_ids = split["test"]
    image_by_stem = {path.stem: path for path in list_herlev_images(raw_dir)}

    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint.get("config", {})
    model = UNet(
        in_channels=3,
        n_classes=N_CLASSES,
        base_channels=int(config.get("base_channels", 32)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for cell_id in test_ids:
            image_path = image_by_stem[cell_id]
            image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
            image_tensor = preprocess_rgb_image(image).unsqueeze(0).to(device_obj)
            logits = model(image_tensor)
            pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2])
            gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)

            pred_nucleus = pred_target == 2
            pred_cytoplasm = pred_target == 1
            gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
            gt_circularity = compute_circularity(gt_nucleus)

            try:
                pred_nc = compute_nc_ratio(pred_nucleus, pred_cytoplasm)
                nc_re = relative_error(pred_nc, gt_nc)
            except ValueError:
                pred_nc = float("nan")
                nc_re = float("inf")

            try:
                pred_circularity = compute_circularity(pred_nucleus)
                circularity_re = relative_error(pred_circularity, gt_circularity)
            except ValueError:
                pred_circularity = float("nan")
                circularity_re = float("inf")

            rows.append(
                {
                    "cell_id": cell_id,
                    "class": image_path.parent.name,
                    "dice": foreground_dice(pred_target, gt_target),
                    "dice_cytoplasm": dice_score(pred_cytoplasm, gt_cytoplasm),
                    "dice_nucleus": dice_score(pred_nucleus, gt_nucleus),
                    "gt_cytoplasm_area": int(np.count_nonzero(gt_cytoplasm)),
                    "pred_cytoplasm_area": int(np.count_nonzero(pred_cytoplasm)),
                    "pred_over_gt_cytoplasm_area": float(
                        np.count_nonzero(pred_cytoplasm) / max(1, np.count_nonzero(gt_cytoplasm))
                    ),
                    "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
                    "pred_nucleus_area": int(np.count_nonzero(pred_nucleus)),
                    "pred_over_gt_nucleus_area": float(
                        np.count_nonzero(pred_nucleus) / max(1, np.count_nonzero(gt_nucleus))
                    ),
                    "gt_nc_ratio": gt_nc,
                    "pred_nc_ratio": pred_nc,
                    "nc_relative_error": nc_re,
                    "gt_circularity": gt_circularity,
                    "pred_circularity": pred_circularity,
                    "circularity_relative_error": circularity_re,
                }
            )

    dice_scores = np.array([row["dice"] for row in rows], dtype=float)
    nc_errors = np.array([row["nc_relative_error"] for row in rows], dtype=float)
    circularity_errors = np.array([row["circularity_relative_error"] for row in rows], dtype=float)
    finite_nc = np.isfinite(nc_errors)
    finite_circ = np.isfinite(circularity_errors)
    high_dice = dice_scores >= 0.90
    high_dice_count = int(high_dice.sum())
    high_dice_nc_error_count = int(np.logical_and(high_dice, nc_errors > 0.10).sum())
    high_dice_nc_error_fraction = (
        float(high_dice_nc_error_count / high_dice_count) if high_dice_count else float("nan")
    )
    degenerate_cytoplasm = np.array(
        [row["pred_over_gt_cytoplasm_area"] < 0.10 for row in rows],
        dtype=bool,
    )
    nondegenerate_high_dice = np.logical_and(high_dice, ~degenerate_cytoplasm)
    nondegenerate_high_dice_count = int(nondegenerate_high_dice.sum())
    nondegenerate_high_dice_nc_error_count = int(
        np.logical_and(nondegenerate_high_dice, nc_errors > 0.10).sum()
    )

    summary = {
        "n_test": len(rows),
        "checkpoint_path": str(checkpoint_path),
        "split_used": "test",
        "calibration_touched": False,
        "spearman_dice_vs_nc_error": compute_dice_error_correlation(dice_scores[finite_nc], nc_errors[finite_nc]),
        "spearman_dice_vs_circularity_error": compute_dice_error_correlation(
            dice_scores[finite_circ], circularity_errors[finite_circ]
        ),
        "dice_bins_nc_error": summarize_error_by_bin(compute_dice_bins(dice_scores[finite_nc]), nc_errors[finite_nc]),
        "discordant_pairs_nc_error": find_discordant_pairs(
            dice_scores[finite_nc],
            nc_errors[finite_nc],
            ids=[row["cell_id"] for row, keep in zip(rows, finite_nc) if keep],
        ),
        "high_dice_threshold": 0.90,
        "high_dice_count": high_dice_count,
        "high_dice_nc_error_gt_10pct_count": high_dice_nc_error_count,
        "high_dice_nc_error_gt_10pct_fraction": high_dice_nc_error_fraction,
        "degenerate_pred_cytoplasm_lt_10pct_gt_count": int(degenerate_cytoplasm.sum()),
        "high_dice_degenerate_pred_cytoplasm_lt_10pct_gt_count": int(
            np.logical_and(high_dice, degenerate_cytoplasm).sum()
        ),
        "high_dice_nc_error_gt_10pct_fraction_excluding_degenerate_cytoplasm": (
            float(nondegenerate_high_dice_nc_error_count / nondegenerate_high_dice_count)
            if nondegenerate_high_dice_count
            else float("nan")
        ),
        "top5_nc_relative_error_area_check": sorted(
            [
                {
                    "cell_id": row["cell_id"],
                    "class": row["class"],
                    "dice": row["dice"],
                    "nc_relative_error": row["nc_relative_error"],
                    "gt_cytoplasm_area": row["gt_cytoplasm_area"],
                    "pred_cytoplasm_area": row["pred_cytoplasm_area"],
                    "pred_over_gt_cytoplasm_area": row["pred_over_gt_cytoplasm_area"],
                    "gt_nucleus_area": row["gt_nucleus_area"],
                    "pred_nucleus_area": row["pred_nucleus_area"],
                    "pred_over_gt_nucleus_area": row["pred_over_gt_nucleus_area"],
                    "degenerate_pred_cytoplasm_lt_10pct_gt": row["pred_over_gt_cytoplasm_area"] < 0.10,
                }
                for row in rows
            ],
            key=lambda row: row["nc_relative_error"],
            reverse=True,
        )[:5],
        "decision": checkpoint_decision(high_dice_nc_error_fraction),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exp1_dice_morphometry_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "exp1_checkpoint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, indent=2))
