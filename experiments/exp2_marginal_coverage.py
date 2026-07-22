"""Experiment 2: marginal split conformal coverage under image degradations."""

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp1_dice_correlation import masks_to_target, restore_prediction_to_original_size
from src.conformal.split_conformal import (
    calibrate_split_conformal,
    compute_nonconformity_scores,
    empirical_coverage,
    predict_interval,
)
from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.degradation.apply_degradations import DEGRADATION_FUNCTIONS
from src.measurements.morphometry import compute_circularity, compute_nc_ratio
from src.segmentation.train_unet import N_CLASSES, preprocess_rgb_image
from src.segmentation.unet_model import UNet
from src.utils.config import (
    DATA_RAW_HERLEV,
    DEGRADATION_SEVERITY_LEVELS,
    MEASUREMENT_DOMAINS,
    MEASUREMENTS,
    RANDOM_SEED,
    RESULTS_TABLES,
    TARGET_COVERAGE,
)


def _load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = UNet(
        in_channels=3,
        n_classes=N_CLASSES,
        base_channels=int(config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _iter_degradation_specs() -> Iterable[tuple[str, float, str]]:
    for degradation_name, levels in DEGRADATION_SEVERITY_LEVELS.items():
        if degradation_name not in DEGRADATION_FUNCTIONS:
            raise KeyError(f"No degradation function registered for {degradation_name}")
        for level in levels:
            level_label = f"{float(level):g}".replace(".", "p")
            yield degradation_name, float(level), f"{degradation_name}_{level_label}"


def _predict_target(model: UNet, image: np.ndarray, original_shape: tuple[int, int], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        image_tensor = preprocess_rgb_image(image).unsqueeze(0).to(device)
        logits = model(image_tensor)
        pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return restore_prediction_to_original_size(pred_resized, original_shape)


def _stable_variant_seed(split_name: str, cell_id: str, variant_id: str, base_seed: int = RANDOM_SEED) -> int:
    """
    Derive a deterministic 32-bit seed for stochastic degradations.

    Seed strategy for reproducibility:
        seed = sha256(base_seed, split, cell_id, variant_id) mod 2**32

    This makes Gaussian-noise variants independent of loop order and of any
    unrelated NumPy random draws elsewhere in the pipeline.
    """
    key = f"{base_seed}|{split_name}|{cell_id}|{variant_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little", signed=False)


def _safe_measurements(nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> Dict[str, float]:
    values = {}
    try:
        values["nc_ratio"] = compute_nc_ratio(nucleus_mask, cytoplasm_mask)
    except ValueError:
        values["nc_ratio"] = float("nan")
    try:
        values["circularity"] = compute_circularity(nucleus_mask)
    except ValueError:
        values["circularity"] = float("nan")
    return values


def _collect_rows(
    split_name: str,
    image_ids: List[str],
    image_by_stem: Dict[str, Path],
    model: UNet,
    device: torch.device,
) -> List[dict]:
    rows = []
    specs = list(_iter_degradation_specs())
    for index, cell_id in enumerate(image_ids, start=1):
        if index == 1 or index % 25 == 0 or index == len(image_ids):
            print(f"{split_name}: {index}/{len(image_ids)} clean images")

        image_path = image_by_stem[cell_id]
        image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
        gt_values = _safe_measurements(gt_nucleus, gt_cytoplasm)
        original_shape = image.shape[:2]

        for degradation_name, severity, variant_id in specs:
            noise_seed = None
            if degradation_name == "gaussian_noise":
                noise_seed = _stable_variant_seed(split_name, cell_id, variant_id)
                degraded = DEGRADATION_FUNCTIONS[degradation_name](image, severity, seed=noise_seed)
            else:
                degraded = DEGRADATION_FUNCTIONS[degradation_name](image, severity)
            pred_target = _predict_target(model, degraded, original_shape, device)
            pred_nucleus = pred_target == 2
            pred_cytoplasm = pred_target == 1
            pred_values = _safe_measurements(pred_nucleus, pred_cytoplasm)

            row = {
                "split": split_name,
                "cell_id": cell_id,
                "class": image_path.parent.name,
                "degradation": degradation_name,
                "severity": severity,
                "variant_id": variant_id,
                "noise_seed": noise_seed,
                "gt_cytoplasm_area": int(np.count_nonzero(gt_cytoplasm)),
                "pred_cytoplasm_area": int(np.count_nonzero(pred_cytoplasm)),
                "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
                "pred_nucleus_area": int(np.count_nonzero(pred_nucleus)),
            }
            for measurement in MEASUREMENTS:
                row[f"gt_{measurement}"] = gt_values[measurement]
                row[f"pred_{measurement}"] = pred_values[measurement]
                row[f"abs_error_{measurement}"] = abs(pred_values[measurement] - gt_values[measurement])
            rows.append(row)
    return rows


def _summarize_measurement(measurement: str, calibration_rows: List[dict], test_rows: List[dict]) -> dict:
    cal_pred = np.array([row[f"pred_{measurement}"] for row in calibration_rows], dtype=float)
    cal_gt = np.array([row[f"gt_{measurement}"] for row in calibration_rows], dtype=float)
    cal_scores = compute_nonconformity_scores(cal_pred, cal_gt)
    q_hat = calibrate_split_conformal(cal_scores, alpha=1.0 - TARGET_COVERAGE)

    test_pred = np.array([row[f"pred_{measurement}"] for row in test_rows], dtype=float)
    test_gt = np.array([row[f"gt_{measurement}"] for row in test_rows], dtype=float)
    finite = np.isfinite(test_pred) & np.isfinite(test_gt)
    intervals = [_bounded_interval(measurement, point_prediction, q_hat) for point_prediction in test_pred[finite]]
    coverage = empirical_coverage(intervals, test_gt[finite].tolist())
    interval_widths = [upper - lower for lower, upper in intervals]

    return {
        "measurement": measurement,
        "target_coverage": TARGET_COVERAGE,
        "q_hat": q_hat,
        "empirical_coverage": coverage,
        "coverage_gap": coverage - TARGET_COVERAGE,
        "mean_interval_width": float(np.mean(interval_widths)),
        "n_calibration_variants_total": len(calibration_rows),
        "n_calibration_scores_finite": int(cal_scores.size),
        "n_test_variants_total": len(test_rows),
        "n_test_variants_finite": int(finite.sum()),
    }


def _bounded_interval(measurement: str, point_prediction: float, q_hat: float) -> tuple:
    """Intersect conformal intervals with known measurement support."""
    lower, upper = predict_interval(point_prediction, q_hat)
    domain_min, domain_max = MEASUREMENT_DOMAINS[measurement]
    if domain_min is not None:
        lower = max(float(domain_min), lower)
    if domain_max is not None:
        upper = min(float(domain_max), upper)
    return lower, upper


def _sample_interval(measurement: str, summary: dict, test_rows: List[dict]) -> dict:
    for row in test_rows:
        point = float(row[f"pred_{measurement}"])
        truth = float(row[f"gt_{measurement}"])
        if np.isfinite(point) and np.isfinite(truth):
            lower, upper = _bounded_interval(measurement, point, summary["q_hat"])
            return {
                "measurement": measurement,
                "cell_id": row["cell_id"],
                "variant_id": row["variant_id"],
                "point_prediction": point,
                "interval": [lower, upper],
                "ground_truth": truth,
                "covered": lower <= truth <= upper,
            }
    raise ValueError(f"No finite test row found for {measurement}")


def run_experiment_2(
    checkpoint_path: Path = Path("results/unet_best_trainval.pt"),
    split_path: Path = Path("results/tables/herlev_group_split.json"),
    raw_dir: Path = DATA_RAW_HERLEV,
    output_dir: Path = RESULTS_TABLES,
    device: str | None = None,
) -> dict:
    """Run marginal split conformal coverage on calibration/test only."""
    np.random.seed(RANDOM_SEED)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    calibration_ids = split["calibration"]
    test_ids = split["test"]
    image_by_stem = {path.stem: path for path in list_herlev_images(raw_dir)}
    model = _load_model(Path(checkpoint_path), device_obj)

    calibration_rows = _collect_rows("calibration", calibration_ids, image_by_stem, model, device_obj)
    test_rows = _collect_rows("test", test_ids, image_by_stem, model, device_obj)

    summaries = [_summarize_measurement(measurement, calibration_rows, test_rows) for measurement in MEASUREMENTS]
    sample_intervals = {
        summary["measurement"]: _sample_interval(summary["measurement"], summary, test_rows) for summary in summaries
    }

    variant_count = {
        "degradation_types": len(DEGRADATION_SEVERITY_LEVELS),
        "variants_per_clean_image": sum(len(levels) for levels in DEGRADATION_SEVERITY_LEVELS.values()),
        "calibration_clean_images": len(calibration_ids),
        "test_clean_images": len(test_ids),
        "calibration_degraded_variants": len(calibration_rows),
        "test_degraded_variants": len(test_rows),
        "total_degraded_variants": len(calibration_rows) + len(test_rows),
        "train_touched": False,
    }

    result = {
        "checkpoint_path": str(checkpoint_path),
        "split_path": str(split_path),
        "target_coverage": TARGET_COVERAGE,
        "variant_count": variant_count,
        "coverage": summaries,
        "sample_intervals": sample_intervals,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "exp2_marginal_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    (output_dir / "exp2_marginal_rows.json").write_text(
        json.dumps({"calibration": calibration_rows, "test": test_rows}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "exp2_marginal_coverage_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_experiment_2(), indent=2))
