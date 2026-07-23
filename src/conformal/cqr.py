"""Conformalized quantile regression for scalar morphometry intervals."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.conformal.mondrian_conformal import bounded_interval
from src.conformal.split_conformal import calibrate_split_conformal
from src.utils.config import (
    CQR_HIDDEN_CHANNELS,
    CQR_LEARNING_RATE,
    CQR_LOWER_QUANTILE,
    CQR_MAX_EPOCHS,
    CQR_PATIENCE,
    CQR_UPPER_QUANTILE,
    DEGRADATIONS,
    MEASUREMENT_DOMAINS,
    RANDOM_SEED,
    TARGET_COVERAGE,
)


FEATURE_DEGRADATIONS = list(DEGRADATIONS)


class QuantileRegressor(nn.Module):
    """Small MLP that predicts lower and upper scalar quantiles."""

    def __init__(self, in_features: int, hidden_channels: int = CQR_HIDDEN_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class CQRModel:
    """Trained quantile regressor plus feature normalization metadata."""

    measurement: str
    model: QuantileRegressor
    feature_mean: np.ndarray
    feature_std: np.ndarray
    q_hat: float
    train_loss: float
    val_loss: float
    epochs_trained: int


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return the mean pinball loss for one quantile."""
    errors = y_true - y_pred
    return torch.maximum(quantile * errors, (quantile - 1.0) * errors).mean()


def cqr_nonconformity_scores(q_lower: np.ndarray, q_upper: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return finite CQR scores ``max(q_lower - y, y - q_upper)``."""
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not (q_lower.shape == q_upper.shape == reference.shape):
        raise ValueError("q_lower, q_upper, and reference must have identical shapes.")
    scores = np.maximum(q_lower - reference, reference - q_upper)
    return scores[np.isfinite(scores)]


def _is_finite_row(row: dict, measurement: str) -> bool:
    fields = (
        f"pred_{measurement}",
        f"gt_{measurement}",
        "pred_cytoplasm_area",
        "pred_nucleus_area",
    )
    try:
        values = [float(row[field]) for field in fields]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def build_feature_matrix(rows: Sequence[dict], measurement: str) -> np.ndarray:
    """Build CQR features available at test time only."""
    features = []
    for row in rows:
        degradation = row["degradation"]
        severity = float(row["severity"])
        one_hot = [1.0 if degradation == name else 0.0 for name in FEATURE_DEGRADATIONS]
        features.append(
            [
                *one_hot,
                severity,
                float(row[f"pred_{measurement}"]),
                np.log1p(float(row["pred_nucleus_area"])),
                np.log1p(float(row["pred_cytoplasm_area"])),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def _normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def _prepare_xy(rows: Sequence[dict], measurement: str) -> tuple[np.ndarray, np.ndarray, List[dict]]:
    finite_rows = [row for row in rows if _is_finite_row(row, measurement)]
    features = build_feature_matrix(finite_rows, measurement)
    targets = np.asarray([float(row[f"gt_{measurement}"]) for row in finite_rows], dtype=np.float32)
    return features, targets, finite_rows


def _loss_for_batch(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    lower_loss = pinball_loss(targets, predictions[:, 0], CQR_LOWER_QUANTILE)
    upper_loss = pinball_loss(targets, predictions[:, 1], CQR_UPPER_QUANTILE)
    crossing_penalty = torch.relu(predictions[:, 0] - predictions[:, 1]).mean()
    return lower_loss + upper_loss + 0.05 * crossing_penalty


def train_quantile_regressor(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    measurement: str,
    seed: int = RANDOM_SEED,
) -> CQRModel:
    """Train a small quantile regressor for one measurement."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x, train_y, _ = _prepare_xy(train_rows, measurement)
    val_x, val_y, _ = _prepare_xy(val_rows, measurement)
    if train_x.size == 0 or val_x.size == 0:
        raise ValueError(f"CQR training needs finite train/val rows for {measurement}.")

    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    train_x = _normalize_features(train_x, feature_mean, feature_std)
    val_x = _normalize_features(val_x, feature_mean, feature_std)

    model = QuantileRegressor(in_features=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=CQR_LEARNING_RATE)
    train_tensor_x = torch.from_numpy(train_x)
    train_tensor_y = torch.from_numpy(train_y)
    val_tensor_x = torch.from_numpy(val_x)
    val_tensor_y = torch.from_numpy(val_y)

    best_state = None
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, CQR_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_tensor_x)
        train_loss = _loss_for_batch(train_predictions, train_tensor_y)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor_x)
            val_loss = _loss_for_batch(val_predictions, val_tensor_y)

        val_loss_value = float(val_loss.item())
        if val_loss_value < best_val_loss - 1e-6:
            best_val_loss = val_loss_value
            best_train_loss = float(train_loss.item())
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CQR_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return CQRModel(
        measurement=measurement,
        model=model.eval(),
        feature_mean=feature_mean,
        feature_std=feature_std,
        q_hat=float("nan"),
        train_loss=best_train_loss,
        val_loss=best_val_loss,
        epochs_trained=epoch,
    )


def predict_quantile_bounds(cqr_model: CQRModel, rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, List[dict]]:
    """Predict sorted lower/upper quantile bounds for finite rows."""
    features, _, finite_rows = _prepare_xy(rows, cqr_model.measurement)
    if features.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), []
    normalized = _normalize_features(features, cqr_model.feature_mean, cqr_model.feature_std)
    with torch.no_grad():
        predictions = cqr_model.model(torch.from_numpy(normalized)).cpu().numpy().astype(float)
    lower = np.minimum(predictions[:, 0], predictions[:, 1])
    upper = np.maximum(predictions[:, 0], predictions[:, 1])
    return lower, upper, finite_rows


def conformalize_cqr(cqr_model: CQRModel, calibration_rows: Sequence[dict]) -> CQRModel:
    """Calibrate a trained quantile regressor with split-conformal CQR scores."""
    lower, upper, finite_rows = predict_quantile_bounds(cqr_model, calibration_rows)
    reference = np.asarray([float(row[f"gt_{cqr_model.measurement}"]) for row in finite_rows], dtype=float)
    scores = cqr_nonconformity_scores(lower, upper, reference)
    cqr_model.q_hat = calibrate_split_conformal(scores, alpha=1.0 - TARGET_COVERAGE)
    return cqr_model


def cqr_intervals(cqr_model: CQRModel, rows: Sequence[dict]) -> tuple[List[tuple], List[float], List[dict]]:
    """Return domain-bounded CQR intervals and references for finite rows."""
    lower, upper, finite_rows = predict_quantile_bounds(cqr_model, rows)
    intervals = []
    references = []
    domain_min, domain_max = MEASUREMENT_DOMAINS[cqr_model.measurement]
    for lo, hi, row in zip(lower, upper, finite_rows):
        lo = float(lo) - cqr_model.q_hat
        hi = float(hi) + cqr_model.q_hat
        if domain_min is not None:
            lo = max(float(domain_min), lo)
        if domain_max is not None:
            hi = min(float(domain_max), hi)
        intervals.append((lo, hi))
        references.append(float(row[f"gt_{cqr_model.measurement}"]))
    return intervals, references, finite_rows


def cqr_point_center_interval(cqr_model: CQRModel, row: dict) -> tuple:
    """Convenience helper for comparing one CQR interval to point-centered intervals."""
    intervals, _, _ = cqr_intervals(cqr_model, [row])
    if not intervals:
        return bounded_interval(cqr_model.measurement, float(row[f"pred_{cqr_model.measurement}"]), cqr_model.q_hat)
    return intervals[0]
