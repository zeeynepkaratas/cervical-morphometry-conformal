"""
Conformalized Quantile Regression + Ortak (Joint) Kalibrasyon — GÜÇLENDİRME katmanı
(PDF Bölüm 3.1, 5.3).

split_conformal.py çalışıp doğrulanmadan bu dosyaya geçme (config.RUN_CQR bayrağı).
"""

import torch
import torch.nn as nn
import numpy as np


class JointQuantileNet(nn.Module):
    """
    Tek ağlı, dört çıktılı quantile regression (PDF 5.3, Aşama 2):
    N/C oranı alt/üst quantile + dairesellik alt/üst quantile.

    TODO:
        - Girdi: segmentasyon olasılık haritası + tahmin edilen ölçümler.
        - Çıktı: 4 değer (q_low_nc, q_high_nc, q_low_circ, q_high_circ).
    """

    def __init__(self, in_features: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    """TODO: standart pinball/quantile loss formülü."""
    raise NotImplementedError


def joint_nonconformity_score(residuals_nc: np.ndarray, residuals_circ: np.ndarray) -> np.ndarray:
    """
    Ortak (joint) skor: her ölçümün normalize residualının MAKSİMUMU (PDF 5.3, 5.5).

    TODO:
        score_i = max(|residual_nc_i| / scale_nc, |residual_circ_i| / scale_circ)
    """
    raise NotImplementedError


def calibrate_joint_cqr(cal_scores: np.ndarray, target_coverage: float) -> float:
    """TODO: joint_nonconformity_score dağılımının quantile'ını hesapla."""
    raise NotImplementedError
