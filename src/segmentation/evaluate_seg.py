"""
Segmentasyon değerlendirme metrikleri (PDF Bölüm 4/5 — Dice, IoU, Boundary F1, Hausdorff).
"""

import numpy as np


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """TODO: Dice = 2*|A∩B| / (|A|+|B|)"""
    raise NotImplementedError


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """TODO: IoU = |A∩B| / |A∪B|"""
    raise NotImplementedError


def boundary_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """TODO: sınır pikselleri arasında F1 (tolerance piksel içinde eşleşme)."""
    raise NotImplementedError


def hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """TODO: scipy.spatial.distance ile Hausdorff mesafesi."""
    raise NotImplementedError
