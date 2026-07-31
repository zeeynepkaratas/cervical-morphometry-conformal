"""
Proje genelinde sabitler ve konfigürasyon.
Kilitli kapsam kararlarını (PDF Bölüm 3.1, 5.1, 5.2) buradan tek noktadan yönet.
"""

from pathlib import Path

# --- Yol sabitleri ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_HERLEV = ROOT_DIR / "data" / "raw" / "herlev"
DATA_RAW_CX22 = ROOT_DIR / "data" / "raw" / "cx22"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
RESULTS_FIGURES = ROOT_DIR / "results" / "figures"
RESULTS_TABLES = ROOT_DIR / "results" / "tables"

# --- Kilitli ölçümler (PDF 5.1 — yalnızca ikisi) ---
MEASUREMENTS = ["nc_ratio", "circularity"]
MEASUREMENT_DOMAINS = {
    "nc_ratio": (0.0, None),
    "circularity": (0.0, 1.0),
}

# --- Kilitli bozulmalar (PDF 5.2 — yalnızca dördü) ---
DEGRADATIONS = ["gaussian_blur", "gaussian_noise", "contrast_change", "low_resolution"]
DEGRADATION_SEVERITY_LEVELS = {
    "gaussian_blur": [1.0, 2.0, 3.0],
    "gaussian_noise": [10.0, 20.0, 30.0],
    "contrast_change": [0.75, 1.25, 1.50],
    "low_resolution": [0.75, 0.50, 0.25],
}

# --- Bağıl hata formülü sabiti (PDF 5.2 — RE = |ŷ-y|/(|y|+eps)) ---
# Used for diagnostic relative-error analyses such as Experiment 1. The
# mandatory split-conformal intervals in Experiment 2 use absolute errors, not
# this epsilon-stabilized relative error. 1e-6 prevents division by zero while
# remaining negligible for the observed Herlev references.
RE_EPSILON = 1e-6

# --- Veri ayrımı (PDF 5.4 — hücre bazlı, sızıntı önleme) ---
SPLIT_RATIOS = {"train": 0.60, "calibration": 0.20, "test": 0.20}
# Reserved for optional repeated-calibration helpers. The mandatory Experiment
# 2 marginal coverage run uses the full deterministic degradation grid once and
# does not consume this repeat count.
N_CALIBRATION_SEED_REPEATS = 5

# --- Conformal hedef kapsam ---
TARGET_COVERAGE = 0.90  # PDF örneklerinde %90 kullanıldı; %95'e çevrilebilir

# --- CQR güçlendirme ayarları ---
# CQR uses these nominal quantiles as the base interval and then applies the
# same finite-sample split-conformal correction on the untouched calibration
# split. The final target coverage is still TARGET_COVERAGE.
CQR_LOWER_QUANTILE = 0.05
CQR_UPPER_QUANTILE = 0.95
CQR_QR_VAL_FRACTION = 0.10
CQR_HIDDEN_CHANNELS = 32
CQR_LEARNING_RATE = 1e-3
CQR_MAX_EPOCHS = 400
CQR_PATIENCE = 40

# --- Zorunlu çekirdek / güçlendirme bayrakları (PDF 3.1) ---
RUN_CQR = True           # Güçlendirme — global + Mondrian baseline doğrulandıktan sonra açıldı
RUN_CX22_VALIDATION = True  # Cx22 pooled ShiftEval uses all official partitions; Multi-Test is a partition-level subgroup.
RUN_EXPLORATORY_5_MEASUREMENTS = False  # Deney 6 — tamamen opsiyonel

RANDOM_SEED = 42
