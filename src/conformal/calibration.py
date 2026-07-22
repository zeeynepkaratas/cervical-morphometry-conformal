"""
Kalibrasyon sürecini orkestre eden yardımcı modül.
5 farklı rastgele tohumla tekrar (PDF 5.4) burada yönetilir.
"""

from src.utils.config import N_CALIBRATION_SEED_REPEATS


def run_calibration_with_repeats(calibration_fn, n_repeats: int = N_CALIBRATION_SEED_REPEATS):
    """
    TODO:
        - n_repeats farklı tohumla calibration_fn'i çağır (her seferinde
          group_split.sample_one_variant_per_cell ile farklı varyant seçimi).
        - Sonuçların ortalama + standart sapmasını raporla (kalibrasyon
          tekrarları arasındaki değişkenlik — Deney 6'da da kullanılacak).
    """
    raise NotImplementedError
