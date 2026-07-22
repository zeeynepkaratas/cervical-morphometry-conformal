"""
Cx22 veri setini okuma ve doğrulama (GÜÇLENDİRME katmanı — PDF Bölüm 3.1/3.2).

Bu modül yalnızca zorunlu çekirdek (Herlev + split conformal) tamamlanıp
doğrulandıktan sonra kullanılmalı.
"""

from pathlib import Path
from typing import List, Tuple


def list_cx22_images(raw_dir: Path) -> List[Path]:
    """
    TODO: Cx22'deki 400 eğitim + 100 test görüntüsünü listele.
    """
    raise NotImplementedError


def load_image_and_masks(image_path: Path) -> Tuple:
    """
    TODO: Cx22 görüntüsünü ve nucleus/cytoplasm maskesini yükle.
    """
    raise NotImplementedError


def validate_cx22_compatibility(raw_dir: Path, herlev_reference_stats: dict) -> dict:
    """
    PDF Bölüm 3.2 "Cx22 İçin Yedek Plan" — ilk gün kontrol edilecek 4 nokta:
        1. Görüntü/maske eşleşmesi doğru mu
        2. nucleus/cytoplasm maskeleri Herlev ile aynı anlamda etiketlenmiş mi
           (aynı piksel kodlama şeması mı, örn. 0=arka plan, 1=sitoplazma, 2=çekirdek)
        3. Çözünürlük/renk formatı Herlev ile ne kadar farklı
        4. Lisans şartları akademik kullanım için uygun mu (repo README'sini kontrol et)

    UYARI (Yedek Plan): Bu kontrollerden biri başarısız olursa Cx22'yi terk etme.
    Bunun yerine RUN_CX22_VALIDATION = False bırak ve dış doğrulamayı
    experiments/exp5_cx22_validation.py yerine, Herlev içi domain-shift testine
    (yalnızca ağır bozulmuş alt kümede kapsam testi) çevir.

    Returns:
        {"compatible": bool, "issues": [...], "recommendation": "proceed" | "fallback_to_herlev_shift"}
    """
    raise NotImplementedError
