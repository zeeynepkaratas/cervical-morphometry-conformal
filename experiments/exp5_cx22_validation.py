"""
DENEY 5 — Cx22 Üzerinde Dış Doğrulama (GÜÇLENDİRME, PDF Bölüm 3.1/3.2/6).

Önkoşul: src.data_prep.load_cx22.validate_cx22_compatibility() "proceed" demiş olmalı.
Eğer "fallback_to_herlev_shift" dönerse, bu dosya yerine Herlev içi ağır-bozulma
alt kümesinde aynı analiz tekrarlanır (yedek plan, PDF 3.2).
"""


def run_experiment_5(use_fallback: bool = False):
    """
    TODO:
        Eğer use_fallback=False:
            1. Herlev'de kalibre edilen ortak conformal aralıkları Cx22 test kümesine uygula.
            2. Empirical coverage Cx22'de hedefi koruyor mu kontrol et.
        Eğer use_fallback=True (Cx22 uyumsuzsa):
            1. Herlev test kümesini bozulma şiddetine göre böl (hafif vs ağır).
            2. Ağır bozulmuş alt kümede kapsamın korunup korunmadığını test et
               (domain-shift proxy'si olarak).
    """
    raise NotImplementedError


if __name__ == "__main__":
    pass
