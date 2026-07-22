# Cervical Morphometry Conformal — Proje İskeleti

Bu iskelet, "Görüntü Bozulmaları Altında Çekirdek/Sitoplazma Oranı ve Çekirdek
Daireselliği İçin Ortak Conformal Tahmin Aralıkları" projesinin nihai teknik
proje önerisinden (Teknik_Proje_Onerisi.pdf) üretilmiştir.

## Kilitli Kapsam (değiştirme)

- **Ölçümler (2):** N/C oranı, çekirdek daireselliği — bkz. `src/measurements/morphometry.py`
- **Bozulmalar (4):** Gaussian blur, Gaussian noise, kontrast değişimi, düşük çözünürlük —
  bkz. `src/degradation/apply_degradations.py`. Şiddet parametreleri tek seferlik seçilir,
  "doğru filtreyi arama" YOKTUR.
- **Veri setleri:** Herlev (birincil, zorunlu), Cx22 (dış doğrulama, güçlendirme katmanı)

## Zorunlu Çekirdek / Güçlendirme Ayrımı (PDF Bölüm 3.1)

| Zorunlu Çekirdek | Güçlendirme (süre elverirse) |
|---|---|
| Herlev veri seti | Cx22 dış doğrulama |
| U-Net segmentasyon | 5 tekrar koşusu |
| İki ölçüm | CQR (Conformalized Quantile Regression) |
| Dice–morfometrik hata analizi (Deney 1) | Deney 6 — 2 vs 5 ölçüm |
| Split conformal + ortak max-score | |

`src/utils/config.py` içindeki `RUN_CQR`, `RUN_CX22_VALIDATION`,
`RUN_EXPLORATORY_5_MEASUREMENTS` bayrakları bu ayrımı kodda da yansıtır —
zorunlu çekirdek doğrulanmadan bu bayrakları `True` yapma.

## Çalıştırma Sırası (fazlı — bkz. proje planı sohbeti)

1. **Faz 1 — Veri:** `src/data_prep/load_herlev.py` → `validate_herlev_dataset()` çalıştır,
   917 görüntü + maske eşleşmesini doğrula.
2. **Faz 2 — Segmentasyon:** `src/segmentation/train_unet.py` ile eğit, validation Dice > 0.85
   hedefle.
3. **Faz 3 — Ölçümler:** `src/measurements/morphometry.py` fonksiyonlarını birkaç örnekte
   elle doğrula.
4. **Faz 4 — ERKEN KONTROL NOKTASI:** `experiments/exp1_dice_correlation.py` çalıştır.
   Örüntü ("benzer Dice, farklı hata") doğrulanmazsa DUR ve kapsamı yeniden değerlendir.
5. **Faz 5 — Zorunlu çekirdek tamamlama:** `apply_degradations.py` + `split_conformal.py`
   + Deney 2.
6. **Faz 6 — Güçlendirme (süre/güven varsa):** `cqr_joint.py`, Cx22 doğrulama, Deney 3-6.

## Veri Sızıntısı — KRİTİK KURAL

`src/data_prep/group_split.py`: bölme işlemi bozulmuş varyantlar üretilmeden ÖNCE,
orijinal hücre kimliği bazında yapılır. Ana conformal kalibrasyon/test deneylerinde
her hücreden yalnızca bir varyant kullanılır (5 farklı tohumla tekrarlanır).

## Cx22 Yedek Planı

`src/data_prep/load_cx22.py::validate_cx22_compatibility()` uyumsuzluk bulursa,
dış doğrulama `experiments/exp5_cx22_validation.py` içinde `use_fallback=True` ile
Herlev-içi ağır-bozulma domain-shift testine döner. Proje bu durumda çökmez.

## Not

Bu iskeletteki tüm fonksiyonlar `NotImplementedError` fırlatan stub'lardır — kasıtlı.
Kod üretiminde modül modül, her fazı doğrulayarak ilerleyin (bkz. proje planlama
sohbetindeki "Codex'e Vereceğin Sıra" tablosu).
