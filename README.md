# Cervical Morphometry Conformal

Bu repo, "Görüntü Bozulmaları Altında Çekirdek/Sitoplazma Oranı ve Çekirdek
Daireselliği İçin Ortak Conformal Tahmin Aralıkları" projesinin uygulanmış
deneysel hattını içerir.

## Kilitli Kapsam (değiştirme)

- **Ölçümler (2):** N/C oranı, çekirdek daireselliği — bkz. `src/measurements/morphometry.py`
- **Bozulmalar (4):** Gaussian blur, Gaussian noise, kontrast değişimi, düşük çözünürlük —
  bkz. `src/degradation/apply_degradations.py`. Şiddet parametreleri tek seferlik seçilir,
  "doğru filtreyi arama" YOKTUR.
- **Veri setleri:** Herlev (birincil, zorunlu), Cx22 (keşifsel dış stres testi, güçlendirme katmanı)

## Zorunlu Çekirdek / Güçlendirme Ayrımı (PDF Bölüm 3.1)

| Zorunlu Çekirdek | Güçlendirme (süre elverirse) |
|---|---|
| Herlev veri seti | Cx22 keşifsel dış stres testi |
| U-Net segmentasyon | 5 tekrar koşusu |
| İki ölçüm | CQR (Conformalized Quantile Regression) |
| Dice–morfometrik hata analizi (Deney 1) | Deney 6 — 2 vs 5 ölçüm |
| Split conformal + ortak max-score | |

`src/utils/config.py` içindeki `RUN_CQR`, `RUN_CX22_VALIDATION`,
`RUN_EXPLORATORY_5_MEASUREMENTS` bayrakları bu ayrımı kodda da yansıtır.

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
6. **Faz 6 — Güçlendirme:** `experiments/exp3_mondrian_coverage.py`,
   `experiments/exp4_cqr_coverage.py` ve Cx22 ShiftEval dış değerlendirmesi.

## Strict Cell-Wise Protocol

The original all-variant grid remains an empirical degradation analysis. The
strict protocol in `experiments/exp2_strict_cellwise_coverage.py` selects one
degradation variant per original cell for each of five fixed seeds, uses the
disjoint U-Net validation split only to fix joint-score normalization scales,
and reports failure-aware coverage. `experiments/exp3_joint_coverage.py` is
the joint split-conformal entry point; `experiments/exp5_cx22_joint_external.py`
applies fixed Herlev calibration to Cx22 as an exploratory external stress test.
Historical script filenames are retained for repository stability; the shared
`exp3_` prefix does not imply that joint and Mondrian analyses are the same
paper experiment.

## Veri Sızıntısı — KRİTİK KURAL

`src/data_prep/group_split.py`: bölme işlemi bozulmuş varyantlar üretilmeden ÖNCE,
orijinal hücre kimliği bazında yapılır. Ana conformal kalibrasyon/test deneylerinde
her hücreden yalnızca bir varyant kullanılır (5 farklı tohumla tekrarlanır).

## Cx22 Yedek Planı

`src/data_prep/load_cx22.py::validate_cx22_compatibility()` yalnızca Cx22
girdi/etiket kullanılabilirliğini denetler. Gerektiğinde
`experiments/exp5_cx22_validation.py` Herlev-içi ağır-bozulma fallback
özetini üretir; Cx22 ShiftEval ise ayrı, pooled n=1320 hattıdır.

## Cx22 Reporting Convention

Cx22 results are reported as one pooled ShiftEval evaluation across all three
official source partitions (Pair n=820, Multi-Train n=400, Multi-Test n=100;
total n=1320). Multi-Test is retained only as a source-partition breakdown
within that pooled evaluation, not as a separate or earlier Cx22 experiment.

## Uygulama Durumu

Herlev ana hattı, strict cell-wise joint re-analysis ve Cx22 ShiftEval uygulanmış;
sonuçlar `results/` altında saklanmıştır. Tamamlanmamış, kullanılmayan deney
taslakları makaleye giden repodan çıkarılmıştır.
