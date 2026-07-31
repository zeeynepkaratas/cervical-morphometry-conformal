# 9. Dosya Haritası

Önceki bölüm: [Metodolojik kararlar ve sınırlar](08_metodolojik_kararlar_ve_sinirlar.md). Sonraki bölüm: [Makaleye geçiş notları](10_makaleye_gecis_notlari.md).

Bu harita, önemli dosyaların görevini ve veri akışını açıklar. Ham görüntüler `data/raw/` altında, üretilmiş tablolar `results/tables/` altında, görsel kontroller `results/figures/` altında tutulur.

> Not: Tarihsel script adları korunmuştur. Bu nedenle `exp3_joint_coverage.py` ile `exp3_mondrian_coverage.py` aynı deney değildir; aşağıdaki görev açıklamaları dosya adından daha belirleyicidir.

## Başlangıç ve ayarlar

| Dosya | Görev | İlişki |
|---|---|---|
| `README.md` | Proje kapsamı ve üst seviye çalıştırma sırası | Bu haritanın kısa sürümü |
| `src/utils/config.py` | Yol, ölçüm, bozulma, split ve CQR sabitleri | Hemen tüm scriptler okur |
| `requirements.txt` | Python paketleri | Çalışma ortamını kurar |

## Veri hazırlama

| Dosya | Görev | Çıktı / sonraki adım |
|---|---|---|
| `src/data_prep/load_herlev.py` | Herlev görüntüsünü ve resmi etiketi yükler; boolean nucleus/cytoplasm maskesi üretir; veri doğrular | `herlev_validation_summary.json`, eğitim verisi |
| `src/data_prep/group_split.py` | Hücre kimliği düzeyinde train/calibration/test bölmesi yapar | `herlev_group_split.json` |
| `src/data_prep/load_cx22.py` | Cx22 ZIP ve MATLAB/HDF5 görüntü/instance maskelerini okur | Cx22 manifest ve ShiftEval |
| `src/data_prep/cx22_bbox_crop.py` | Hedef instance merkezli, scale-normalized crop üretir | Cx22 kontrollü crop |
| `experiments/exp5_cx22_shifteval_manifest.py` | Outcome-blind Cx22 seçim ve shift etiket manifestini dondurur | `cx22_shifteval_manifest.csv` ve özeti |
| `experiments/exp5_cx22_*_qc.py` | Herlev/Cx22 crop ve overlay kalite kontrol görselleri | `results/figures/` |

## Segmentasyon ve ölçüm

| Dosya | Görev | Çıktı / sonraki adım |
|---|---|---|
| `src/segmentation/unet_model.py` | 3-kanallı giriş, 3 sınıflı U-Net yapısı | Eğitim ve inference |
| `src/segmentation/train_unet.py` | RGB hazırlama, padding, target oluşturma, Dice+CrossEntropy kaybı ve eğitim | `unet_best_trainval.pt`, `unet_trainval_training_log.json` |
| `src/measurements/morphometry.py` | N/C, dairesellik, göreli hata | Exp1, Exp2 ve Cx22 ölçümleri |
| `experiments/exp1_dice_correlation.py` | Herlev testte Dice–morfometri hata ilişkisi | `exp1_dice_morphometry_rows.json`, `exp1_checkpoint_summary.json` |

## Bozulma ve conformal zinciri

| Dosya | Görev | Çıktı / sonraki adım |
|---|---|---|
| `src/degradation/apply_degradations.py` | Dört bozulma ve deterministik gürültü tohumu | Exp2 varyantları |
| `src/conformal/split_conformal.py` | Nonconformity, q_hat, aralık, coverage | Global Exp2 |
| `experiments/exp2_marginal_coverage.py` | Herlev calibration/test varyantları ile global coverage | `exp2_marginal_rows.json`, `exp2_marginal_coverage*.json/csv` |
| `experiments/exp2_scale_split_inference.py` | Disjoint `unet_val` bölmesinde joint skor ölçekleri için inference | `exp2_scale_split_rows.json` |
| `experiments/exp2_strict_cellwise_coverage.py` | Beş tohumla hücre-başına-tek-varyant, failure-aware marjinal ve joint coverage | `exp2_strict_cellwise_*.json/csv` |
| `experiments/exp2_diagnostics.py` | Non-finite, clipping ve koşullu coverage tanıları | Exp2 diagnostics tabloları |
| `src/conformal/joint_conformal.py` | İki ölçüm için maksimum normalize hata ile joint split conformal | Strict Exp2/Exp3 |
| `experiments/exp3_joint_coverage.py` | Strict joint coverage giriş noktası | Strict Exp2 çıktıları |
| `src/conformal/mondrian_conformal.py` | Bozulma-türü/şiddet grubuna özgü q_hat | Exp3 |
| `experiments/exp3_mondrian_coverage.py` | Global ve Mondrian karşılaştırması | `exp3_global_vs_mondrian.csv` |
| `src/conformal/cqr.py` | Quantile-regression tabanlı conformal aralık | Exp4, yalnız Herlev |
| `experiments/exp4_cqr_coverage.py` | Global/Mondrian/CQR karşılaştırması | `exp4_cqr_coverage.csv`, `exp4_cqr_vs_mondrian_summary.json` |

Tamamlanmamış joint-CQR, adaptivity ve 2-vs-5-measurement taslakları aktif deney zincirine bağlı değildi; makaleye giden repo temiz tutulduğu için çıkarılmıştır.

## Cx22 ShiftEval zinciri

| Dosya | Görev | Çıktı / sonraki adım |
|---|---|---|
| `experiments/exp5_cx22_shifteval.py` | Dondurulmuş manifesti doğrular, 1320 RGB crop üzerinde inference ve ölçüm yapar | `exp5_cx22_shifteval_rows.csv`, `...summary.json`, `...strata.csv` |
| `experiments/exp5_cx22_mechanism_diagnostics.py` | Nucleus Dice–hata ve buffer-ratio açıklama analizi | `exp5_cx22_mechanism_diagnostics.json` |
| `experiments/exp5_cx22_joint_external.py` | Sabit strict Herlev joint calibration’ını Cx22’ye uygular | `exp5_cx22_joint_external_*.json/csv` |
| `experiments/exp5_cx22_scale_diagnostic.py` | Herlev ve Cx22 crop ölçeğini karşılaştırır | `exp5_cx22_scale_diagnostic.json` |
| `experiments/exp5_cx22_scale_normalized.py` | Multi-Test üzerinde yardımcı raw/normalized crop tanısı | `exp5_cx22_scale_normalized_*.json/csv` |
| `experiments/exp5_cx22_stage_a.py` | Yardımcı protokol metrik hesaplama fonksiyonları | Stage-A tabloları, Cx22 yardımcıları |
| `experiments/exp5_cx22_stage_b_lock.py` | Multi-Test partition düzeyi istatistik özeti | `exp5_cx22_stage_b_lock_summary.json` |
| `experiments/exp5_cx22_validation.py` | Cx22 uyumluluk kapısı ve Herlev ağır-shift fallback özeti | `exp5_cx22_compatibility_summary.json`, `exp5_herlev_heavy_shift_summary.csv` |

## Uçtan uca akış

`Herlev ham veri` → `load_herlev.py` → `group_split.py` → `train_unet.py` → `unet_best_trainval.pt` → `morphometry.py` → `exp1_dice_correlation.py` → `apply_degradations.py + exp2_marginal_coverage.py` → `exp2_scale_split_inference.py + exp2_strict_cellwise_coverage.py` → `exp3_joint_coverage.py`; ayrı karşılaştırma hattı `exp3_mondrian_coverage.py / exp4_cqr_coverage.py`.

Paralel dış değerlendirme hattı: `Cx22 ham arşivleri` → `load_cx22.py` → `cx22_bbox_crop.py + exp5_cx22_shifteval_manifest.py` → `exp5_cx22_shifteval.py` → `exp5_cx22_mechanism_diagnostics.py`.

Ana sonuçları okumak için önce `exp2_marginal_coverage_summary.json`, sonra `exp5_cx22_shifteval_summary.json`, sonra da ilgili rows/strata CSV dosyaları okunmalıdır.
