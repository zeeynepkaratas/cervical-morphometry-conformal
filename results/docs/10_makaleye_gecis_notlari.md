# 10. Makaleye Geçiş Notları

Önceki bölüm: [Dosya haritası](09_dosya_haritasi.md). Başlangıca dön: [Giriş ve amaç](01_giris_ve_amac.md).

Bu belge deneysel repo ile makale bölümleri arasında köprü kurar. Sayılar için her zaman kaynak JSON/CSV dosyasına dönülmelidir; bu belge yeni sonuç üretmez.

## Introduction (Giriş)

- Rahim ağzı hücre görüntülerinde morfometrik ölçümlerin rolü: [Giriş ve amaç](01_giris_ve_amac.md).
- Ana problem: segmentasyon Dice’ı yüksek görünse bile ölçüm hatası ve belirsizliği ayrı değerlendirilmelidir.
- Araştırma sorusu ve iki ölçümün seçilme gerekçesi: [Morfometrik ölçümler](04_morfometrik_olcumler.md).

## Methods (Yöntemler)

- Veri kaynakları, görüntü sayıları ve ayrım: [Veri setleri](02_veri_setleri.md), `results/tables/herlev_group_split.json`.
- U-Net, RGB giriş, 128×128 padding, kayıp fonksiyonu, checkpoint seçimi: [Model eğitimi](03_model_egitimi.md), `src/segmentation/train_unet.py`.
- N/C ve dairesellik formülleri: [Morfometrik ölçümler](04_morfometrik_olcumler.md), `src/measurements/morphometry.py`.
- Dört bozulma türü ve şiddetleri: `src/degradation/apply_degradations.py`, `src/utils/config.py`.
- Global split conformal, finite-sample q_hat kuralı ve hedef %90: [Conformal prediction](05_conformal_prediction.md), `src/conformal/split_conformal.py`.
- Hücre-başına-tek-varyant strict protocol, failure-aware coverage ve joint split conformal: `experiments/exp2_strict_cellwise_coverage.py`, `src/conformal/joint_conformal.py`.
- Cx22 instance seçimi, scale-normalized known-localisation protokolü, pooled n=1320 ve pre-inference manifest: [Cx22 dış değerlendirme](06_cx22_disaridan_degerlendirme.md).
- Çoklu test düzeltmesi: Cx22 within-stratum ailesi 16 test, crowding between-group ailesi 2 testtir.

## Results (Bulgular)

- Herlev Exp1 Dice–morfometri ilişkisi: `exp1_checkpoint_summary.json` ve `exp1_dice_morphometry_rows.json`. Circularity Spearman sonucu finite-only `n=183` üzerinden raporlanmalıdır.
- Herlev global coverage ve q_hat: `exp2_marginal_coverage_summary.json`.
- Strict marjinal/joint coverage: `exp2_strict_cellwise_summary.json`; bu, bağımlı varyant havuzlamasına karşı ana sağlamlık analizi olarak sunulmalıdır.
- Global/Mondrian/CQR karşılaştırması: `exp3_global_vs_mondrian.csv`, `exp4_cqr_vs_mondrian_summary.json`. CQR’nin bazı koşullarda daha dar ama undercoverage üreten bir takas olduğu açık yazılmalıdır.
- Cx22 pooled ana sonuç ve partition breakdown: `exp5_cx22_shifteval_summary.json`.
- Shift-stratifikasyonu: `exp5_cx22_shifteval_strata.csv`.
- Mekanizma tanısı: `exp5_cx22_mechanism_diagnostics.json`; post-hoc olarak etiketlenmelidir.

## Discussion (Tartışma)

- Ana yorum: Dice, morfometrik güvenilirliğin tek başına yeterli özeti değildir.
- N/C ile circularity’nin farklı coverage davranışı; daireselliğin sınır hassasiyetine ilişkin ihtiyatlı yorum.
- Global aralığın koşula göre yetersiz kalabildiği ve adaptif yöntemlerin genişlik-coverage takası.
- Cx22’nin aynı aralıklarla yapılan dış stres testi olduğu; formal Cx22 coverage garantisi olmadığı.

## Limitations (Sınırlamalar)

[Metodolojik kararlar ve sınırlar](08_metodolojik_kararlar_ve_sinirlar.md) bölümündeki tüm maddeler kullanılabilir. Özellikle Cx22’nin known-localisation crop protokolü, tek dış kaynak, finite circularity vakaları ve exploratory analiz ayrımı açıkça yazılmalıdır.

## Açık karar noktaları

Kod ve sonuç dosyaları açısından yeni bir deneysel zorunluluk görünmüyor. Makale yazımı öncesi verilecek editoryal kararlar şunlardır:

1. Hedef dergi/konferansın double-blind olup olmadığı ve buna göre anonim repo/paket stratejisi.
2. CQR’nin ana metinde mi, ek materyalde mi sunulacağı. Sonuçta bazı koşullarda undercoverage ürettiği için “varsayılan en iyi yöntem” olarak yazılmamalıdır.
3. Cx22 yardımcı raw-vs-scale-normalized tanısının ana metinde kısa bir kontrollü protokol olarak mı, ek materyalde mi yer alacağı.
4. Bozulma şiddetleri için kullanılacak literatür kaynakları ve veri setlerinin resmi atıfları.
5. Tarihsel olarak incelenmiş Herlev test sonuçlarının exploratory olarak nasıl çerçeveleneceği; yeni/prospektif bir doğrulama veri kaynağı elde edilirse confirmatory testin nasıl ayrılacağı.

Bu kararlar, mevcut sayıları veya kod mantığını değiştirmeyi gerektirmez; makalenin kapsamını ve sunumunu belirler.
