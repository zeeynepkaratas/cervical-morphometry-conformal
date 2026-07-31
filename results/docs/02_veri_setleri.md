# 2. Veri Setleri

Önceki bölüm: [Giriş ve amaç](01_giris_ve_amac.md). Sonraki bölüm: [Model eğitimi](03_model_egitimi.md).

## Herlev: ana eğitim ve kalibrasyon kaynağı

**Herlev**, rahim ağzı hücre görüntülerinden oluşan bir veri setidir. Repo doğrulamasında 917 görüntü ve bunların 917 maskesi eşleşmiştir. Her görüntü için resmi etiket maskesi, arka plan ile çekirdek ve sitoplazma etiketlerini içerir. Yükleyici bu etiketten iki ayrı boolean maske üretir: nucleus ve cytoplasm-only.

- Ham klasör: `data/raw/herlev/`
- Görüntü/maskeyi yükleyen kod: `src/data_prep/load_herlev.py`
- Doğrulama kaydı: `results/tables/herlev_validation_summary.json`
- Doğrulama sonucu: 917/917 eşleşme, maske değerleri her iki maske için `[0, 1]`, sorun listesi boş.

Herlev, modelin öğrenmesi, conformal kalibrasyon ve iç test için kullanılır. Bu üç kullanım birbirinden ayrılmış örneklerle yapılır; ayrıntı için [Model eğitimi](03_model_egitimi.md) ve [Conformal prediction](05_conformal_prediction.md) bölümlerine bakın.

## Cx22: dış değerlendirme kaynağı

**Cx22**, birden fazla hücre içerebilen rahim ağzı hücre görüntüleri ve örnek-bazlı çekirdek/sitoplazma instance maskeleri içeren ayrı bir kaynaktır. Bu projede Cx22 üzerinde model eğitimi, model seçimi veya conformal kalibrasyon yapılmamıştır. Bu yüzden Cx22 sonucu, modelin yeni bir kaynağa karşı davranışını inceleyen dış stres testidir.

- Ham klasör: `data/raw/cx22/`
- Resmi arşivler: `Cx22-Pair.zip`, `Cx22-Multi-Train.zip`, `Cx22-Multi-Test.zip`
- Cx22 yükleme ve MATLAB/HDF5 maske okuma kodu: `src/data_prep/load_cx22.py`
- Cx22 örnek/maske manifesti: `results/tables/cx22_shifteval_manifest.csv`

Cx22 değerlendirmesinde üç resmi kaynak bölmesi birlikte kullanılır: Pair 820, Multi-Train 400 ve Multi-Test 100 görüntü; toplam **n=1320**. Multi-Test `n=100`, ayrı veya önceki bir deney değildir; pooled n=1320 analizinin kaynak-bölmesi düzeyindeki kırılımıdır.

## Neden iki veri seti?

Tek bir veri setinde iyi sonuç almak, aynı yaklaşımın başka bir görüntü kaynağında aynı şekilde çalışacağını kanıtlamaz. Boyama rengi, hücre ölçeği, hücre yoğunluğu, kesme biçimi ve etiketleme geleneği değişebilir. Bu nedenle Herlev modeli kurmak ve kalibre etmek için; Cx22 ise **yeniden eğitim veya yeniden kalibrasyon yapılmadan** bu aktarımın ne kadar zor olduğunu görmek için kullanılır.

Cx22’de hedef hücre seçme ve pooled değerlendirme mantığı [Cx22 dış değerlendirme](06_cx22_disaridan_degerlendirme.md) bölümünde anlatılır.
