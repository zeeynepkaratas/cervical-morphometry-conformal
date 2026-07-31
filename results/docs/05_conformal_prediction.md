# 5. Conformal Prediction

Önceki bölüm: [Morfometrik ölçümler](04_morfometrik_olcumler.md). Sonraki bölüm: [Cx22 dış değerlendirme](06_cx22_disaridan_degerlendirme.md).

## Temel fikir

**Conformal prediction**, bir modelin tek sayı tahmininin etrafında veriyle kalibre edilmiş bir belirsizlik aralığı kurma yöntemidir. Bu projede hedef coverage (kapsama) `%90`dır. %90 coverage demek, aynı koşullardaki çok sayıda yeni tahmin için gerçek değerin yaklaşık her 100 örneğin 90’ında tahmin aralığının içinde kalması hedeflenir.

Örnek: model N/C için `0.45` tahmin etmiş, aralık `[0.38, 0.52]` olabilir. Gerçek N/C `0.50` ise bu örnek covered (kapsanmış) sayılır; `0.60` ise sayılmaz.

## Split conformal nasıl çalışır?

**Split**, verinin bir kısmını model eğitimi için, ayrı bir kısmını ise belirsizlik aralığını ayarlamak için kullanmak demektir. Bu projede U-Net eğitiminden sonra 183 Herlev görüntüsünün calibration bölmesi ayrılır.

1. Calibration görüntülerinde model tahmini ile gerçek ölçüm arasındaki mutlak hata hesaplanır. Buna **nonconformity score** denir: tahminin gerçekle ne kadar uyuşmadığını gösteren sayı.
2. Bu hataların %90 hedefi için uygun sıralı değeri seçilir. Bu değer `q_hat` olarak adlandırılır. Basitçe, yeni tahminin iki yanına eklenecek “güven tamponu”dur.
3. Test tahmini için aralık `tahmin - q_hat` ile `tahmin + q_hat` olur. Ölçümün doğal sınırları varsa aralık bu sınırlarda kırpılır: circularity `[0, 1]`, N/C ise `[0, sonsuz)`.

Kod: `src/conformal/split_conformal.py`. `calibrate_split_conformal()` sonlu örneklem düzeltmeli sıralı quantile kuralını kullanır; `predict_interval()` aralığı üretir; `empirical_coverage()` gerçek test coverage’ını sayar.

## Bu projedeki q_hat değerleri

`results/tables/exp2_marginal_coverage_summary.json` dosyasında Herlev calibration’dan gelen değerler şöyledir:

| Ölçüm | q_hat |
|---|---:|
| N/C oranı | 0.49895 |
| Dairesellik | 0.25242 |

Bu değerler dört görüntü bozulması altında oluşturulan calibration varyantlarından hesaplanır. Bozulmalar Gaussian blur (bulanıklık), Gaussian noise (rastgele piksel gürültüsü), mean-pivot contrast change (ortalama parlaklık etrafında kontrast değişimi) ve low resolution (çözünürlüğü düşürüp geri büyütme) türleridir. Her türde üç şiddet vardır; her temiz görüntüden 12 varyant oluşur.

## Global, Mondrian ve CQR

**Global** yaklaşım, tüm bozulma türleri için tek q_hat kullanır. **Mondrian conformal**, her bozulma-türü/şiddet grubu için ayrı q_hat hesaplar; kod `src/conformal/mondrian_conformal.py` ve `experiments/exp3_mondrian_coverage.py` içindedir. **CQR (Conformalized Quantile Regression)**, önce koşula bağlı bir alt/üst aralık modeli öğrenir ve ardından calibration ile düzeltir; yalnızca Herlev hattında `src/conformal/cqr.py` ve `experiments/exp4_cqr_coverage.py` ile kullanılır. Cx22’de CQR eğitilmez veya kalibre edilmez.

## Strict cell-wise ve joint conformal

Eski all-variant grid, aynı hücrenin 12 bozulmuş kopyasını içerir. Bu kopyalar birbirinden bağımsız kabul edilmemelidir. `experiments/exp2_strict_cellwise_coverage.py`, her hücreden yalnız bir varyant seçer ve bunu beş sabit tohumla tekrarlar. Her tekrarda calibration 183, test 184 hücreden oluşur. Geçersiz bir ölçüm üretildiğinde örnek “uncovered” sayılır; ayrıca geçerli tahmin oranı ayrı raporlanır.

**Joint conformal**, iki ölçümün aynı hücrede birlikte kapsanmasını ister. `src/conformal/joint_conformal.py`, N/C ve dairesellik hatalarını ayrı, disjoint `unet_val` bölmesinden sabitlenen ölçeklerle normalize eder; iki değerin maksimumunu tek hücre skoru yapar. Calibration bu tek skordan `q_hat_joint` üretir. `experiments/exp3_joint_coverage.py` bu strict analizin giriş noktasıdır.
