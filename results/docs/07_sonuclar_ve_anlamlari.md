# 7. Sonuçlar ve Anlamları

Önceki bölüm: [Cx22 dış değerlendirme](06_cx22_disaridan_degerlendirme.md). Sonraki bölüm: [Metodolojik kararlar ve sınırlar](08_metodolojik_kararlar_ve_sinirlar.md).

Bu bölümdeki ana kaynaklar `results/tables/exp2_marginal_coverage_summary.json`, `results/tables/exp5_cx22_shifteval_summary.json`, `results/tables/exp5_cx22_shifteval_strata.csv` ve `results/tables/exp5_cx22_mechanism_diagnostics.json` dosyalarıdır.

## Herlev iç değerlendirme: dört bozulma birlikte

| Ölçüm | Test varyantı | q_hat | Coverage | Ortalama aralık genişliği |
|---|---:|---:|---:|---:|
| N/C oranı | 2208 | 0.49895 | %88.59 | 0.86185 |
| Dairesellik | 2196 finite | 0.25242 | %86.79 | 0.48628 |

Hedef %90’dır. Dolayısıyla N/C için %88.59, 100 test varyantının yaklaşık 89’unda gerçek N/C’nin aralıkta kaldığı; dairesellik için %86.79, yaklaşık 87’sinde kaldığı anlamına gelir. Dairesellikte 12 test varyantı finite ölçüm üretmediği için coverage 2196 finite örnek üzerinden hesaplanır; bu örnekler gizlice dahil edilmemiştir.

En zor örneklerden biri circularity + Gaussian noise + şiddet 30 hücresidir: global coverage %62.84’tür. Bu, global tek q_hat’in bazı bozulma koşullarında yetersiz genişlikte kalabildiğini gösterir.

## Cx22 pooled dış değerlendirme (n=1320)

| Ölçüm | Foreground Dice ile ilgili bağlam | Coverage | Ortalama mutlak hata | Ortalama aralık genişliği |
|---|---|---:|---:|---:|
| N/C oranı | Pooled foreground Dice %63.21 | %91.36 | 0.16021 | 0.63212 |
| Dairesellik | Pooled foreground Dice %63.21 | %65.83 | 0.20832 | 0.48601 |

**Foreground Dice**, arka plan dışındaki çekirdek ve sitoplazma maskelerinin gerçek maskelerle örtüşmesidir. Cx22’de N/C aralığı 1320 örneğin 1206’sını kapsar. Dairesellik aralığı ise yalnız 869 örneği kapsar. Bu, iki ölçümün aynı segmentasyon tahmini altında eşit biçimde güvenilir davranmadığını gösterir.

## Strict hücre-bazlı joint sonuç

`exp2_strict_cellwise_summary.json` beş sabit tohumla, her hücreden tek varyant seçilerek üretilmiştir. Herlev’de ortalama joint failure-aware coverage `%88.59`dur; tekrar aralığı `%86.41–%90.22`dir. Bu oran, iki ölçümün aynı hücrede birlikte kapsanmasını ve geçersiz tahminleri kapsanmamış saymayı gerektirir. Bu yüzden eski ayrı marjinal coverage sayılarından daha katı bir ölçüttür.

`exp5_cx22_joint_external_summary.json` aynı sabit Herlev calibration’ını Cx22’ye taşır. Cx22 pooled joint coverage ortalaması `%87.67`dir; beş tekrar aralığı `%85.23–%90.83`dir. Cx22 bu eşiği kalibre etmek için kullanılmadığından, sonuç dış stres testidir; Cx22 için formal coverage garantisi değildir.

## Kaynak-bölmesi düzeyindeki kırılım

| Cx22 bölmesi | n | Dice | N/C coverage | Dairesellik coverage |
|---|---:|---:|---:|---:|
| Pair | 820 | %62.93 | %92.20 | %64.27 |
| Multi-Train | 400 | %64.45 | %91.00 | %68.75 |
| Multi-Test | 100 | %60.50 | %86.00 | %67.00 |
| Pooled | 1320 | %63.21 | %91.36 | %65.83 |

Multi-Test satırı ayrı bir deney değil, pooled analizin bir kaynak-bölmesi kırılımıdır.

## Shift-stratifikasyon özeti

Dairesellik coverage’ı sekiz stratumun tamamında %90’ın altındadır ve 16-test Bonferroni düzeltmesinden sonra anlamlıdır: scale gruplarında %61.64–%70.43, nucleus-occupancy gruplarında %64.32–%67.27, crowding gruplarında %63.36–%67.52. **Bonferroni düzeltmesi**, çok sayıda hipotez testi yapılınca yanlış pozitif riskini azaltmak için p-değerini daha katı değerlendirme yöntemidir.

N/C için aynı sistematik undercoverage görülmez. Crowded grupta coverage %94.39’dur ve %90 hedeften anlamlı biçimde yüksektir; not-crowded grupta %86.92’dir. İki grubun N/C coverage farkı da istatistiksel olarak anlamlıdır. Bu, crowding’in N/C’deki farklı coverage davranışıyla ilişkili olduğunu gösterir; neden-sonuç kanıtı değildir. Dairesellikte crowded ve not-crowded gruplar arasındaki fark anlamlı değildir.

## Buffer-ratio mekanizma tanısı

**Buffer**, q_hat’in tahminin etrafına eklediği güven payıdır. Bunu, bir paketin çevresindeki koruyucu köpük gibi düşünebilirsiniz. Hata ne kadar değişkense aynı koruyucu pay her örneğe uygun olmayabilir.

| Ölçüm | q_hat / ortalama hata | q_hat / medyan hata | Nucleus Dice ile Spearman ilişkisi |
|---|---:|---:|---:|
| N/C oranı | 3.11x | 16.32x | -0.841 |
| Dairesellik | 1.21x | 1.85x | -0.616 |

**Spearman korelasyonu**, iki değerin sıralı olarak birlikte değişimini -1 ile +1 arasında ölçer. Negatif değer burada nucleus Dice yükseldikçe mutlak hatanın azalma eğiliminde olduğunu anlatır. N/C’de q_hat, tipik (medyan) hataya göre çok büyük bir tampon iken dairesellikte daha sınırlıdır. Bu, Cx22’de N/C coverage’ın yüksek ama dairesellik coverage’ın düşük olmasına yardımcı olan açıklayıcı bir bulgudur. Bu mekanizma analizi post-hoc’tur; modeli, veri seçimini veya kalibrasyonu değiştirmemiştir.

## Temkinli yorum

Sonuçlar, “yüksek Dice her morfometrik ölçüm için güvenilir aralık demektir” varsayımını desteklememektedir. Herlev’de hedefe yakın global coverage elde edilse de koşula bağlı kırılmalar vardır. Cx22’de ise N/C ve dairesellik farklı davranır: N/C dış stres testinde hedefe yakın görünürken dairesellik belirgin biçimde düşük coverage verir. Bu, daireselliğin sınır hatalarına daha hassas olabileceği açıklamasıyla uyumludur; klinik kullanıma hazır bir performans veya Cx22’ye formal coverage garantisi iddia edilmez.
