# 6. Cx22 Dışarıdan Değerlendirme

Önceki bölüm: [Conformal prediction](05_conformal_prediction.md). Sonraki bölüm: [Sonuçlar ve anlamları](07_sonuclar_ve_anlamlari.md).

## Neden “dış” değerlendirme?

Cx22 görüntüleri, Herlev üzerinde eğitilen U-Net’in eğitiminde, checkpoint seçiminde veya conformal calibration’ında kullanılmamıştır. Bu nedenle Cx22, modelin tamamen yeni bir görüntü kaynağındaki davranışını sınar. Ancak Herlev q_hat değerleri Cx22’de yeniden kalibre edilmediği için, Cx22 coverage sonuçları Cx22 için resmi sonlu-örneklem garantisi değildir; Herlev-kalibreli aralıkların dış stres testidir.

Ana kod: `experiments/exp5_cx22_shifteval.py`. Hazır, outcome-blind (model sonuçlarını görmeden dondurulmuş) seçim manifesti: `results/tables/cx22_shifteval_manifest.csv`.

## Çok hücreli görüntüden hedef instance seçimi

Cx22 görüntülerinde birden fazla hücre olabilir. **Instance**, aynı sınıfa ait tek bir ayrı nesnedir; örneğin iki ayrı çekirdek iki ayrı nucleus instance’ıdır. U-Net ise instance kimliği değil semantik sınıf tahmin eder. Bu nedenle tek bir hedef hücreyi tutarlı biçimde seçmek gerekir.

Seçim kuralı, model tahmininden önce çalışır:

1. Her görüntüde nucleus ve cytoplasm instance maskeleri MATLAB/HDF5 etiketlerinden okunur.
2. Bir nucleus instance’ı, bir cytoplasm-only instance’ının içinde en az `%50` oranında bulunuyorsa bu ikisi aday çift olur.
3. Adaylar arasından cytoplasm-only alanı en büyük olan çift seçilir.
4. Alan eşitse önce cytoplasm instance indeksi, sonra nucleus instance indeksi en küçük olan seçilir.
5. Seçilen gerçek maske yalnızca crop merkezini ve crop ölçeğini belirler. U-Net’e yalnız RGB crop verilir; gerçek maske tahmine yol göstermez ve ölçümde yalnız referans olarak kullanılır.

Bu kural `experiments/exp5_cx22_shifteval_manifest.py` ve `src/data_prep/cx22_bbox_crop.py` içinde uygulanmıştır. Manifest tüm 1320 adaydan 1320’sini içerir; dışlama yoktur.

## n=1320 pooled yapı

**Pooled**, üç resmi Cx22 kaynak bölmesini tek analiz havuzunda birleştirmek demektir: Pair `820` + Multi-Train `400` + Multi-Test `100` = `1320`. Multi-Test n=100, ayrı “ilk aşama” değildir; pooled sonucun içinde kaynak-bölmesi kırılımıdır. Bu yapı daha büyük örneklem sunar ve bölmeler arasındaki farkı ayrıca görünür kılar.

## Shift-stratifikasyon nedir?

**Stratifikasyon**, sonuçlara bakarak örnek seçmek değildir. Önceden belirlenmiş ölçülebilir özelliklere göre tüm havuzu gruplara ayırıp, her grupta sonucu ayrı raporlamaktır. Cx22 manifesti model çıkışlarını görmeden üç eksen tanımlar:

- **Scale shift:** Ham crop içindeki tüm hücre alanı oranı. Düşük/orta/yüksek üçte bir grupları vardır. Örnek: hücre çerçevenin küçük kısmını dolduruyorsa düşük scale grubundadır.
- **Nucleus occupancy:** Çekirdek alanının hedef hücre alanına oranı. Yine düşük/orta/yüksek üçte bir grubu vardır. Bu, çekirdeğin hücreye göre ne kadar yer kapladığını gösterir.
- **Crowding:** Scale-normalized crop içinde birden fazla nucleus adayı varsa `crowded`, yalnız hedef çekirdek varsa `not_crowded` olur. Bu, görüntüde komşu hücre etkisini temsil eder.

Eksenler ayrı ayrı analiz edilir; küçük hücreler oluşmaması için eksenler çaprazlanmaz. Shift kırılımları `results/tables/exp5_cx22_shifteval_strata.csv` içindedir.
