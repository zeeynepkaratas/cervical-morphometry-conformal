# 4. Morfometrik Ölçümler

Önceki bölüm: [Model eğitimi](03_model_egitimi.md). Sonraki bölüm: [Conformal prediction](05_conformal_prediction.md).

Kod: `src/measurements/morphometry.py`. Bu bölümdeki iki ana fonksiyon gerçek ve tahmin edilen maskelere aynı şekilde uygulanır.

## N/C oranı

Fonksiyon: `compute_nc_ratio(nucleus_mask, cytoplasm_mask)`.

Formül:

`N/C = nucleus_area / cytoplasm_only_area`

Buradaki **alan**, maskede `1` olan piksel sayısıdır. Payda, çekirdeği de içeren toplam hücre alanı değildir; yalnızca çekirdek dışındaki sitoplazmadır. Kod boş maske veya çekirdek-sitoplazma örtüşmesi görürse ölçüm üretmek yerine hata verir. N/C oranı 1’i aşabilir; bu nedenle üst sınırı 1’e kırpılmaz.

Bu ölçüm, çekirdeğin hücre içindeki göreli büyüklüğünü özetler. Segmentasyonda çekirdek veya sitoplazma alanındaki hata, oranı doğrudan etkiler.

## Çekirdek daireselliği

Fonksiyon: `compute_circularity(nucleus_mask)`.

Formül:

`dairesellik = 4 × pi × alan / çevre²`

**Çevre**, çekirdek maskesinin sınırı boyunca ölçülen uzunluktur. Kod OpenCV mevcutsa `cv2.contourArea` ve `cv2.arcLength` kullanır; bu, görüntü konturunu izleyen yöntemdir. Sonuç 0 ile 1 arasında tutulur. 1 ideal daireye yakınlığı, düşük değer ise uzamış veya düzensiz şekli ifade eder.

Dairesellik, sınırdaki küçük değişimlere alan oranından daha duyarlı olabilir; çünkü formülde çevrenin karesi vardır. Bu fark projenin sonuçlarında önemlidir.

## Hata nasıl hesaplanır?

Fonksiyon: `relative_error(predicted, reference)`. Tanısal Deney 1’de göreli hata şudur:

`|tahmin - referans| / (|referans| + 1e-6)`

Buradaki küçük `1e-6` sayısı, referans sıfıra çok yakınsa sıfıra bölmeyi önler. Conformal aralıklar ise bu göreli hatayı değil, tahmin ile referans arasındaki **mutlak farkı** kullanır. Bu ayrım, [Conformal prediction](05_conformal_prediction.md) bölümünde açıklanır.

Bu iki ölçüm seçilmiştir çünkü biri alan oranını (N/C), diğeri sınır şeklinin hassasiyetini (dairesellik) temsil eder.
