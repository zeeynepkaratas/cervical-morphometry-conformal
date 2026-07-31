# 1. Giriş ve Amaç

Bu rehber, projeyi sıfırdan okumak içindir. Sonraki bölüm: [Veri setleri](02_veri_setleri.md).

## Proje hangi soruyu soruyor?

Rahim ağzı sürüntüsü (Pap smear) görüntülerinde hücrenin çekirdeği ve sitoplazması incelenir. Çekirdek, hücrenin genetik materyalini taşıyan merkezidir; sitoplazma ise çekirdeğin dışındaki hücre içi bölgedir. Bu yapıların boyutu ve şekli, hücre görünümünü sayısallaştırmak için kullanılabilir.

Bu proje tek cümlede şunu sorar: **Bir görüntü-segmentasyon modeli hücreyi iyi çizmiş görünse bile, bundan türetilen N/C oranı ve çekirdek daireselliği ölçümlerine ne kadar güvenebiliriz; bu güven görüntü bozulunca ve başka bir veri kaynağına geçince korunur mu?**

Bu önemlidir; çünkü bir görüntüde çizimin genel başarısı ile klinik olarak anlamlı sayısal ölçümün doğruluğu aynı şey değildir. Modelin çizdiği sınırdaki küçük bir hata, alan veya çevreye dayalı bir ölçümü belirgin değiştirebilir.

## Temel kavramlar

- **Segmentasyon:** Görüntünün her pikseline bir sınıf atama işlemidir. Bu projede sınıflar arka plan, cytoplasm-only (çekirdek hariç sitoplazma) ve çekirdektir.
- **Piksel:** Dijital görüntünün en küçük kare parçasıdır. Segmentasyonda her piksel için “bu piksel hangi sınıfa ait?” sorusu cevaplanır.
- **Maske:** Belirli bir bölgeyi işaretleyen piksel haritasıdır. Boolean maske yalnızca `0` (bölge dışında) ve `1` (bölgede) taşır.
- **Morfometri:** Biyolojik yapıların alan, şekil veya oran gibi sayısal özelliklerini ölçme işidir.
- **N/C oranı:** Çekirdek alanının, çekirdek hariç sitoplazma alanına oranıdır. Büyük bir oran, çekirdeğin sitoplazmaya göre büyük olduğunu ifade eder.
- **Circularity / dairesellik:** Bir şeklin daireye ne kadar benzediğini gösteren 0 ile 1 arasındaki ölçüdür. 1, ideal daireye en yakın değerdir.
- **Tahmin aralığı:** Tek bir sayı vermek yerine, gerçek değerin bulunmasının beklendiği alt ve üst sınır vermektir. Örneğin `[0.38, 0.52]` aralığı, tahminin belirsizliğini açıkça gösterir.

## Projenin mantığı

1. Herlev görüntülerinden çekirdek ve sitoplazmayı segmentleyen bir U-Net modeli eğitilir.
2. Modelin çizimlerinden N/C oranı ve dairesellik hesaplanır.
3. Görüntü bulanıklaştırıldığında, gürültülendirildiğinde, kontrastı değiştiğinde veya çözünürlüğü düşürüldüğünde ölçümlerin hata payı incelenir.
4. Conformal prediction adlı yöntemle, ölçüm tahminlerinin etrafına hedefi %90 olan aralıklar eklenir.
5. Aynı, Herlev üzerinde kalibre edilmiş aralıklar Cx22 adlı ayrı kaynaktaki görüntülere uygulanarak dış değerlendirme yapılır.

Bu sıralama, çizim kalitesi ile ölçüm güvenilirliğinin ayrı ayrı değerlendirilmesini sağlar. Model ve eğitim ayrıntıları için [Model eğitimi](03_model_egitimi.md), ölçümler için [Morfometrik ölçümler](04_morfometrik_olcumler.md) okunmalıdır.
