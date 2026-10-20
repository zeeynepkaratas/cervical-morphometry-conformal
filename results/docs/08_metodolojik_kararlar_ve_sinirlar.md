# 8. Metodolojik Kararlar ve Sınırlar

Önceki bölüm: [Sonuçlar](07_sonuclar_ve_anlamlari.md). Sonraki bölüm: [Dosya haritası](09_dosya_haritasi.md).

## Kilitli kararlar

1. **İki ana ölçüm:** N/C oranı ve çekirdek daireselliği. Bu odak, sonuçların gereksiz çok sayıda ölçüm testiyle dağılmasını önler.
2. **RGB U-Net:** Renk bilgisi korunur; görüntüler en-boy oranı korunarak 128×128’e hazırlanır.
3. **Hücre kimliği düzeyinde ayrım:** Aynı orijinal hücrenin eğitim, calibration ve test bölmelerine dağılması önlenir.
4. **Calibration yalnız Herlev’de:** q_hat yalnız Herlev calibration bölmesinden hesaplanır. Cx22’de q_hat yeniden ayarlansaydı, “Herlev’de öğrenilmiş belirsizlik Cx22’ye nasıl taşınıyor?” sorusu cevaplanamazdı.
5. **Cx22’de GT’nin sınırlı rolü:** Gerçek maskeler yalnız hedef instance’ın crop merkezi/ölçeği ile değerlendirme referansını belirler. RGB crop modelin tek girdisidir.
6. **Outcome-blind Cx22 manifest:** Örnek dahil etme ve shift grupları model Dice’ı, coverage’ı veya hata sonuçları görülmeden belirlenmiştir.
7. **Strict hücre-bazlı conformal:** Aynı hücrenin 12 bozulmuş kopyası klasik bağımsız conformal örnek sayısı olarak kullanılmaz. Beş tekrarda hücre başına tek varyant seçilir; joint skorun ölçekleri disjoint `unet_val` bölmesinden sabitlenir.

## Confirmatory ve exploratory ayrımı

**Confirmatory (önceden planlı) analiz**, yöntem ve karar kuralları sabitlendikten sonra dokunulmamış veride sınanır. Herlev U-Net, iki ölçüm ve dört bozulma başlangıçtaki planlı kapsamdır; ancak Herlev test sonuçları daha sonra yöntem kararlarında incelenmiştir. Bu nedenle mevcut Herlev sonuçları ve strict yeniden analiz, dokunulmamış confirmatory kanıt olarak sunulmaz. Planlı kapsam ile son değerlendirmedeki kanıtın statüsü farklıdır.

**Exploratory / post-hoc (sonradan açıklayıcı) analiz**, görülen bir örüntünün nedenini anlamak için sonradan yapılır. Cx22 pooled ShiftEval, scale-normalized instance-crop karşılaştırması, shift-stratifikasyonu, crowding karşılaştırmaları, mechanism/buffer-ratio analizi ve global-Mondrian-CQR karşılaştırması bu kategoride temkinli yorumlanmalıdır. Exploratory bulgu değerli olabilir; ancak önceden planlanmış kanıtla aynı kesinlikte sunulmaz.

Bu ayrım önemlidir çünkü sonucu gördükten sonra çok sayıda seçim yapmak, tesadüfi örüntü bulma olasılığını artırır. Dokümantasyon, manifest ve Bonferroni düzeltmesi bu riski görünür kılar.

## Bilinen sınırlamalar

- Herlev içindeki formal split-conformal hedefi, aynı veri üretim sürecine benzer örnekler için düşünülür. Cx22, farklı kaynak olduğu için Cx22’de formal coverage garantisi yoktur.
- Cx22 scale-normalized değerlendirmesi, hedef hücrenin yerini/ölçeğini gerçek maskeden alır. Bu bilinçli kontrollü bir protokoldür; gerçek zamanlı deployment (görüntü geldiğinde hedef hücrenin yerini otomatik bulma) testi değildir.
- Cx22 tek bağımsız dış kaynaktır; çok merkezli veya klinik düzeyde genelleme iddia edilmez.
- Circularity için bazı Herlev bozulmuş varyantlar finite ölçüm üretmemiştir; strict raporlar hem geçerli tahmin oranını hem de bu örnekleri başarısız sayan failure-aware coverage’ı verir.
- Normal yaklaşım tabanlı coverage testleri ve Bonferroni düzeltmesi, seçilmiş hipotez ailelerine uygulanmıştır; bunlar etki büyüklüğünün yerine geçmez.
- Cx22 mekanizma korelasyonları ilişki gösterir; bir değişkenin diğerine neden olduğunu kanıtlamaz.
- CQR aralıkları bazı koşullarda daralırken anlamlı conditional undercoverage üretmiştir. Bu nedenle “daha dar aralık her zaman daha iyi” sonucu çıkarılamaz.
- Herlev test bölmesi, ilk deneylerden sonra yöntem geliştirme kararlarında incelenmiştir. Strict re-analysis bağımlı varyant sorununu giderir; geçmiş test incelemesini geriye dönük olarak “dokunulmamış” hale getirmez. Bu nedenle adaptif/karşılaştırmalı sonuçlar exploratory olarak yazılmalıdır.

## Yeniden üretim ve aralık verimliliği

[Yeniden üretim rehberi](../../REPRODUCIBILITY.md), ham veri gerektirmeyen doğrulama ile görüntülerden yeni inference/eğitimi ayırır. Beş tohum, aynı hücreler üzerinde varyant seçimini tekrarlar; beş bağımsız eğitim veya beş bağımsız test kümesi değildir.

Joint genişlik ve dikdörtgen alanı sonradan eklenen tanılardır. Geniş aralıklar kapsama oranını artırabileceğinden yalnız coverage ile yöntem üstünlüğü iddia edilmez. Bu tanılar yeni eşik seçimi veya test sonuçlarına göre optimizasyon için kullanılmamıştır.

Kalibrasyonda sonlu olmayan skorlar dışlanır. Dolayısıyla geçerli skorlar üzerindeki split-conformal hedefi, geçersiz tahminleri de başarısız sayan failure-aware kapsama için otomatik bir teorik garanti değildir. Her ikisi ayrı raporlanmalıdır.
