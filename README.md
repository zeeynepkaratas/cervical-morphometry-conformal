# Cervical Morphometry Conformal

Rahim ağzı hücre segmentasyonundan elde edilen **çekirdek/sitoplazma oranı** ve **çekirdek daireselliği** için belirsizlik analizi.

Araştırma sorusu: Segmentasyonun Dice skoru, morfometrik ölçümlerin güvenilirliğini ne kadar yansıtıyor ve görüntü bozulduğunda kalibre edilmiş aralıklar nasıl davranıyor?

## Durum ve kapsam

Herlev ana deneyleri, strict hücre-bazlı joint analiz ve Cx22 dış stres testi tamamlanmıştır. Kayıtlı sayısal tahminlerden ana sonuçlar yeniden hesaplanabilir. Bu paket klinik kullanım veya yeni, dokunulmamış bir doğrulama çalışması değildir.

- **Ölçümler:** N/C oranı (çekirdek alanı / çekirdek hariç sitoplazma alanı), çekirdek daireselliği.
- **Bozulmalar:** Gaussian blur, Gaussian noise, kontrast değişimi, düşük çözünürlük; her biri üç sabit şiddet.
- **Model:** RGB U-Net; en-boy oranını koruyan 128×128 giriş.
- **Herlev:** 495 eğitim, 55 doğrulama, 183 kalibrasyon, 184 test hücresi.
- **Cx22:** Pair 820 + Multi-Train 400 + Multi-Test 100; toplam 1.320 görüntü. Üçü tek pooled dış stres testidir.

Kapsam ve bozulma şiddetleri sonuçlara göre yeniden seçilmez. Cx22 model eğitiminde, model seçiminde veya kalibrasyonda kullanılmaz.

## Hızlı sonuç doğrulama

Python 3.12 ile, depo kökünde:

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python -m experiments.reproduce_results
```

Bu yol yalnız NumPy gerektirir; ham görüntü veya PyTorch gerektirmez. Altı girdi dosyasının SHA-256 özeti doğrulanır, Herlev'deki 15 ve Cx22'deki 20 tekrar satırının eski eşik/kapsama değerleri karşılaştırılır. Yeni çıktılar `results/reproduced/` içine yazılır; dondurulmuş referanslar korunur.

Model indirme, ham veri yerleşimi, eğitim ve görüntülerden yeniden hesaplama için [yeniden üretim rehberi](REPRODUCIBILITY.md).

## Ana sonuçlar

| Analiz | Sonuç |
| --- | --- |
| Herlev strict joint failure-aware coverage | Ortalama %88,59; beş varyant-seçim tekrarında %86,41–%90,22 |
| Cx22 strict joint coverage | Ortalama %87,67; beş tekrarda %85,23–%90,83 |
| Cx22 joint dairesellik aralığı genişliği | [0,1] alanına kırpılmış ortalama genişlik, tekrara göre 0,820–0,939 |

Hedef kapsama %90'dır. Aralık genişliği sonuçları **sonradan eklenen verimlilik tanılarıdır**; eşikleri veya kapsama sonuçlarını değiştirmez. Geniş aralıklarla yüksek kapsama elde etmek, tek başına kullanışlı belirsizlik tahmini demek değildir.

Kaynaklar: `results/tables/exp2_strict_cellwise_summary.json`, `exp5_cx22_joint_external_summary.json`, `joint_interval_efficiency_summary.json`.

## Yöntem ve yorum sınırları

- Aynı hücrenin 12 bozulmuş kopyası bağımsız örnek sayılmaz. Strict analiz her hücreden bir varyant seçer; beş tohum yeni model eğitimleri veya bağımsız test kümeleri değildir.
- Joint skorun ölçekleri ayrı `unet_val` bölmesinden sabitlenir. Geçersiz test tahminleri failure-aware coverage'da başarısız sayılır.
- Herlev test sonuçları geçmiş geliştirme kararlarında incelenmiştir. Strict yeniden analiz bu geçmişi silmez; mevcut sonuçlar dokunulmamış confirmatory doğrulama olarak sunulmaz.
- Cx22'de gerçek maske hedef hücrenin crop merkezini/ölçeğini belirler. Bu, otomatik hücre bulma veya deployment testi değildir; Cx22 için formal coverage garantisi yoktur.
- Kalibrasyonda sonlu olmayan skorların çıkarılması nedeniyle teorik hedef, geçerli skor koşullarından ve exchangeability varsayımından ayrı düşünülemez. Failure-aware kapsama ampirik olarak raporlanır.

## Projeyi okumak için

- [Amaç ve araştırma sorusu](results/docs/01_giris_ve_amac.md)
- [Conformal yöntem](results/docs/05_conformal_prediction.md)
- [Sonuçlar ve anlamları](results/docs/07_sonuclar_ve_anlamlari.md)
- [Metodolojik kararlar ve sınırlar](results/docs/08_metodolojik_kararlar_ve_sinirlar.md)
- [Dosya haritası](results/docs/09_dosya_haritasi.md)
- [Makaleye geçiş notları](results/docs/10_makaleye_gecis_notlari.md)
