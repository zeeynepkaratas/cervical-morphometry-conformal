# 3. Model Eğitimi

Önceki bölüm: [Veri setleri](02_veri_setleri.md). Sonraki bölüm: [Morfometrik ölçümler](04_morfometrik_olcumler.md).

## U-Net nedir?

**U-Net**, görüntüyü küçük ayrıntılarla birlikte inceleyip her pikselin hangi sınıfa ait olduğunu tahmin eden bir sinir ağıdır. Bunu, bir mikroskop görüntüsünü kare kare boyayan bir yardımcı gibi düşünebilirsiniz: her kare için “arka plan”, “sitoplazma” veya “çekirdek” der. Ağın ilk kısmı görüntüdeki genel şekilleri öğrenir; ikinci kısmı bu bilgiyi kullanarak piksel düzeyinde ayrıntılı sınıf haritası üretir.

- Ağ tanımı: `src/segmentation/unet_model.py::UNet`
- Eğitim ve veri hazırlama: `src/segmentation/train_unet.py::train_unet`
- Değerlendirme: deney script'lerindeki Dice ve morfometri yardımcıları
- En iyi ağırlık dosyası (checkpoint): `results/unet_best_trainval.pt`
- Eğitim günlüğü: `results/tables/unet_trainval_training_log.json`

**Checkpoint**, eğitilmiş modelin öğrendiği sayısal ağırlıkların kaydedildiği dosyadır. İnference (eğitimden sonra yeni görüntüde tahmin alma) sırasında aynı U-Net yapısı oluşturulur, checkpoint yüklenir ve RGB görüntü modele verilir.

## Girdi ve hedef sınıflar

Model gri tonlama kullanmaz; üç kanallı **RGB** görüntü alır. RGB, kırmızı-yeşil-mavi renk kanallarını ifade eder. Görüntü değerleri 0–255 yerine 0–1 aralığına ölçeklenir. Görüntü, en-boy oranı korunarak ve gerektiğinde sıfır dolgusu eklenerek 128×128 piksele hazırlanır; bu, şekli yatay veya dikey sıkıştırmayı önler.

Hedef harita üç sınıftır: `0=arka plan`, `1=cytoplasm-only`, `2=nucleus`. Birleşim kuralında çekirdek önceliklidir; cytoplasm-only zaten çekirdeği içermez.

## Veri ayrımı ve sızıntı önleme

**Data leakage (veri sızıntısı)**, modelin veya karar sürecinin test edilecek örnek hakkında eğitim aşamasında bilgi edinmesidir. Bu, sonucun olduğundan iyi görünmesine yol açar. Bölme, bozulmuş kopyalar üretilmeden önce orijinal hücre kimliği düzeyinde yapılmıştır.

`results/tables/herlev_group_split.json` içindeki gerçek ayrım:

| Amaç | Görüntü sayısı | Kullanım |
|---|---:|---|
| `unet_train` | 495 | Model ağırlıklarını güncelleme |
| `unet_val` | 55 | Erken durdurma ve checkpoint seçimi |
| `calibration` | 183 | Yalnızca conformal aralık yarıçapını hesaplama |
| `test` | 184 | Son Herlev değerlendirmeleri |

`unet_train + unet_val = 550` görüntülük ana eğitim bölmesidir. Calibration ve test görüntülerine U-Net eğitimi sırasında dokunulmaz. Eğitim 50 epoch yürütülmüş; **epoch**, eğitim verisinin tamamı üzerinden bir geçiştir. En yüksek doğrulama foreground Dice değeri epoch 41’de yaklaşık `0.8619` olmuştur.

**Dice skoru**, model maskesi ile gerçek maskenin örtüşmesini ölçer: 1 tam örtüşme, 0 örtüşme yok demektir. Bu proje foreground Dice hesabında arka planı değil, hücre sınıflarını değerlendirir.
