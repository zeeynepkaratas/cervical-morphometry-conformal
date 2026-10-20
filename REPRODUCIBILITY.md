# Sonuçları yeniden üretme

Komutlar depo kökünde çalıştırılır. İki farklı işlem ayrılır: dondurulmuş sayısal tahminlerden sonuçları yeniden hesaplamak ve ham görüntülerden yeni model tahminleri üretmek.

## 1. Ham veri olmadan sonuç doğrulama

Python 3.12:

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python -m experiments.reproduce_results
```

Çıktılar `results/reproduced/` içindedir:

- `verification.json`: altı SHA-256 denetimi ve tarihsel sonuç eşleşmesi.
- `exp2_strict_cellwise_*`: hücre başına tek varyantla beş tekrar.
- `exp5_cx22_joint_external_*`: aynı Herlev kalibrasyonlarının Cx22'ye uygulanması.
- `joint_interval_efficiency_summary.json`: genişlik, [0,1] alanının tamamını kaplama oranı ve ortalama dikdörtgen alanı.

Herlev ara satırları `results/tables/exp2_marginal_rows.json.gz` içinde kayıpsız sıkıştırılmıştır. Bu dosya 183 kalibrasyon ve 184 test hücresinin 12'şer varyantını içerir; görüntü içermez. Sayısal satırların geçmiş commit ile ilişkisi ve dosya hash'leri `results/tables/frozen_analysis_inputs.json` içindedir.

Referans tablolar doğrudan karşılaştırılır; eşik, örnek sayısı veya kapsama farklıysa komut başarısız olur. Bu kontrol yeniden eğitim veya yeni bağımsız doğrulama değildir.

## 2. Dondurulmuş model ağırlıkları

Model dosyası herkese açık paylaşılmamıştır. Ham veri olmadan sayısal sonuç doğrulaması bu dosyayı gerektirmez. Görüntülerden inference için özgün yerel kopyayı kurup SHA-256 değerini doğrulayın:

```bash
python -m experiments.fetch_checkpoint --source /path/to/unet_best_trainval.pt
```

Beklenen SHA-256:

```text
91e8b3430ea8425634485bbfdd210c2b559a10f71a906498db1c4bdd089a835f
```

Hedef: `results/unet_best_trainval.pt`. Eşleşmeyen mevcut bir dosya üzerine yazılmaz. Yerel kopya yoksa 5. bölümdeki eğitim yoluyla yeni bir model üretilebilir; yeni modelin tarihsel sayılarla aynı sonucu vermesi garanti edilmez. Modelin herkese açık dağıtımı, sahibinin ayrıca açık paylaşım onayını beklemektedir. Bu nedenle paket görüntülerden tarihsel inference için tamamen bağımsız bir indirme paketi olarak sunulmaz.

## 3. Ham verileri edinme ve yerleştirme

Ham görüntüler bu depoda veya model sürümünde yeniden dağıtılmaz. Veri sahiplerinin erişim ve kullanım koşulları geçerlidir.

- **Herlev:** [MDE-Lab resmi indirme sayfası](https://mde-lab.aegean.gr/index.php/downloads/), yeni Pap-smear görüntü veritabanı (`smear2005.zip`). Görüntüleri ve `*-d.bmp` etiketlerini klasörleriyle `data/raw/herlev/` altına açın. Yükleyici alt klasörleri tarar; beklenen eşleşme 917 görüntü ve 917 maskedir.
- **Cx22:** [Resmi veri deposu](https://github.com/LGQ330/Cx22) üzerinden `Cx22-Pair.zip`, `Cx22-Multi-Train.zip`, `Cx22-Multi-Test.zip` etiket arşivlerini edinin.
- **Cx22 kaynak görüntüleri:** [LLPC/CCEDD resmi deposu](https://github.com/nachifur/LLPC). Bu projenin Python yolu `CCEDD.zip` içindeki `CCEDD/json.zip` arşivinin gömülü görüntü verisini okur. Yalnız `png.zip` indirmek bu yükleyici için yeterli değildir. Bu yolu kullanırken özgün arşiv yapısını koruyun; Cx22'nin resmi MATLAB üretim yolu ayrı bir seçenektir.

```text
data/raw/herlev/
    smear2005/.../*.bmp
    smear2005/.../*-d.bmp
data/raw/cx22/
    CCEDD.zip
    Cx22-Pair.zip
    Cx22-Multi-Train.zip
    Cx22-Multi-Test.zip
```

## 4. Görüntülerden yeniden hesaplama

Eğitim/inference için tam bağımlılıklar gerekir:

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; from src.data_prep.load_herlev import validate_herlev_dataset; print(validate_herlev_dataset(Path('data/raw/herlev')))"
```

`requirements-lock.txt` tarihsel doğrudan paket sürümleri kaydıdır; tam ortam ihracı değildir. Tarihsel manifestin üretildiği ortamda bazı eğitim paketleri kurulu değildi. Temiz ortamda model eğitiminin birebir tekrarlandığı iddia edilmez. Yeni bir inference/eğitim çalışmasında kullanılan gerçek paket sürümlerini ayrıca kaydedin; kilit dosyasını doğrulanmış ortam kaydı gibi yorumlamayın.

Dondurulmuş modeli yerel kopyadan doğruladıktan sonra Herlev calibration/test ve ayrı ölçek bölmesinin tahminleri:

```bash
python -m experiments.exp2_marginal_coverage
python -m experiments.exp2_scale_split_inference
python -m experiments.exp2_strict_cellwise_coverage
python -m experiments.exp5_cx22_shifteval
python -m experiments.exp5_cx22_joint_external
```

**Bu görüntü komutları `results/tables/` altındaki deney çıktılarını yeniden yazar.** Bunları ayrı bir çalışma kopyasında çalıştırın. İlk komut sıkıştırılmış Herlev satırlarını da günceller. Yeni çıktılar farklıysa dondurulmuş hash kontrolünün başarısız olması beklenir; referans hash'lerini yalnız testi geçirmek için değiştirmeyin. Cx22 manifestinde yalnız Windows CRLF/Linux LF farkı normalize edilir; seçim içeriği değişirse doğrulama durur.

## 5. Modeli baştan eğitme

Eski eğitim dosyasının boş giriş noktası yerine açık bir komut vardır:

```bash
python -m experiments.train_frozen_split --raw-dir data/raw/herlev --device cpu --epochs 50
```

495 eğitim ve 55 doğrulama ID'si kayıtlı split'ten alınır. Calibration/test ile kesişim varsa eğitim başlamaz. Yeni ağırlık ve günlükler `results/retraining/` içine yazılır; dondurulmuş model korunur. Yeni modeli değerlendirmek için deney fonksiyonlarına yeni checkpoint yolunu açıkça verin. Yeniden eğitim çıktılarının tarihsel checkpoint ile bit düzeyinde aynı olması beklenmez.

## İstatistiksel yorum

Kapsama ile aralık genişliği birlikte değerlendirilir. Dairesellik için tüm [0,1] alanını veren aralık, geçerli referanslarda %100 kapsama sağlayan fakat ayırt edici olmayan basit bir karşılaştırmadır. N/C üstten sınırlı olmadığından aynı sonlu genişlik karşılaştırması joint aralık için geçerli değildir.

Mevcut Herlev testi geçmişte incelendiğinden çalışma keşifsel olarak çerçevelenir. Cx22 known-localisation dış stres testidir. Yeni confirmatory kanıt gerekiyorsa yöntem sabitlendikten sonra dokunulmamış bir veri kaynağı gerekir; yazılım düzeltmesi bunun yerine geçmez.
