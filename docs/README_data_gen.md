 Bank Churn Sentetik Veri Üreticisi v (Realistic Mode)

Bu proje, banka müşteri davranışlarını ve churn (müşteri kaybı) sinyallerini simüle eden gerçekçi bir sentetik veri üreticisidir. Üretilen veri, Cleora Graph Embedding ve EMDE (Efficient Manifold Density Estimator) modellerini eğitmek için optimize edilmiştir.

  Amaç
Gerçek bir banka verisindeki "gürültülü" ve "istatistiksel" churn sinyallerini simüle etmektir. Modelin "kopya çekmesini" (churn-only eventler veya explicit etiketler yoluyla) engelleyerek, davranışsal kalıpları (behavioral drift) öğrenmesini zorlar.

  Hızlı Başlangıç

```bash
 Sanal ortamı aktif et
source venv/bin/activate

 Veriyi üret (Yaklaşık - dakika sürer)
python generate_bank_data.py
```

---

 🏗 Veri Üretim Mantığı

Veri üretimi  aşamadan oluşur:

 . Müşteri Segmentasyonu (`generate_customers`)
Müşteriler  ana segmente ayrılır. Her segmentin churn olasılığı ve aktivite seviyesi farklıdır:

| Segment | Dağılım | Aktivite Çarpanı | Churn Olasılığı (Base) |
|---------|---------|-------------------|------------------------|
| Mass | % | .x | % |
| Affluent | % | .x | % |
| Private | % | .x | % |
| Business | % | .x | % |

Her müşteri yaratılırken `is_churning` bayrağı bu olasılıklara göre atanır.

 . Ürün Sahipliği (`assign_products`)
Müşterilere segmentlerine uygun ürünler atanır (Örn: `private` müşteriye `bonds`, `mass` müşteriye `personal_loan`).

 . Olay (Event) Simülasyonu (`generate_events`)
En kritik aşama burasıdır. Churn sinyalleri burada "davranışsal gürültü" olarak eklenir.

Churn Sinyalleri (Deterministik DEĞİL, İstatistiksel):
.  Aktivite Düşüşü: Churn eden müşterilerin işlem frekansı, işlem türüne göre `.x` ile `.x` arasına düşürülür. Ancak bu bir kural değil, dağılımdır. (Gürültü faktörü: `random(., .)`).
.  Kanal Değişimi: Memnuniyetsiz müşteriler şubeye daha az uğrayıp, çağrı merkezini daha sık arayabilir.
.  Şikayet Artışı: Churn edenlerin şikayet etme olasılığı (`churn_multiplier: .`) çok daha yüksektir, ancak her churn eden şikayet etmez (%'u sessizce gider). Retained olanların da %'i "rastgele" şikayet eder.

Not: `account_close_inquiry` gibi "churn-only" eventler gerçekçilik adına devre dışı bırakılmıştır. Model sadece işlem sıklığındaki ve türündeki değişimi analiz etmelidir.

 . Cleora Hyperedge Oluşturma (`generate_cleora_hyperedges`)
Graph embedding için veri "hiper-kenar" (hyperedge) formatına dönüştürülür.
Format: `customer_id` + `ürünler` + `son_eventler`

Örnek:
```text
C checking_account credit_card_gold login_mobile transfer_eft card_transaction_pos
```

---

  Çıktı Dosyaları

Script `data/` klasörüne şu dosyaları yazar:

| Dosya | Boyut (Tahmini) | İçerik |
|-------|-----------------|--------|
| `bank_customers.csv` | ~ KB | `customer_id`, `segment`, `age`, `risk_score`, `tenure`, `churn_label` |
| `bank_products.csv` | ~ MB | Müşteri-Ürün eşleşmeleri ve edinim tarihleri |
| `bank_events.csv` | ~ MB | Yaklaşık K - M satır işlem logu (`timestamp`, `event_type`, `channel`, `amount`) |
| `cleora_hyperedges.txt` | ~ MB | Cleora eğitimi için graph verisi |

---

  Örnek İstatistikler
(, Müşteri için ortalama değerler)

- Churn Oranı: ~%.
- Ortalama Event (Retained): ~ event/ay
- Ortalama Event (Churn): ~ event/ay (Belirgin bir sinyal var ama gürültülü)
- Churner Recall (Model ile): ~% (Bu veri setiyle eğitilen iyi bir modelin başarısı)

 🛠 Özelleştirme

`generate_bank_data.py` içindeki şu sabitleri değiştirerek veriyi modifiye edebilirsiniz:

```python
NUM_CUSTOMERS = _  Müşteri sayısı
DAYS =                Simülasyon süresi
SEGMENTS = {...}        Segment tanımları
EVENTS = {...}          Event frekansları ve çarpanları
```

 ️ Önemli Not
Bu veri seti, gerçek hayattaki "Sessiz Churn" (Silent Churn) problemini simüle eder. Churn eden müşteri "Ben gidiyorum" demez (account close event yok), sadece sessizleşir (activity reduction) ve belki bir kez şikayet eder. Modelin başarısı bu sessizliği duymasına bağlıdır.
