 Bank Cleora Embedding Generator

 Bağımlılıkları yükle
pip install -r requirements.txt
pip install adjustText   Grafik etiketlerini düzeltmek için

Bu doküman, müşterileri, ürünleri ve davranışları (event) ortak bir vektör uzayına gömmek için kullanılan Cleora scriptini (`bank_cleora.py`) açıklar.

  Amaç
`generate_bank_data.py` tarafından üretilen hyperedge verisini alıp, Müşteri-Ürün-Event ilişkilerini koruyan yüksek boyutlu (-dim) vektörler üretmektir. Bu vektörler daha sonra EMDE (sketch generation) aşamasında girdi olarak kullanılır.

---

  Hızlı Başlangıç

```bash
 Sanal ortamı aktif et
source venv/bin/activate

 Embedding üretimini başlat
python bank_cleora.py
```

İşlem yaklaşık - dakika sürer. İki farklı "yürüyüş uzunluğu" (Walk  & ) için ayrı ayrı embedding üretir.

---

 ️ Teknik Detaylar

 . Girdi Formatı
Script, `data/cleora_hyperedges.txt` dosyasını okur. Her satır bir "hiper-kenar"dır:
```text
C checking_account login_mobile transfer_eft
```
Bu yapı, `C` adlı müşterinin `checking_account` ile ve `login_mobile` eylemi ile ilişkili olduğunu grafta tanımlar.

 . Algoritma (Cleora)
Cleora, yinelemeli Markov Zinciri (Markov Chain) yayılımı kullanır:
.  Başlangıç (Initialization): Her entity (Müşteri, Ürün, Event) için deterministik,  boyutlu rastgele bir vektör oluşturulur.
.  Yayılım (Propagation):
    - Her entity'nin vektörü, bağlı olduğu diğer entity'lerin vektörlerinin ortalamasıyla güncellenir.
    - Bu işlem `num_walks` kadar tekrarlanır.
.  Normalizasyon: Her adımda vektörlerin L normu alınır.

 . Konfigürasyon
Script içinde şu parametreler sabittir:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `EMBED_DIM` |  | Vektör boyutu. Monad-EMDE için yüksek boyut tercih edildi. |
| `WALK_OPTIONS` | [, ] | Kaç adım komşuluğa bakılacağı. =Direkt komşular, =Dolaylı ilişkiler. |

---

  Çıktılar

Script, `data/embeddings/` klasörüne şu dosyaları kaydeder:

 Embedding Dosyaları (.npz)
- `cleora_embeddings_walk.npz`: Sadece . dereceden ilişkileri içeren vektörler.
- `cleora_embeddings_walk.npz`: . dereceye kadar yayılmış (daha zengin) vektörler.

Bu dosyalar `bank_emde_session.py` tarafından okunur.

 Görselleştirmeler (t-SNE)
Kalite kontrolü için her bir konfigürasyon (Walk  ve ) için  adet t-SNE grafiği üretilir:

| Dosya Adı | Açıklama |
|-----------|----------|
| `tsne_all_entities_...` | Müşteriler (Mavi) ve Ürünler/Eventler (Kırmızı) aynı uzayda nasıl duruyor? |
| `tsne_customers_segment_...` | Müşteriler segmentlerine (Mass, Private vb.) göre ayrışıyor mu? |
| `tsne_customers_churn_...` | Churn eden müşteriler (Kırmızı) ayrı bir küme oluşturuyor mu? |

---

  İdeal Sonuç Nasıl Görünmeli?

İyi bir Cleora çıktısında şunları görmelisiniz:

.  Segment Ayrımı: `tsne_customers_segment` grafiğinde "Private" ve "Mass" müşterileri belirgin şekilde ayrılmalı veya en azından farklı yoğunluk merkezlerine sahip olmalıdır.
.  Churn Ayrımı: Churn eden müşteriler (kırmızı noktalar) grafiğin belli bölgelerinde (düşük aktivite kümeleri veya yüksek şikayet kümeleri) yoğunlaşmalıdır. Rastgele dağılmışsa sinyal zayıf demektir.
.  Ürün/Event Konumu: `tsne_all_entities` grafiğinde, benzer ürünler (örn: `credit_card_platinum` ve `luxury_car_loan`) birbirine yakın durmalıdır.

---

  Sonraki Adım
Bu script tamamlandığında, `bank_emde_session.py` scriptini çalıştırarak bu vektörlerden "Müşteri Eskizleri" (Customer Sketches) oluşturabilirsiniz.
