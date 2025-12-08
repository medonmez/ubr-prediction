 Bank EMDE Session Sketch Generator

Bu doküman, Cleora embedding'lerinden zaman ağırlıklı ve yoğunluk tabanlı (density-based) müşteri temsilleri (sketches) üreten `bank_emde_session.py` scriptini açıklar.

  Amaç
Ham olay (event) vektörlerini, bir müşterinin tüm geçmişini özetleyen sabit boyutlu (-dim) tek bir vektöre dönüştürmektir. Bu vektör, Foundation Model'in girdisi olacaktır.

---

  Hızlı Başlangıç

```bash
 Sanal ortamı aktif et
source venv/bin/activate

 Sketch üretimini başlat
python bank_emde_session.py
```

İşlem - dakika sürebilir. Cleora embedding'lerinin (`data/embeddings/`) önceden üretilmiş olması gerekir.

---

  Algoritma ve Mantık

EMDE (Efficient Manifold Density Estimator), klasik "ortalama alma" (mean pooling) yönteminden çok daha gelişmiş bir yöntemdir. Müşterinin davranış uzayındaki dağılımını saklar.

 . Parametreler (Neden  Boyut?)
- N =  (Subspaces): Vektör uzayı rastgele  alt uzaya bölünür.
- K =  (Bins): Her alt uzay $^ = $ parçaya bölünür.
- Sonuç: $ \times  = $ boyutlu bir sparse vektör.

 . Zaman Çürümesi (Time Decay)
Müşteri davranışı zamanla değişir. Geçmiş olayların etkisi, bugüne yaklaştıkça artmalı, eskidikçe azalmalıdır.
Formül: $Weight = e^{-\lambda \times \text{gün}}$
- Lambda: . (Yaklaşık . alpha'ya denk gelir)
- Etki:  gün önceki bir işlem %, dünkü işlem % etkiye sahiptir.

 . Past/Future Ayrımı (Self-Supervised Learning için)
Modeli eğitmek için veriyi zamana göre böleriz:
- Past UBR (Girdi): İlk - gün. Model bunu görüp geleceği tahmin etmeye çalışır.
- Future UBR (Hedef): Son - gün. Modelin tahmin etmesi gereken "gelecek davranışı"dır.

---

  Çıktılar

Script, `data/emde/` klasörüne `.npz` formatında şu matrisleri kaydeder:

| Değişken Adı | Boyut | Açıklama |
|--------------|-------|----------|
| `past_sketches` | (, ) | Müşterinin ilk  günlük davranışı (Input) |
| `future_sketches` | (, ) | Müşterinin son  günlük davranışı (Target) |
| `portfolio_sketches`| (, ) | Müşterinin sahip olduğu ürünlerin özeti (Static Input) |
| `churn_labels` | (,) | Churn durumu ( veya ) |

Bu dosyalar `future_ubr_ffn.py` (Foundation Model) ve `churn_prediction_finetune.py` (Churn Model) tarafından kullanılır.

---

  Görselleştirmeler

Script ayrıca `data/emde/` içinde t-SNE grafikleri (`emde_session_tsne_walk...png`) oluşturur. Bu grafiklerde:
.  Past UBR: Müşterilerin geçmiş davranışlarına göre nasıl kümelendiği.
.  Future UBR: Gelecekteki davranışların (tahmin hedefinin) nasıl dağıldığı 
görülebilir.
