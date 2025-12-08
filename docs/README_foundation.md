 Monad-EMDE Foundation Model

Bu doküman, müşterinin gelecekteki davranış yoğunluğunu (Future UBR) tahmin eden Foundation Model'in (`future_ubr_ffn.py`) teknik detaylarını ve mimari kararlarını açıklar.

  Amaç
Modelin amacı, bir müşterinin geçmiş davranışlarını (Past UBR) ve sahip olduğu ürünleri (Portfolio) alarak, gelecekteki davranış dağılımını (Future UBR) tahmin etmektir. Bu, "Self-Supervised Learning" (Kendi Kendine Eğitilen) bir görevdir; etiketler verinin kendisinden gelir.

---

  Hızlı Başlangıç

```bash
 Sanal ortamı aktif et
source venv/bin/activate

 Model eğitimini başlat
python future_ubr_ffn.py
```

Eğitim GPU/MPS üzerinde yaklaşık - dakika sürer. Model `data/ffn_model/` altına kaydedilir.

---

  Mimari (Monad-EMDE)

Script, Monad ve EMDE makalelerinde önerilen Residual Feed-Forward Network mimarisini kullanır.

 Girdi ve Çıktı
- Girdi:  Boyut (`Past UBR` [] + `Portfolio` [])
- Çıktı:  Boyut (`Future UBR` - Log Probabilities)

 Katman Yapısı ve Nedenleri

| Katman | Ne Yapıyor? | Neden Gerekli? |
|--------|-------------|----------------|
| L Normalizasyon | Girdi vektörlerini birim küreye projete eder. | EMDE sketch'leri sparse ve farklı büyüklüklerde olabilir. L norm, tüm müşterileri aynı ölçeğe getirir ve gradient patlamasını önler. |
| Projeksiyon (→) | Boyut indirgeme. | Girdi çok büyük (K). Daha küçük bir manifold'a sıkıştırarak genelleme kabiliyetini artırır. |
| Residual Bloklar (×) | `Linear→BatchNorm→LeakyReLU→Dropout + Skip` | Skip Connection sayesinde gradientler doğrudan en başa ulaşır (vanishing gradient önlenir). Derin ağlar bu yapı olmadan eğitilemez. |
| BatchNorm | Her katmanın çıktısını normalize eder. | Eğitimi hızlandırır ve regularizasyon sağlar. |
| LeakyReLU | Negatif değerlere küçük bir eğim verir. | Standart ReLU'daki "ölü nöron" (dead neuron) problemini önler. |
| LogSoftmax | Çıktıyı log-olasılık dağılımına çevirir. | KL-Divergence loss'u için gereklidir. Ayrıca sayısal kararlılık sağlar (underflow önler). |

---

  Eğitim Stratejisi

 Loss Function: KL-Divergence
Modelin çıktısı bir sınıf değil, bir dağılımdır (bir histogram gibi). İki dağılım arasındaki farkı ölçmek için en doğru metrik Kullback-Leibler (KL) Divergence'dır.

 Optimizasyon
- Optimizer: AdamW (Weight Decay ile regülarizasyon)
- Scheduler: ReduceLROnPlateau (Loss düşmezse learning rate'i azaltır)
- Patience:  epoch boyunca iyileşme olmazsa eğitim durur (Early Stopping).

---

  Çıktılar

Eğitim bittiğinde `data/ffn_model/` klasörüne şunlar kaydedilir:

| Dosya | Açıklama |
|-------|----------|
| `future_ubr_model_walk.pt` | Eğitilmiş PyTorch model ağırlıkları |
| `future_ubr_training.png` | Loss eğrisi ve Cosine Similarity histogramı |
| `future_ubr_predictions...npz` | Test seti tahminleri |

 Başarı Kriteri: Cosine Similarity
Eğitim sonrası modelin başarısı "Cosine Similarity" ile ölçülür.
- Hedef: > .
- Mevcut Başarı (Realistic Mode): ~. (Ortalama)

Bu yüksek benzerlik, modelin müşterinin gelecekte ne yapacağını (hangi alt uzayda yoğunlaşacağını) çok iyi öğrendiğini gösterir. Bu öğrenilen ağırlıklar (Backbone), Churn tahmini için transfer edilecektir.
