 Churn Prediction Model (Fine-Tuning)

Bu doküman, müşterinin churn (kayıp) olasılığını tahmin eden `churn_prediction_finetune.py` scriptinin teknik detaylarını açıklar. Bu model, "Transfer Learning" yöntemiyle, önceden eğitilmiş Foundation Model üzerine inşa edilir.

  Amaç
Sıfırdan bir model eğitmek yerine, müşterinin gelecekteki davranışını (Future UBR) öğrenmiş olan Foundation Model'in bilgisini kullanarak, daha az etiketli veri ile daha yüksek performanslı bir churn tahmini yapmaktır.

---

  Hızlı Başlangıç

```bash
 Sanal ortamı aktif et
source venv/bin/activate

 Fine-tuning işlemini başlat
python churn_prediction_finetune.py
```

İşlem GPU/MPS üzerinde  dakikadan kısa sürer. Öncesinde `future_ubr_ffn.py` çalıştırılmış olmalıdır.

---

  Model Mimarisi

Model iki ana parçadan oluşur:

 . Backbone (Omurga) - Foundation Model'den Transfer
Foundation Model'in (`future_ubr_model`) eğitilmiş ağırlıkları yüklenir:
- Yapı:  Katmanlı Residual Network ( nöron).
- Bilgi: Müşterinin geçmiş davranışlarından gelecekteki yoğunluk haritasını çıkarmayı öğrenmiştir.
- Modifikasyon: Orijinal modelin sonundaki `LogSoftmax` katmanı atılır, çünkü artık density estimation değil, sınıflandırma yapıyoruz.

 . Classification Head (Sınıflandırıcı) - Yeni Eklenen
Backbone'un çıkardığı  boyutlu öznitelik vektörünü alıp churn olasılığına (- arası) çevirir:
- `Linear( -> )` + `Bn` + `LeakyReLU` + `Dropout`
- `Linear( -> )` + `Bn` + `LeakyReLU` + `Dropout`
- `Linear( -> )` (Binary Output)

---

  Eğitim Stratejisi

 Loss Function: Weighted BCEWithLogitsLoss
Churn verisi dengesizdir (Örn: % Churn, % Retained). Standart loss fonksiyonu, çoğunluk sınıfını (Retained) tahmin etmeye odaklanır.
Bunu çözmek için Pozitif Sınıf Ağırlıklandırması (Positive Class Weighting) kullanılır:
- `pos_weight = Retained Sayısı / Churn Sayısı` (Yaklaşık .)
- Bu sayede model, bir churn müşterisini kaçırdığında  kat daha fazla ceza alır.

 Optimizer & Learning Rate
- Backbone: Düşük hızda öğrenir (`e-`). Mevcut bilgisini koruması gerekir.
- Classifier: Yüksek hızda öğrenir (`e-`). Sıfırdan öğrendiği için hızlı adapte olmalıdır.

---

  Değerlendirme Metrikleri

Modelin başarısı şu metriklerle ölçülür:

| Metrik | Anlamı | Hedef |
|--------|--------|-------|
| ROC-AUC | Modelin churn ve retained sınıflarını ayırma yeteneği. | > . (Mevcut: ~.) |
| Recall | Gerçekten churn edenlerin kaçını yakaladık? (En kritik metrik). | > . |
| Precision | "Churn edecek" dediklerimizin kaçı gerçekten etti? | > . |

 Görselleştirmeler
Eğitim sonrası `data/churn_model/churn_prediction_results.png` dosyasında:
- ROC Eğrisi: Eğri sol üste ne kadar yakınsa o kadar iyi.
- Tahmin Dağılımı: Churn (Kırmızı) ve Retained (Yeşil) histogramları ne kadar ayrık?

---

  Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `churn_predictor_walk.pt` | Eğitilmiş son model. |
| `churn_predictions_walk.npz` | Test seti üzerindeki olasılıklar ve gerçek etiketler. |

Bu modelin çıktısı (Churn Skoru), pazarlama departmanı tarafından önleyici aksiyonlar (kampanya, arama vb.) almak için kullanılabilir.
