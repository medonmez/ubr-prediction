# Bank Customer Churn & Segment Prediction Pipeline

Banka müşterilerinin **churn (kayıp) olasılığını** ve **segmentini** tahmin eden uçtan uca (end-to-end) bir makine öğrenmesi pipeline'ı.

## 🎯 Özellikler

- **Cleora Graph Embeddings**: Müşteri-ürün-event ilişkilerinden 1024 boyutlu vektörler
- **EMDE/DLSH Sketches**: Density-dependent LSH ile 320 boyutlu yoğunluk tahminleri
- **Foundation Model**: Future UBR tahmini için pre-trained Monad-EMDE FFN
- **Churn Prediction**: Transfer learning ile binary churn tahmini (~98% AUC)
- **Segment Prediction**: 4-class segment tahmini (~99% AUC)

---

## 📁 Proje Yapısı

```
churn-test/
├── README.md              
├── requirements.txt       # Python bağımlılıkları
├── scripts/               # Tüm Python scriptleri
│   ├── generate_bank_data.py         # 1. Sentetik veri üretimi
│   ├── bank_cleora.py                # 2. Cleora graph embeddings
│   ├── bank_emde_session.py          # 3. EMDE sketch generation (DLSH)
│   ├── future_ubr_ffn.py             # 4. Foundation model eğitimi
│   ├── churn_prediction_finetune.py  # 5. Churn model fine-tuning
│   └── segment_prediction_finetune.py # 6. Segment model fine-tuning
├── docs/                  # Dokümantasyon
│   ├── README_data_gen.md
│   ├── README_cleora.md
│   ├── README_emde.md
│   ├── README_foundation.md
│   ├── README_churn.md
│   └── README_segment.md
└── data/                  # Üretilen veriler ve modeller
    ├── bank_customers.csv
    ├── bank_products.csv
    ├── bank_events.csv
    ├── cleora_hyperedges.txt
    ├── embeddings/        # Cleora vektörleri + t-SNE görselleri
    ├── emde/              # EMDE sketches + visualizations
    ├── ffn_model/         # Foundation model
    ├── churn_model/       # Churn prediction model
    └── segment_model/     # Segment prediction model
```

---

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pipeline'ı Çalıştır
Scriptleri sırayla çalıştırın:

```bash
cd scripts

# Adım 1: Sentetik veri üret (~1 dk)
python generate_bank_data.py

# Adım 2: Graph embeddings oluştur (~2 dk)
python bank_cleora.py

# Adım 3: EMDE sketches oluştur (~5 dk)
python bank_emde_session.py

# Adım 4: Foundation model eğit (~2 dk)
python future_ubr_ffn.py

# Adım 5: Churn model fine-tune (~1 dk)
python churn_prediction_finetune.py

# Adım 6: Segment model fine-tune (~1 dk)
python segment_prediction_finetune.py
```

---

## 📊 Pipeline Diyagramı

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA GENERATION                                  │
│  generate_bank_data.py                                                   │
│  ├── bank_customers.csv (10K müşteri, segment, churn_label)             │
│  ├── bank_products.csv (ürün sahiplikleri)                              │
│  ├── bank_events.csv (30 günlük event'ler)                              │
│  └── cleora_hyperedges.txt                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLEORA EMBEDDINGS                                │
│  bank_cleora.py                                                          │
│  └── 1024-dim entity vectors (customers, products, events)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EMDE/DLSH SKETCHES                               │
│  bank_emde_session.py                                                    │
│  ├── Past UBR Sketch (320-dim, day 0-25, time-decay)                    │
│  ├── Future UBR Sketch (320-dim, day 25-30, target)                     │
│  └── Portfolio Sketch (320-dim, static products)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FOUNDATION MODEL                                 │
│  future_ubr_ffn.py                                                       │
│  └── Input: Past+Portfolio (640) → Hidden: 3000 → Output: Future (320)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       CHURN PREDICTION          │   │      SEGMENT PREDICTION         │
│  churn_prediction_finetune.py   │   │  segment_prediction_finetune.py │
│  ├── Binary classification      │   │  ├── 4-class classification     │
│  ├── Weighted BCE Loss          │   │  ├── Weighted CrossEntropy      │
│  └── ROC-AUC: ~98%              │   │  └── ROC-AUC: ~99%              │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 📈 Pipeline Özeti

| Adım | Script | Girdi | Çıktı | Süre |
|------|--------|-------|-------|------|
| 1 | `generate_bank_data.py` | - | CSV dosyaları, hyperedges | ~1 dk |
| 2 | `bank_cleora.py` | hyperedges | 1024-dim entity embeddings | ~2 dk |
| 3 | `bank_emde_session.py` | embeddings + events | 320-dim customer sketches | ~5 dk |
| 4 | `future_ubr_ffn.py` | Past/Future UBR | Pre-trained foundation model | ~2 dk |
| 5 | `churn_prediction_finetune.py` | Sketches + Churn Labels | Churn predictor | ~1 dk |
| 6 | `segment_prediction_finetune.py` | Sketches + Segments | Segment predictor | ~1 dk |

---

## 📊 Sonuçlar

### Churn Prediction (`data/churn_model/`)
| Metrik | Değer |
|--------|-------|
| ROC-AUC | ~98% |
| Recall | ~95% |
| F1 Score | ~90% |

### Segment Prediction (`data/segment_model/`)
| Metrik | Değer |
|--------|-------|
| Accuracy | 98.27% |
| F1 Macro | 95.64% |
| ROC-AUC (OvR) | 99.54% |

**Segment Sınıfları:** mass, affluent, business, private

---

## 📚 Dokümantasyon

Her script için detaylı açıklamalar `docs/` klasöründe:

| Doküman | Açıklama |
|---------|----------|
| [Veri Üretimi](docs/README_data_gen.md) | Sentetik banka verisi üretimi |
| [Cleora Embeddings](docs/README_cleora.md) | Graph embedding yöntemi |
| [EMDE Sketches](docs/README_emde.md) | DLSH-based density estimation |
| [Foundation Model](docs/README_foundation.md) | Monad-EMDE FFN mimarisi |
| [Churn Prediction](docs/README_churn.md) | Binary churn tahmini |
| [Segment Prediction](docs/README_segment.md) | 4-class segment tahmini |

---

## 🔧 Teknik Detaylar

### EMDE Sketch Boyutları
- **Subspaces:** 10
- **Bins per subspace:** 32
- **Sketch dimension:** 10 × 32 = 320

### Model Mimarisi
- **Input:** 640 (Past UBR + Portfolio)
- **Hidden:** 3000 neurons × 3 residual blocks
- **Output:** 320 (Future UBR) veya sınıflandırma

### Kullanılan Teknolojiler
- **Cleora:** Graph embedding (pycleora)
- **DLSH:** Density-dependent Locality Sensitive Hashing
- **PyTorch:** Neural network training
- **scikit-learn:** Evaluation metrics
