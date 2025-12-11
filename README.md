# Bank Customer Churn Prediction Pipeline

Banka müşterilerinin churn (kayıp) olasılığını tahmin eden uçtan uca (end-to-end) bir makine öğrenmesi pipeline'ı.

## Proje Yapısı

```
churn-test/
├── README.md              
├── requirements.txt       # Python bağımlılıkları
├── scripts/               # Tüm Python scriptleri
│   ├── generate_bank_data.py      # 1. Sentetik veri üretimi
│   ├── bank_cleora.py             # 2. Cleora graph embeddings
│   ├── bank_emde_session.py       # 3. EMDE sketch generation
│   ├── future_ubr_ffn.py          # 4. Foundation model eğitimi
│   └── churn_prediction_finetune.py # 5. Churn model fine-tuning
├── docs/                  # Dokümantasyon
│   ├── README_data_gen.md
│   ├── README_cleora.md
│   ├── README_emde.md
│   ├── README_foundation.md
│   └── README_churn.md
└── data/                  # Üretilen veriler ve modeller
    ├── bank_customers.csv
    ├── bank_products.csv
    ├── bank_events.csv
    ├── cleora_hyperedges.txt
    ├── embeddings/        # Cleora vektörleri
    ├── emde/              # EMDE sketches
    ├── ffn_model/         # Foundation model
    └── churn_model/       # Churn prediction model
```

---

## Hızlı Başlangıç

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
```

---

## Pipeline Özeti

| Adım | Script | Girdi | Çıktı |
|------|--------|-------|-------|
| 1 | `generate_bank_data.py` | - | CSV dosyaları, hyperedges |
| 2 | `bank_cleora.py` | hyperedges | 1024-dim entity embeddings |
| 3 | `bank_emde_session.py` | embeddings + events | 4096-dim customer sketches |
| 4 | `future_ubr_ffn.py` | Past/Future UBR | Pre-trained foundation model |
| 5 | `churn_prediction_finetune.py` | Sketches + Churn Labels | Final churn predictor |

---

## Sonuçlar

Pipeline tamamlandığında `data/churn_model/` içinde:
- `churn_prediction_results.png`: Temel metrikler
- `churn_advanced_kpis.png`: KPI'lar (Lift, PR Curve, Calibration)
- `churn_predictor_walk4.pt`: Eğitilmiş model


---

## Dokümantasyon

Her script için detaylı açıklamalar `docs/` klasöründe:
- [Veri Üretimi](docs/README_data_gen.md)
- [Cleora Embeddings](docs/README_cleora.md)
- [EMDE Sketches](docs/README_emde.md)
- [Foundation Model](docs/README_foundation.md)
- [Churn Prediction](docs/README_churn.md)
