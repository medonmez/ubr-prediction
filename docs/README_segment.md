# 🎯 Segment Prediction Model (Fine-Tuning)

Bu doküman, müşterinin segmentini (mass, affluent, business, private) tahmin eden `segment_prediction_finetune.py` scriptinin teknik detaylarını açıklar. Bu model, **Transfer Learning** yöntemiyle, önceden eğitilmiş **Monad-EMDE Foundation Model** üzerine inşa edilir.

---

## 📋 İçindekiler

1. [Amaç](#-amaç)
2. [Hızlı Başlangıç](#-hızlı-başlangıç)
3. [Segment Sınıfları](#-segment-sınıfları)
4. [Konfigürasyon Parametreleri](#️-konfigürasyon-parametreleri)
5. [Veri Akışı](#-veri-akışı)
6. [Model Mimarisi](#-model-mimarisi)
7. [Eğitim Stratejisi](#-eğitim-stratejisi)
8. [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
9. [Görselleştirmeler](#-görselleştirmeler)
10. [Çıktılar](#-çıktılar)

---

## 🎯 Amaç

Müşterilerin davranış örüntülerinden (event'ler) ve ürün portföylerinden segmentlerini tahmin etmek. Bu, Foundation Model'in öğrendiği temsilleri kullanarak **4 sınıflı (multi-class)** bir sınıflandırma problemidir.

**Kullanım Alanları:**
- Yeni müşteri segment ataması
- Segment geçiş tahmini (mass → affluent potansiyeli)
- Pazarlama stratejisi optimizasyonu
- Kişiselleştirilmiş ürün önerileri

---

## 🚀 Hızlı Başlangıç

```bash
# Sanal ortamı aktif et
source venv/bin/activate

# Çalışma dizinine git
cd scripts

# Fine-tuning işlemini başlat
python segment_prediction_finetune.py
```

> [!IMPORTANT]
> **Ön Koşul:** Bu script çalıştırılmadan önce aşağıdaki adımlar tamamlanmış olmalıdır:
> 1. `bank_emde_session.py` → EMDE sketch'lerinin üretilmesi
> 2. `future_ubr_ffn.py` → Foundation Model'in eğitilmesi

İşlem GPU/MPS üzerinde ~5 dakikadan kısa sürer.

---

## 🏷️ Segment Sınıfları

| Segment | Açıklama | Dağılım |
|---------|----------|---------|
| **mass** | Standart bireysel müşteriler | ~65% (6,495) |
| **affluent** | Yüksek gelirli bireysel müşteriler | ~20% (1,994) |
| **business** | Kurumsal/ticari müşteriler | ~10% (989) |
| **private** | VIP/özel bankacılık müşterileri | ~5% (522) |

> [!NOTE]
> Veri seti dengesiz (imbalanced). Model, **Weighted CrossEntropyLoss** kullanarak bu dengesizliği ele alır.

---

## ⚙️ Konfigürasyon Parametreleri

### Dizin Ayarları
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `DATA_DIR` | `../data/emde` | EMDE sketch dosyalarının konumu |
| `FOUNDATION_MODEL_DIR` | `../data/ffn_model` | Pre-trained Foundation Model konumu |
| `OUTPUT_DIR` | `../data/segment_model` | Çıktı dosyalarının kaydedileceği yer |
| `WALK` | `4` | Kullanılacak Cleora walk sayısı |

### Eğitim Hiperparametreleri
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `FREEZE_BACKBONE` | `False` | Backbone ağırlıklarını dondur/aç |
| `LEARNING_RATE` | `1e-4` | Temel öğrenme hızı |
| `BATCH_SIZE` | `256` | Mini-batch boyutu |
| `EPOCHS` | `50` | Maksimum epoch sayısı |
| `PATIENCE` | `10` | Early stopping sabır değeri |
| `NUM_CLASSES` | `4` | Segment sınıf sayısı |

---

## 📊 Veri Akışı

### 1. Veri Yükleme

```
emde_session_sketches_walk4.npz
├── past_sketches      → [N, 320] - Geçmiş davranış sketch'i
├── portfolio_sketches → [N, 320] - Portföy sketch'i
└── segments           → [N] - Segment etiketleri (string)
```

### 2. SegmentDataset Sınıfı

```python
class SegmentDataset(Dataset):
    def __init__(self, past_sketches, portfolio_sketches, segment_labels):
        # Past ve Portfolio sketch'lerini birleştir
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1)
        # X boyutu: [N, 640] (320 + 320)
        
        # String etiketleri integer'a çevir
        segment_to_idx = {'mass': 0, 'affluent': 1, 'business': 2, 'private': 3}
        self.y = [segment_to_idx[s] for s in segment_labels]
```

### 3. Veri Bölme

```
+------------------+
|   Toplam Veri    | 100% (10,000 müşteri)
+--------+---------+
         |
    +----+----+----+
    |    |    |    |
  Train Val  Test
   70%  15%  15%
  7000  1500  1500
```

---

## 🧠 Model Mimarisi

### Genel Yapı

```
┌─────────────────────────────────────────────────────────────┐
│                   SegmentPredictor Model                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input [640]                                                │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐                                           │
│   │ L2 Normalize│  F.normalize(x, p=2, dim=-1)              │
│   └─────────────┘                                           │
│        │                                                     │
│        ▼                                                     │
│   ╔═════════════════════════════════════════════════════╗   │
│   ║           BACKBONE (From Foundation Model)           ║   │
│   ╠═════════════════════════════════════════════════════╣   │
│   ║  Input Projection                                    ║   │
│   ║  ├── Linear(640 → 3000)                             ║   │
│   ║  ├── BatchNorm1d(3000)                              ║   │
│   ║  └── LeakyReLU(0.01)                                ║   │
│   ║                                                      ║   │
│   ║  Residual Blocks (x3)                               ║   │
│   ║  ┌────────────────────────────────────┐             ║   │
│   ║  │ Linear(3000 → 3000)               │             ║   │
│   ║  │ BatchNorm1d(3000)                 │──┐          ║   │
│   ║  │ LeakyReLU(0.01)                   │  │ Skip     ║   │
│   ║  │ Dropout(0.1)                      │  │ Conn.    ║   │
│   ║  └────────────────────────────────────┘  │          ║   │
│   ║            +  ◄──────────────────────────┘          ║   │
│   ║                                                      ║   │
│   ║  Output Projection                                   ║   │
│   ║  └── Linear(3000 → 320)                             ║   │
│   ╚═════════════════════════════════════════════════════╝   │
│        │                                                     │
│        ▼ [320]                                               │
│   ╔═════════════════════════════════════════════════════╗   │
│   ║           CLASSIFICATION HEAD (New)                  ║   │
│   ╠═════════════════════════════════════════════════════╣   │
│   ║  ├── Linear(320 → 256)                              ║   │
│   ║  ├── BatchNorm1d(256)                               ║   │
│   ║  ├── LeakyReLU(0.01)                                ║   │
│   ║  ├── Dropout(0.3)                                   ║   │
│   ║  ├── Linear(256 → 64)                               ║   │
│   ║  ├── BatchNorm1d(64)                                ║   │
│   ║  ├── LeakyReLU(0.01)                                ║   │
│   ║  ├── Dropout(0.2)                                   ║   │
│   ║  └── Linear(64 → 4)  ← 4 sınıf                      ║   │
│   ╚═════════════════════════════════════════════════════╝   │
│        │                                                     │
│        ▼                                                     │
│   Output: Logits [4] → Softmax → Probabilities              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Churn vs Segment Model Farkı

| Özellik | Churn Model | Segment Model |
|---------|-------------|---------------|
| Çıktı boyutu | 1 (binary) | 4 (multi-class) |
| Loss function | BCEWithLogitsLoss | CrossEntropyLoss |
| Aktivasyon | Sigmoid | Softmax |
| Metrikler | ROC-AUC, Recall | Macro F1, OvR AUC |

---

## 📚 Eğitim Stratejisi

### 1. Loss Function: Weighted CrossEntropyLoss

Segment dağılımı dengesiz olduğu için, az temsil edilen sınıflara daha yüksek ağırlık verilir:

```python
def compute_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Inverse frequency weighting
    weights = total / (len(classes) * counts)
    return weights
```

**Hesaplanan Ağırlıklar (örnek):**
| Segment | Count | Weight |
|---------|-------|--------|
| mass | 6,495 | ~0.38 |
| affluent | 1,994 | ~1.25 |
| business | 989 | ~2.53 |
| private | 522 | ~4.79 |

### 2. Optimizer: AdamW with Differential Learning Rates

```python
# Backbone: Düşük LR (mevcut bilgiyi koru)
backbone_lr = 1e-4

# Classifier: Yüksek LR (sıfırdan öğren)
classifier_lr = 1e-3  # (10x backbone)
```

### 3. Learning Rate Scheduler & Early Stopping

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# 10 epoch boyunca val_loss iyileşmezse early stopping
```

---

## 📈 Değerlendirme Metrikleri

### Mevcut Sonuçlar

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Accuracy** | 98.27% | Genel doğruluk |
| **F1 Macro** | 95.64% | Tüm sınıflar eşit ağırlıklı F1 |
| **F1 Weighted** | 98.28% | Sınıf dağılımına göre ağırlıklı F1 |
| **ROC-AUC (OvR)** | 99.54% | One-vs-Rest multi-class AUC |
| **Precision (Macro)** | 95.31% | Ortalama precision |
| **Recall (Macro)** | 96.03% | Ortalama recall |

### Metrik Açıklamaları

| Metrik | Ne Ölçer? | Ne Zaman Önemli? |
|--------|-----------|------------------|
| **F1 Macro** | Tüm sınıfların eşit ağırlıklı performansı | Az temsil edilen sınıflar önemliyse |
| **F1 Weighted** | Sınıf dağılımına göre ağırlıklı performans | Genel başarı önemliyse |
| **ROC-AUC (OvR)** | Her sınıfın ayrılabilirliği | Sınıflandırma kalitesi |

### Confusion Matrix Yorumlama

```
                     Predicted
              mass  affluent  business  private
Actual  ┌─────────────────────────────────────┐
mass    │  ████     ░         ░         ░     │
affluent│  ░        ████      ░         ░     │
business│  ░        ░         ████      ░     │
private │  ░        ░         ░         ████  │
        └─────────────────────────────────────┘
        
████ = Doğru tahmin (diagonal)
░    = Yanlış tahmin (off-diagonal)
```

---

## 📊 Görselleştirmeler

Script iki görselleştirme dosyası üretir:

### 1. `segment_prediction_results.png` (2x2 Grid)

| Konum | Grafik | Açıklama |
|-------|--------|----------|
| Sol Üst | Training & Validation Loss | Epoch bazında CrossEntropy loss |
| Sağ Üst | Validation Accuracy & F1 | Epoch bazında metrikler |
| Sol Alt | Confusion Matrix Heatmap | 4x4 sınıflandırma doğruluğu |
| Sağ Alt | Prediction Distribution | Her sınıf için olasılık dağılımı |

### 2. `segment_roc_curves.png` (2x2 Grid)

Her segment için ayrı ROC eğrisi:
- **mass** ROC-AUC
- **affluent** ROC-AUC
- **business** ROC-AUC
- **private** ROC-AUC

---

## 📁 Çıktılar

| Dosya | Konum | Açıklama |
|-------|-------|----------|
| `segment_predictor_walk4.pt` | `data/segment_model/` | Eğitilmiş model checkpoint'i |
| `segment_predictions_walk4.npz` | `data/segment_model/` | Test seti tahminleri |
| `segment_prediction_results.png` | `data/segment_model/` | Eğitim grafikleri |
| `segment_roc_curves.png` | `data/segment_model/` | ROC eğrileri |

### Model Checkpoint İçeriği

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'freeze_backbone': FREEZE_BACKBONE,
    'walk': WALK,
    'metrics': {
        'accuracy': float,
        'precision_macro': float,
        'recall_macro': float,
        'f1_macro': float,
        'f1_weighted': float,
        'auc_ovr': float
    },
    'segment_classes': ['mass', 'affluent', 'business', 'private'],
    'architecture': 'MonadEMDE_SegmentPredictor'
}, model_path)
```

### Predictions NPZ İçeriği

```python
np.savez_compressed(pred_path,
    predictions=preds,       # [N_test] - Tahmin edilen sınıflar (0-3)
    probabilities=probs,     # [N_test, 4] - Her sınıf için olasılıklar
    labels=labels,           # [N_test] - Gerçek etiketler (0-3)
    metrics=metrics,         # Dict - Test metrikleri
    segment_classes=SEGMENT_CLASSES
)
```

---

## 🔧 İnference (Tahmin Yapma)

```python
import torch
import numpy as np

# Model yükle
checkpoint = torch.load('data/segment_model/segment_predictor_walk4.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tahmin yap
with torch.no_grad():
    logits = model(input_tensor)  # [B, 4]
    probs = torch.softmax(logits, dim=-1)  # [B, 4] olasılıklar
    predictions = torch.argmax(probs, dim=-1)  # [B] sınıf indeksleri

# İndeksi segment ismine çevir
SEGMENT_CLASSES = ['mass', 'affluent', 'business', 'private']
segment_names = [SEGMENT_CLASSES[p] for p in predictions]
```

---

## 💼 Kullanım Senaryoları

### 1. Yeni Müşteri Segmentasyonu
```
Yeni müşteri davranışı → Model → Segment tahmini → CRM'e yaz
```

### 2. Segment Geçiş Potansiyeli
```python
# mass müşterisi affluent'a ne kadar yakın?
if probs[0, 1] > 0.3:  # affluent olasılığı > %30
    print("Affluent potansiyeli var!")
```

### 3. Pazarlama Kampanyası Hedefleme
```python
# En belirsiz müşterileri bul (entropy yüksek)
entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
uncertain_customers = np.argsort(entropy)[-100:]  # En belirsiz 100
```

---

## 📚 Bağımlılıklar

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
```
