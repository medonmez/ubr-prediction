# 🔴 Churn Prediction Model (Fine-Tuning)

Bu doküman, müşterinin churn (kayıp) olasılığını tahmin eden `churn_prediction_finetune.py` scriptinin teknik detaylarını açıklar. Bu model, **Transfer Learning** yöntemiyle, önceden eğitilmiş **Monad-EMDE Foundation Model** üzerine inşa edilir.

---

## 📋 İçindekiler

1. [Amaç](#-amaç)
2. [Hızlı Başlangıç](#-hızlı-başlangıç)
3. [Konfigürasyon Parametreleri](#️-konfigürasyon-parametreleri)
4. [Veri Akışı](#-veri-akışı)
5. [Model Mimarisi](#-model-mimarisi)
6. [Eğitim Stratejisi](#-eğitim-stratejisi)
7. [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
8. [Görselleştirmeler](#-görselleştirmeler)
9. [Çıktılar](#-çıktılar)
10. [Teknik Detaylar](#-teknik-detaylar)

---

## 🎯 Amaç

Sıfırdan bir model eğitmek yerine, müşterinin **gelecekteki davranışını (Future UBR)** öğrenmiş olan Foundation Model'in bilgisini kullanarak, daha az etiketli veri ile daha yüksek performanslı bir churn tahmini yapmaktır.

**Neden Transfer Learning?**
- Foundation Model, müşterilerin davranış örüntülerini zaten öğrenmiştir
- Daha az etiketli veri ile daha iyi sonuçlar elde edilir
- Eğitim süresi önemli ölçüde kısalır
- Modelin genelleme yeteneği artar

---

## 🚀 Hızlı Başlangıç

```bash
# Sanal ortamı aktif et
source venv/bin/activate

# Çalışma dizinine git
cd scripts

# Fine-tuning işlemini başlat
python churn_prediction_finetune.py
```

> [!IMPORTANT]
> **Ön Koşul:** Bu script çalıştırılmadan önce aşağıdaki adımlar tamamlanmış olmalıdır:
> 1. `bank_emde_session.py` → EMDE sketch'lerinin üretilmesi
> 2. `future_ubr_ffn.py` → Foundation Model'in eğitilmesi

İşlem GPU/MPS üzerinde ~5 dakikadan kısa sürer.

---

## ⚙️ Konfigürasyon Parametreleri

Script başında tanımlanan parametreler:

### Dizin Ayarları
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `DATA_DIR` | `../data/emde` | EMDE sketch dosyalarının konumu |
| `FOUNDATION_MODEL_DIR` | `../data/ffn_model` | Pre-trained Foundation Model konumu |
| `OUTPUT_DIR` | `../data/churn_model` | Çıktı dosyalarının kaydedileceği yer |
| `WALK` | `4` | Kullanılacak Cleora walk sayısı (DLSH ile) |

### Eğitim Hiperparametreleri
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `FREEZE_BACKBONE` | `False` | Backbone ağırlıklarını dondur/aç |
| `LEARNING_RATE` | `1e-4` | Temel öğrenme hızı |
| `BATCH_SIZE` | `256` | Mini-batch boyutu |
| `EPOCHS` | `50` | Maksimum epoch sayısı |
| `PATIENCE` | `10` | Early stopping sabır değeri |
| `LEAKY_RELU_SLOPE` | `0.01` | LeakyReLU negatif eğim |
| `SEED` | `42` | Rastgelelik tohumu (reproducibility) |

### Cihaz Seçimi
Script otomatik olarak en uygun cihazı seçer:
```python
DEVICE = "cuda" if torch.cuda.is_available() 
         else "mps" if torch.backends.mps.is_available() 
         else "cpu"
```

---

## 📊 Veri Akışı

### 1. Veri Yükleme (`load_data` fonksiyonu)

```
emde_session_sketches_walk4.npz
├── past_sketches      → [N, 320] - Geçmiş davranış sketch'i (10 subspace × 32 bin)
├── portfolio_sketches → [N, 320] - Portföy sketch'i (10 subspace × 32 bin)
└── churn_labels       → [N] - Binary churn etiketi (0/1)
```

### 2. ChurnDataset Sınıfı

```python
class ChurnDataset(Dataset):
    def __init__(self, past_sketches, portfolio_sketches, churn_labels):
        # Past ve Portfolio sketch'lerini birleştir
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1)
        # X boyutu: [N, 640] (320 + 320)
        self.y = churn_labels
```

**Girdi Vektörü Yapısı (640 boyut):**
```
[0:320]      → Past UBR Sketch (dinamik davranış)
[320:640]    → Portfolio Sketch (statik ürün sahipliği)
```

### 3. Veri Bölme

```
+------------------+
|   Toplam Veri    | 100% (N müşteri)
+--------+---------+
         |
    +----+----+----+
    |    |    |    |
  Train Val  Test
   70%  15%  15%
```

---

## 🧠 Model Mimarisi

### Genel Yapı

```
┌─────────────────────────────────────────────────────────────┐
│                    ChurnPredictor Model                      │
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
│   ║  │ Dropout(0.1)                      │  │ Connection║   │
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
│   ║  └── Linear(64 → 1)                                 ║   │
│   ╚═════════════════════════════════════════════════════╝   │
│        │                                                     │
│        ▼                                                     │
│   Output: Logits [1]                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1. Backbone (Omurga) - Foundation Model'den Transfer

Foundation Model'in (`future_ubr_model_walk4.pt`) eğitilmiş ağırlıkları yüklenir:

**Yükleme Süreci:**
```python
checkpoint = torch.load(foundation_model_path, ...)
input_dim = checkpoint['input_dim']      # 640 (320 + 320)
hidden_dim = checkpoint['hidden_dim']    # 3000 (~3000 nöron, paper spec)
num_layers = checkpoint['num_layers']    # 3
output_dim = checkpoint['output_dim']    # 320
```

**Ağırlık Aktarımı:**
- `input_proj` → Linear + BN + LeakyReLU
- `residual_blocks` → 3 adet ResidualBlock
- `output_proj` → Linear (LogSoftmax olmadan!)

> [!NOTE]
> Orijinal Foundation Model'in sonundaki `LogSoftmax` katmanı **atılır**, çünkü artık density estimation değil, binary sınıflandırma yapıyoruz.

### 2. ResidualBlock (Monad-EMDE Paper'dan)

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim=3000, dropout=0.1):
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.block(x) + x  # ← Skip connection
```

### 3. Classification Head (Yeni Eklenen)

Backbone'un çıkardığı 320 boyutlu öznitelik vektörünü alıp churn olasılığına (0-1 arası) çevirir:

| Katman | Girdi → Çıktı | Açıklama |
|--------|---------------|----------|
| Linear + BN + LeakyReLU | 320 → 256 | Feature reduction |
| Dropout(0.3) | 256 → 256 | Overfitting engelleyici |
| Linear + BN + LeakyReLU | 256 → 64 | Daha fazla sıkıştırma |
| Dropout(0.2) | 64 → 64 | Overfitting engelleyici |
| Linear | 64 → 1 | Binary çıktı (logit) |

---

## 📚 Eğitim Stratejisi

### 1. Loss Function: Weighted BCEWithLogitsLoss

Churn verisi dengesizdir (Örn: %20 Churn, %80 Retained). Standart loss fonksiyonu, çoğunluk sınıfını (Retained) tahmin etmeye odaklanır.

**Çözüm: Pozitif Sınıf Ağırlıklandırması**

```python
n_positive = churn_labels.sum()      # Churn eden müşteri sayısı
n_negative = len(labels) - n_positive # Kalan müşteri sayısı
pos_weight = n_negative / n_positive  # ~4.0 (veri setine göre değişir)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Etki:** Model, bir churn müşterisini kaçırdığında ~4 kat daha fazla ceza alır.

### 2. Optimizer: AdamW with Differential Learning Rates

```python
# Backbone: Düşük LR (mevcut bilgiyi koru)
backbone_params = [...input_proj, residual_blocks, output_proj...]
backbone_lr = 1e-4

# Classifier: Yüksek LR (sıfırdan öğren)
classifier_lr = 1e-3  # (10x backbone)

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': backbone_lr},
    {'params': model.classifier.parameters(), 'lr': classifier_lr}
])
```

**Neden Farklı Learning Rate?**
- **Backbone:** Zaten değerli bilgi içeriyor → yavaş güncelle, bilgiyi koru
- **Classifier:** Sıfırdan başlıyor → hızlı öğren, adapte ol

### 3. Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Val loss azaldığında
    factor=0.5,      # LR'yi yarıya indir
    patience=5       # 5 epoch iyileşme olmazsa
)
```

### 4. Early Stopping

```python
PATIENCE = 10  # 10 epoch boyunca val_loss iyileşmezse dur

if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = model.state_dict().copy()
    patience_counter = 0
else:
    patience_counter += 1
    
if patience_counter >= PATIENCE:
    model.load_state_dict(best_model_state)  # En iyi modele geri dön
    break
```

### 5. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Gradientlerin patlamasını önler, eğitimi stabilize eder.

### 6. Backbone Freezing (Opsiyonel)

```python
FREEZE_BACKBONE = False  # True yapılırsa:

# Backbone parametreleri dondurulur
for param in self.input_proj.parameters():
    param.requires_grad = False
for block in self.residual_blocks:
    for param in block.parameters():
        param.requires_grad = False
for param in self.output_proj.parameters():
    param.requires_grad = False
```

| Mod | Trainable Params | Kullanım Durumu |
|-----|------------------|-----------------|
| `FREEZE_BACKBONE=True` | ~850K (sadece classifier) | Az veri, hızlı eğitim |
| `FREEZE_BACKBONE=False` | ~5M+ (tüm model) | Çok veri, daha iyi performans |

---

## 📈 Değerlendirme Metrikleri

### Test Seti Metrikleri

| Metrik | Anlamı | Hedef |
|--------|--------|-------|
| **ROC-AUC** | Modelin churn ve retained sınıflarını ayırma yeteneği | > 0.85 |
| **Recall** | Gerçekten churn edenlerin kaçını yakaladık? (**En kritik**) | > 0.70 |
| **Precision** | "Churn edecek" dediklerimizin kaçı gerçekten etti? | > 0.60 |
| **F1 Score** | Precision ve Recall'un harmonik ortalaması | > 0.65 |
| **Accuracy** | Genel doğruluk oranı | > 0.80 |
| **Average Precision** | Precision-Recall eğrisi altındaki alan | > 0.70 |

### Confusion Matrix Yorumlama

```
                  Predicted
              Retained  Churned
Actual  ┌────────────┬────────────┐
Retained│     TN     │     FP     │  ← False alarm (gereksiz kampanya)
        ├────────────┼────────────┤
Churned │     FN     │     TP     │  ← Kaçırılan churn (kritik!)
        └────────────┴────────────┘
```

- **False Negative (FN):** En tehlikeli! Churn edecek müşteriyi kaçırdık
- **False Positive (FP):** Gereksiz retention kampanyası (maliyet var ama kabul edilebilir)

---

## 📊 Görselleştirmeler

Script iki kapsamlı görselleştirme dosyası üretir:

### 1. `churn_prediction_results.png` (2x2 Grid)

| Konum | Grafik | Açıklama |
|-------|--------|----------|
| Sol Üst | Training & Validation Loss | Epoch bazında BCE loss değişimi |
| Sağ Üst | Validation AUC | Epoch bazında AUC değişimi + en iyi değer |
| Sol Alt | Prediction Distribution | Churn vs Retained tahmin olasılıkları histogramı |
| Sağ Alt | ROC Curve | True Positive Rate vs False Positive Rate |

### 2. `churn_advanced_kpis.png` (2x3 Grid)

| Konum | Grafik | Açıklama |
|-------|--------|----------|
| Sol Üst | Precision-Recall Curve | İmbalanced data için önemli |
| Orta Üst | Confusion Matrix Heatmap | Sınıflandırma doğruluğu detayı |
| Sağ Üst | Calibration Plot | Tahmin olasılıklarının güvenilirliği |
| Sol Alt | Metrics vs Threshold | Farklı threshold'larda Precision/Recall/F1 |
| Orta Alt | Cumulative Gain (Lift) Chart | Modelin random'a göre üstünlüğü |
| Sağ Alt | Summary Metrics Table | Tüm KPI'ların özet tablosu |

**Lift Chart Yorumlama:**
```
Örnek: "Lift @ 20% = 2.5x" 
→ En riskli %20 müşteriyi hedeflediğimizde, 
  rastgele seçime göre 2.5 kat daha fazla churn yakalıyoruz.
```

---

## 📁 Çıktılar

| Dosya | Konum | Açıklama |
|-------|-------|----------|
| `churn_predictor_walk4.pt` | `data/churn_model/` | Eğitilmiş model checkpoint'i |
| `churn_predictions_walk4.npz` | `data/churn_model/` | Test seti tahminleri ve etiketleri |
| `churn_prediction_results.png` | `data/churn_model/` | Temel eğitim grafikleri |
| `churn_advanced_kpis.png` | `data/churn_model/` | Gelişmiş KPI grafikleri |

### Model Checkpoint İçeriği

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'freeze_backbone': FREEZE_BACKBONE,
    'walk': WALK,
    'metrics': {
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'auc': float
    },
    'architecture': 'MonadEMDE_ChurnPredictor'
}, model_path)
```

### Predictions NPZ İçeriği

```python
np.savez_compressed(pred_path, 
    predictions=preds,  # [N_test] - Churn olasılıkları (0-1)
    labels=labels,      # [N_test] - Gerçek etiketler (0/1)
    metrics=metrics     # Dict - Test metrikleri
)
```

---

## 🔧 Teknik Detaylar

### Forward Pass Akışı

```python
def forward(self, x):
    # 1. L2 Normalization (Foundation Model ile tutarlılık)
    x = F.normalize(x, p=2, dim=-1)  # [B, 640] → [B, 640]
    
    # 2. Input Projection
    h = self.input_proj(x)  # [B, 640] → [B, 3000]
    
    # 3. Residual Blocks (x3)
    for block in self.residual_blocks:
        h = block(h)  # [B, 3000] → [B, 3000] (skip connection ile)
    
    # 4. Output Projection
    features = self.output_proj(h)  # [B, 3000] → [B, 320]
    
    # 5. Classification Head
    logits = self.classifier(features)  # [B, 320] → [B, 1]
    
    return logits.squeeze(-1)  # [B]
```

### İnference (Tahmin Yapma)

```python
model.eval()
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits)  # Logits → 0-1 olasılık
    predictions = (probabilities >= 0.5).int()  # Binary sınıflandırma
```

### Eğitim Döngüsü Özeti

```
For each epoch:
    1. model.train()
    2. For each batch:
        a. Forward pass → logits
        b. Compute weighted BCE loss
        c. Backward pass
        d. Gradient clipping (max_norm=1.0)
        e. Optimizer step
    3. model.eval()
    4. Validation loop → val_loss, val_auc
    5. Learning rate scheduler step
    6. Early stopping check
```

---

## 💼 Kullanım Senaryosu

Bu modelin çıktısı (Churn Skoru), pazarlama departmanı tarafından önleyici aksiyonlar almak için kullanılabilir:

```
Churn Score > 0.7  → Yüksek Risk  → Acil arama + özel kampanya
Churn Score > 0.5  → Orta Risk   → Email kampanyası
Churn Score > 0.3  → Düşük Risk  → Genel bilgilendirme
Churn Score < 0.3  → Güvenli     → Standart hizmet
```

---

## 📚 Bağımlılıklar

```python
import torch                  # PyTorch core
import torch.nn as nn         # Neural network modülleri
import torch.nn.functional as F  # Fonksiyonel operasyonlar
import torch.optim as optim   # Optimizasyon algoritmaları
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
```
