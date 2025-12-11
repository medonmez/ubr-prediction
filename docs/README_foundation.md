# Monad-EMDE Foundation Model

Bu doküman, müşterinin gelecekteki davranış yoğunluğunu (Future UBR) tahmin eden Foundation Model'in (`future_ubr_ffn.py`) teknik detaylarını ve mimari kararlarını açıklar.

---

## Amaç

Modelin amacı, bir müşterinin **geçmiş davranışlarını** (Past UBR) ve **sahip olduğu ürünleri** (Portfolio) alarak, **gelecekteki davranış dağılımını** (Future UBR) tahmin etmektir.

> [!IMPORTANT]
> Bu bir **Self-Supervised Learning** görevidir. Etiketler verinin kendisinden gelir - modele "gelecek" olarak verdiğimiz şey aslında zamansal olarak ayrılmış gerçek veridir.

---

## Hızlı Başlangıç

```bash
source venv/bin/activate
python future_ubr_ffn.py
```

Eğitim MPS/GPU üzerinde ~1-2 dakika sürer. Çıktılar `data/ffn_model/` altına kaydedilir.

---

## Veri Akışı

```
┌─────────────────────────────────────────────────────────────────┐
│                         GİRDİ                                   │
├─────────────────────────────────────────────────────────────────┤
│  Past UBR (320-dim)  +  Portfolio (320-dim)  =  640-dim         │
│  [0-25 gün event]       [sahip olunan ürünler]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MonadEMDE FFN                                │
│  L2 Normalize → Input Proj → 3× Residual Block → Output Proj   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ÇIKTI                                   │
├─────────────────────────────────────────────────────────────────┤
│  Future UBR (320-dim) - Log Probabilities                       │
│  [25-30 gün event tahmin]                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mimari: MonadEMDEPredictor

### Genel Yapı

```python
class MonadEMDEPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=3000, num_layers=3):
        # 1. Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # 640 → 3000
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # 2. Residual Blocks (3 adet)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # 3. Output Projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)  # 3000 → 320
        
        # 4. LogSoftmax
        self.log_softmax = nn.LogSoftmax(dim=-1)
```

### Residual Block Detayı

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        self.block = nn.Sequential(
            nn.Linear(dim, dim),          # 3000 → 3000
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.block(x) + x  # Skip Connection
```

### Forward Pass

```python
def forward(self, x):
    # 1. L2 Normalize input
    x = F.normalize(x, p=2, dim=-1)
    
    # 2. Input projection (640 → 3000)
    h = self.input_proj(x)
    
    # 3. Residual blocks (3× skip connections)
    for block in self.residual_blocks:
        h = block(h)
    
    # 4. Output projection (3000 → 320)
    out = self.output_proj(h)
    
    # 5. LogSoftmax (KL-Divergence için)
    return self.log_softmax(out)
```

---

## Katman Açıklamaları

| Katman | Boyut | Neden Gerekli? |
|--------|-------|----------------|
| **L2 Normalize** | - | EMDE sketch'leri farklı büyüklüklerde olabilir. L2 norm tüm müşterileri birim küreye projekte eder. |
| **Input Projection** | 640 → 3000 | Düşük boyutlu girdiyi yüksek kapasiteli hidden space'e genişletir. |
| **Residual Blocks** | 3000 → 3000 | Skip connection ile gradient akışı korunur, vanishing gradient önlenir. |
| **BatchNorm** | - | Internal covariate shift'i azaltır, eğitimi stabilize eder. |
| **LeakyReLU** | slope=0.01 | Standart ReLU'daki "dead neuron" problemini önler. |
| **Output Projection** | 3000 → 320 | Hidden features'ı çıktı boyutuna sıkıştırır. |
| **LogSoftmax** | - | KL-Divergence loss için log-probability çıktısı. |

---

## Konfigürasyon

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `HIDDEN_DIM` | 3000 | Hidden layer boyutu (Monad paper önerisi) |
| `NUM_HIDDEN_LAYERS` | 3 | Residual block sayısı |
| `DROPOUT` | 0.1 | Regularizasyon için |
| `LEAKY_RELU_SLOPE` | 0.01 | Negatif bölge eğimi |
| `LEARNING_RATE` | 1e-3 | Başlangıç öğrenme oranı |
| `BATCH_SIZE` | 256 | Mini-batch boyutu |
| `PATIENCE` | 10 | Early stopping patience |

---

## Eğitim Stratejisi

### Loss Function: KL-Divergence

Model bir dağılım tahmin ettiği için (histogram benzeri), iki dağılım arasındaki farkı ölçmek için **Kullback-Leibler Divergence** kullanılır:

```python
criterion = nn.KLDivLoss(reduction='batchmean')

# Target'ı probability distribution'a çevir
y_prob = normalize_to_distribution(y_batch)  # Sum = 1

# Model output zaten log-probability
pred_logprob = model(X_batch)

# KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
loss = criterion(pred_logprob, y_prob)
```

### Target Normalization

```python
def normalize_to_distribution(x, eps=1e-8):
    x = torch.clamp(x, min=0)  # Negatif değerleri sıfırla
    x = x + eps                 # Zero division önle
    x = x / x.sum(dim=-1, keepdim=True)  # Sum = 1
    return x
```

### Optimizer ve Scheduler

```python
# AdamW: Weight decay ile L2 regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Validation loss düşmezse LR'ı yarıya indir
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Dataset Yapısı

```python
class UBRDataset(Dataset):
    def __init__(self, past_sketches, portfolio_sketches, future_sketches):
        # Girdi: Past + Portfolio concatenation
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1)
        # (10000, 320) + (10000, 320) = (10000, 640)
        
        # Hedef: Future UBR
        self.y = future_sketches  # (10000, 320)
```

### Train/Val/Test Split

```python
n_train = int(0.7 * n_samples)   # 7,000
n_val = int(0.15 * n_samples)    # 1,500
n_test = n_samples - n_train - n_val  # 1,500
```

---

## Çıktılar

`data/ffn_model/` klasörüne kaydedilen dosyalar:

| Dosya | İçerik |
|-------|--------|
| `future_ubr_model_walk4.pt` | Model ağırlıkları + checkpoint |
| `future_ubr_training.png` | Loss curve + similarity histogram |
| `future_ubr_predictions_walk4.npz` | Test seti predictions |

### Checkpoint İçeriği

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,        # 640
    'output_dim': output_dim,      # 320
    'hidden_dim': HIDDEN_DIM,      # 3000
    'num_layers': NUM_HIDDEN_LAYERS,  # 3
    'dropout': DROPOUT,
    'walk': WALK,
    'mean_similarity': similarities.mean(),
    'architecture': 'MonadEMDE_ResidualFFN'
}, model_path)
```

---

## Değerlendirme: Cosine Similarity

Eğitim sonrası başarı **Cosine Similarity** ile ölçülür:

```python
from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(preds)):
    sim = cosine_similarity(preds[i:i+1], targets[i:i+1])[0, 0]
    similarities.append(sim)
```

### Sonuçlar

| Metrik | Değer |
|--------|-------|
| Mean Cosine Similarity | **0.72** |
| Median | **0.80** |
| Max | 0.97 |

> Bu, modelin müşterinin gelecekte hangi davranış bölgesinde yoğunlaşacağını iyi tahmin ettiğini gösterir.

---

## Transfer Learning

Eğitilen **backbone** (input_proj + residual_blocks + output_proj), churn prediction için transfer edilir:

```python
# churn_prediction_finetune.py
class ChurnPredictor(nn.Module):
    def __init__(self, foundation_model_path, freeze_backbone=False):
        # Pre-trained ağırlıkları yükle
        checkpoint = torch.load(foundation_model_path)
        
        # Backbone'u yeniden oluştur
        self.input_proj = ...
        self.residual_blocks = ...
        self.output_proj = ...
        
        # Ağırlıkları yükle
        self.input_proj.load_state_dict(...)
        
        # Classification head ekle
        self.classifier = nn.Sequential(
            nn.Linear(320, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )
```

Bu sayede model, "davranış → gelecek davranış" ilişkisini öğrenir ve bu bilgi churn prediction'a transfer edilir.
