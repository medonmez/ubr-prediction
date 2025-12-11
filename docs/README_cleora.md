# Bank Cleora Embedding Generator

Bu doküman, müşterileri, ürünleri ve davranışları (event) ortak bir vektör uzayına gömmek için kullanılan `bank_cleora.py` scriptini açıklar.

---

## Amaç

Hyperedge verisinden **Müşteri-Ürün-Event** ilişkilerini koruyan yüksek boyutlu (1024-dim) vektörler üretmektir. Bu vektörler EMDE aşamasında girdi olarak kullanılır.

> [!IMPORTANT]
> Cleora, tüm entity'leri (customer, product, event) **aynı vektör uzayına** yerleştirir. Bu sayede "login_mobile" ile "C000001" arasındaki ilişki vektör benzerliği olarak ölçülebilir.

---

## Hızlı Başlangıç

```bash
source venv/bin/activate
python bank_cleora.py
```

---

## Algoritma: Cleora

Cleora, yinelemeli **Markov Zinciri yayılımı** kullanır:

### Adımlar

```python
def create_cleora_embeddings(hyperedges, embed_dim, num_walks):
    # 1. Sparse matris oluştur
    mat = SparseMatrix.from_iterator(iter(hyperedges), columns='complex::reflexive::entity')
    
    # 2. Deterministik başlatma (reproducible)
    embeddings = mat.initialize_deterministically(embed_dim)  # 1024-dim
    
    # 3. Markov yayılımı (num_walks kez)
    for i in range(num_walks):
        # Her entity'nin vektörü = komşularının ortalaması
        embeddings = mat.left_markov_propagate(embeddings)
        
        # L2 normalizasyon
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    
    return entity_ids, embeddings
```

### Walk Sayısının Etkisi

| Walk | Öğrenilen İlişki | Örnek |
|------|------------------|-------|
| 1 | Direkt komşular | `C000001 → login_mobile` |
| 2 | 2. derece | `C000001 → login_mobile → (diğer login yapanlar)` |
| 3 | 3. derece | Daha geniş ağ yapısı |
| 4 | 4. derece | Global yapı (over-smoothing riski) |

> [!WARNING]
> Walk sayısı arttıkça **over-smoothing** riski artar. Tüm vektörler birbirine yaklaşır ve ayırt edicilik kaybolur. Bu projede walk=1-2 en iyi sonucu verdi.

---

## Konfigürasyon

```python
# bank_cleora.py
EMBED_DIM = 1024           # Vektör boyutu
WALK_OPTIONS = [1, 2, 3, 4]  # Üretilecek walk'lar
INPUT_FILE = "../data/cleora_hyperedges.txt"
OUTPUT_DIR = "../data/embeddings"
```

---

## Girdi Formatı

Script `data/cleora_hyperedges.txt` dosyasını okur:

```text
C000001 checking_account credit_card_gold login_mobile transfer_eft card_transaction_pos
C000002 checking_account savings_account mortgage bill_payment atm_withdrawal complaint
```

Her satır bir **hyperedge**:
- İlk eleman: Müşteri ID
- Sonra: Ürünler
- Sonra: Son N event tipi

---

## Çıktılar

### Embedding Dosyaları

`data/embeddings/` klasörüne kaydedilir:

| Dosya | İçerik |
|-------|--------|
| `cleora_embeddings_walk1.npz` | Walk=1 embedding'leri |
| `cleora_embeddings_walk2.npz` | Walk=2 embedding'leri |
| `cleora_embeddings_walk3.npz` | Walk=3 embedding'leri |
| `cleora_embeddings_walk4.npz` | Walk=4 embedding'leri |

### NPZ İçeriği

```python
data = np.load("cleora_embeddings_walk1.npz")
data["entity_ids"]   # ["C000001", "C000002", ..., "login_mobile", "credit_card_gold"]
data["embeddings"]   # (10039, 1024) - tüm entity'lerin vektörleri
data["num_walks"]    # 1
data["embed_dim"]    # 1024
```

---

## t-SNE Görselleştirmeleri

Her walk için 3 görselleştirme üretilir:

### 1. All Entities (`tsne_all_entities_walkX.png`)

```python
# Müşteriler (mavi) + Ürünler/Eventler (kırmızı)
# Ürün ve event etiketleri gösterilir
```

**Ne Aramalı**: Benzer ürünler yakın mı? (örn: tüm credit_card'lar bir arada)

### 2. Customers by Segment (`tsne_customers_segment_walkX.png`)

```python
segment_colors = {
    'mass': '#3498db',      # Mavi
    'affluent': '#2ecc71',  # Yeşil
    'private': '#9b59b6',   # Mor
    'business': '#e67e22'   # Turuncu
}
```

**Ne Aramalı**: Segmentler ayrışıyor mu?

### 3. Customers by Churn (`tsne_customers_churn_walkX.png`)

```python
churn_colors = {
    0: '#2ecc71',  # Yeşil (Retained)
    1: '#e74c3c'   # Kırmızı (Churned)
}
```

**Ne Aramalı**: Churn edenler belirli bölgelerde mi yoğunlaşmış?

---

## Entity Sayıları

```
Matrix: 10,039 entities
  - Customers: 10,000
  - Products & Events: 39
```

### Entity Türleri

| Tür | Örnek | Sayı |
|-----|-------|------|
| Customer | C000001, C000002, ... | 10,000 |
| Product | checking_account, credit_card_gold | ~17 |
| Event | login_mobile, complaint | ~22 |

---

## EMDE ile Entegrasyon

EMDE sadece **Product ve Event** embedding'lerini kullanır:

```python
# bank_emde_session.py
def load_cleora_embeddings(walk: int):
    data = np.load(f"cleora_embeddings_walk{walk}.npz")
    entity_ids = data["entity_ids"]
    embeddings = data["embeddings"]
    
    # Sadece product/event'leri filtrele (C ile başlamayanlar)
    mask = ~np.array([str(e).startswith("C") for e in entity_ids])
    return entity_ids[mask], embeddings[mask]
```

---

## İdeal Sonuçlar

### İyi Cleora Çıktısı:

1. **Segment Ayrımı**: Private ve Mass müşterileri farklı bölgelerde
2. **Churn Ayrımı**: Churner'lar belirli kümede yoğunlaşmış
3. **Ürün Gruplaması**: Benzer ürünler (tüm credit_card'lar) birbirine yakın

### Kötü Cleora Çıktısı:

- Tüm noktalar rastgele dağılmış (sinyal yok)
- Over-smoothing: Herkes birbirine çok yakın (walk çok yüksek)

---

## Sonraki Adım

```bash
python bank_emde_session.py  # EMDE sketch oluştur
```
