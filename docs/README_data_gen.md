# Bank Churn Sentetik Veri Üreticisi

Bu doküman, banka müşteri davranışlarını ve churn (müşteri kaybı) sinyallerini simüle eden `generate_bank_data.py` scriptini detaylı olarak açıklar.

---

## Amaç

Gerçek bir banka verisindeki **"gürültülü"** ve **"istatistiksel"** churn sinyallerini simüle etmektir. Model sadece davranışsal kalıpları (behavioral drift) öğrenmeli, "kopya çekmemelidir".

> [!IMPORTANT]
> **Sessiz Churn Simülasyonu**: Churn eden müşteri "Ben gidiyorum" demez, sadece sessizleşir ve belki bir kez şikayet eder. Modelin başarısı bu sessizliği duymasına bağlıdır.

---

## Hızlı Başlangıç

```bash
source venv/bin/activate
python generate_bank_data.py
```

---

## Veri Üretim Pipeline'ı

```
┌─────────────────────────────────────────────────────────────┐
│  1. generate_customers()    → 10,000 müşteri                │
│     - Segment ataması (mass, affluent, private, business)   │
│     - Churn durumu ÖNCEden belirlenir                       │
├─────────────────────────────────────────────────────────────┤
│  2. assign_products()       → ~43,000 ürün ataması          │
│     - Segmente uygun ürünler                                │
│     - Ağırlıklı random seçim                                │
├─────────────────────────────────────────────────────────────┤
│  3. generate_events()       → ~750,000 event                │
│     - Churn sinyalleri enjekte edilir                       │
│     - Gürültü eklenir (deterministic DEĞİL)                 │
├─────────────────────────────────────────────────────────────┤
│  4. generate_cleora_hyperedges() → 10,000 hyperedge         │
│     - Graph embedding formatı                               │
│     - customer + products + events                          │
├─────────────────────────────────────────────────────────────┤
│  5. save_data()             → CSV + TXT dosyaları           │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Müşteri Üretimi (`generate_customers`)

### Segment Dağılımı

| Segment | Dağılım | Aktivite | Ürün Sayısı | Churn Base |
|---------|---------|----------|-------------|------------|
| **Mass** | 65% | 0.8x | 2-5 | 12% |
| **Affluent** | 20% | 1.2x | 4-8 | 8% |
| **Private** | 5% | 1.5x | 6-12 | 5% |
| **Business** | 10% | 1.3x | 3-7 | 10% |

### Churn Pre-determination

```python
# Müşteri yaratılırken churn ÖNCEden belirlenir
is_churning = random.random() < seg_config["churn_base"]

customer = {
    "customer_id": f"C{i:06d}",
    "segment": segment,
    "age": random.randint(*seg_config["age_range"]),
    "risk_score": round(random.uniform(0, 100), 2),
    "tenure_months": random.randint(1, 240),
    "is_churning": is_churning,  # Bu bilgi event üretimini etkiler
}
```

> [!NOTE]
> Bu yaklaşım sentetik veride kasıtlı bir "veri sızıntısı" oluşturur. Gerçek veride böyle değildir - gerçek churn tarihsel veriden belirlenir.

---

## 2. Ürün Ataması (`assign_products`)

### Ürün Kategorileri

```python
PRODUCTS = {
    # Kredi Kartları
    "credit_card_basic": {"segment": ["mass"], "weight": 0.4},
    "credit_card_gold": {"segment": ["mass", "affluent"], "weight": 0.25},
    "credit_card_platinum": {"segment": ["affluent", "private"], "weight": 0.15},
    
    # Hesaplar
    "checking_account": {"segment": ["all"], "weight": 0.95},
    "savings_account": {"segment": ["mass", "affluent", "private"], "weight": 0.6},
    "investment_account": {"segment": ["affluent", "private"], "weight": 0.3},
    
    # Krediler
    "mortgage": {"segment": ["mass", "affluent", "private"], "weight": 0.15},
    "personal_loan": {"segment": ["mass", "affluent"], "weight": 0.2},
    
    # Yatırım Ürünleri
    "mutual_fund": {"segment": ["affluent", "private"], "weight": 0.25},
    "stocks": {"segment": ["affluent", "private"], "weight": 0.15},
    "bonds": {"segment": ["private"], "weight": 0.1},
}
```

### Atama Mantığı

```python
# Segmente uygun ürünler filtrelenir
eligible_products = [p for p in PRODUCTS if segment in PRODUCTS[p]["segment"]]

# Segment'e göre ürün sayısı belirlenir
num_products = random.randint(*seg_config["product_count"])

# Ağırlıklı random seçim
selected = random.choices(eligible_products, weights=[...])
```

---

## 3. Event Üretimi (`generate_events`)

### Event Kategorileri ve Churn Multiplier

| Kategori | Event | Base Freq | Churn Mult | Açıklama |
|----------|-------|-----------|------------|----------|
| **Dijital** | login_mobile | 15/ay | 0.1x | ↓ Çok düşer |
| | login_web | 8/ay | 0.1x | ↓ Çok düşer |
| **Transfer** | transfer_eft | 4/ay | 0.15x | ↓ Düşer |
| | transfer_fast | 6/ay | 0.15x | ↓ Düşer |
| **Ödeme** | bill_payment | 5/ay | 0.3x | ↓ Düşer |
| | loan_payment | 1/ay | 0.5x | ↓ Az düşer |
| **Kart** | card_transaction_pos | 20/ay | 0.1x | ↓ Çok düşer |
| **Müşteri Hizm.** | customer_service_call | 0.5/ay | **8.0x** | ↑ Çok artar |
| | complaint | 0.1/ay | **15.0x** | ↑ En çok artar |
| **Olumlu** | loan_application | 0.1/ay | 0.2x | ↓ Düşer |
| | investment_buy | 1/ay | 0.2x | ↓ Düşer |

### Gürültü Mekanizması

```python
if is_churning:
    churn_mult = event_config.get("churn_multiplier", 1.0)
    churn_noise = random.uniform(0.5, 1.5)  # Gürültü faktörü
    
    # Şikayet artışı probabilistik (deterministik DEĞİL)
    if "complaint" in event_type:
        if random.random() < 0.3:  # 30% sessiz kalır
            churn_mult = 1.0
    
    base_freq *= (churn_mult * churn_noise)
else:
    # Retained müşteriler de bazen şikayet eder
    if "complaint" in event_type and random.random() < 0.05:
        base_freq *= 5.0  # Rastgele öfkeli müşteri
```

### Poisson Dağılımı

```python
expected_count = base_freq * activity_factor * (days / 30)
event_count = np.random.poisson(max(expected_count, 0))
```

### Amount Üretimi (Log-Uniform)

```python
if event_config["amount_range"]:
    min_amt, max_amt = event_config["amount_range"]
    # Log-uniform: Küçük tutarlar daha sık
    amount = round(np.exp(np.random.uniform(
        np.log(min_amt), np.log(max_amt)
    )), 2)
```

---

## 4. Cleora Hyperedge Üretimi

### Format

```
customer_id product1 product2 ... event1 event2 ...
```

### Örnek

```text
C000001 checking_account credit_card_gold login_mobile transfer_eft card_transaction_pos
C000002 checking_account savings_account mortgage bill_payment atm_withdrawal
```

### Kod

```python
def generate_cleora_hyperedges(customers_df, products_df, events_df, top_n_events=12):
    for customer in customers_df.iterrows():
        edge_elements = [cust_id]
        
        # Ürünler
        products = customer_products.get(cust_id, [])
        edge_elements.extend(products)
        
        # Son N unique event tipi
        recent_events = cust_events["event_type"].drop_duplicates().head(top_n_events)
        edge_elements.extend(recent_events)
        
        hyperedge = " ".join(edge_elements)
```

> [!IMPORTANT]
> Hyperedge'lerde **explicit churn token yok**. Model, davranış paterninden churn'ü öğrenmeli.

---

## 5. Çıktı Dosyaları

| Dosya | Boyut | İçerik |
|-------|-------|--------|
| `bank_customers.csv` | ~500 KB | customer_id, segment, age, risk_score, tenure, churn_label |
| `bank_products.csv` | ~1.5 MB | customer_id, product_id, acquisition_date |
| `bank_events.csv` | ~50 MB | event_id, customer_id, event_type, timestamp, amount, channel |
| `cleora_hyperedges.txt` | ~1 MB | Cleora graph formatı |

---

## Konfigürasyon

```python
# Değiştirilebilir Sabitler
NUM_CUSTOMERS = 10_000
SEED = 42

# Segment Karakteristikleri
SEGMENTS = {
    "mass": {"activity_multiplier": 0.8, "product_count": (2, 5), "churn_base": 0.12},
    "affluent": {"activity_multiplier": 1.2, "product_count": (4, 8), "churn_base": 0.08},
    "private": {"activity_multiplier": 1.5, "product_count": (6, 12), "churn_base": 0.05},
    "business": {"activity_multiplier": 1.3, "product_count": (3, 7), "churn_base": 0.10},
}
```

---

## Örnek İstatistikler

```
Customers: 10,000
  - mass: 6,500 (65%)
  - affluent: 2,000 (20%)
  - private: 500 (5%)
  - business: 1,000 (10%)

Churn Distribution:
  - Churned: ~1,100 (11%)
  - Retained: ~8,900 (89%)

Event Statistics:
  - Churned: avg 45 events/customer
  - Retained: avg 80 events/customer
```

---

## Churn Sinyalleri Özeti

| Sinyal | Churner Davranışı | Retained Davranışı |
|--------|-------------------|-------------------|
| **Login Frekansı** | 0.1x (↓ %90 düşüş) | 1.0x (normal) |
| **Şikayet** | 15x (↑ %70'i şikayet eder) | 1x (%5'i rastgele şikayet) |
| **Yatırım** | 0.2x (↓ %80 düşüş) | 1.0x (normal) |
| **Müşteri Hizmetleri** | 8x (↑ sık arama) | 1.0x (normal) |

---

## Pipeline Sonraki Adımlar

```bash
python generate_bank_data.py       # 1. Veri üret
python bank_cleora.py              # 2. Cleora embedding
python bank_emde_session.py        # 3. EMDE sketch
python future_ubr_ffn.py           # 4. Foundation model
python churn_prediction_finetune.py # 5. Churn model
```
