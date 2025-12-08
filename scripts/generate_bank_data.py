#!/usr/bin/env python3

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from faker import Faker

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
NUM_CUSTOMERS = 10_000
OUTPUT_DIR = "../data"

fake = Faker('tr_TR')
Faker.seed(SEED)

# =============================================================================
# BANK PRODUCT DEFINITIONS
# =============================================================================

PRODUCTS = {
    # Credit Cards
    "credit_card_basic": {"segment": ["mass"], "monthly_fee": 0, "weight": 0.4},
    "credit_card_gold": {"segment": ["mass", "affluent"], "monthly_fee": 50, "weight": 0.25},
    "credit_card_platinum": {"segment": ["affluent", "private"], "monthly_fee": 150, "weight": 0.15},
    "credit_card_business": {"segment": ["business"], "monthly_fee": 100, "weight": 0.2},
    
    # Accounts
    "checking_account": {"segment": ["mass", "affluent", "private", "business"], "monthly_fee": 0, "weight": 0.95},
    "savings_account": {"segment": ["mass", "affluent", "private"], "monthly_fee": 0, "weight": 0.6},
    "investment_account": {"segment": ["affluent", "private"], "monthly_fee": 0, "weight": 0.3},
    "business_account": {"segment": ["business"], "monthly_fee": 25, "weight": 0.9},
    
    # Loans
    "mortgage": {"segment": ["mass", "affluent", "private"], "monthly_fee": 0, "weight": 0.15},
    "personal_loan": {"segment": ["mass", "affluent"], "monthly_fee": 0, "weight": 0.2},
    "car_loan": {"segment": ["mass", "affluent"], "monthly_fee": 0, "weight": 0.1},
    "business_loan": {"segment": ["business"], "monthly_fee": 0, "weight": 0.3},
    
    # Insurance
    "life_insurance": {"segment": ["affluent", "private"], "monthly_fee": 100, "weight": 0.15},
    "health_insurance": {"segment": ["affluent", "private", "business"], "monthly_fee": 200, "weight": 0.1},
    
    # Investment Products
    "mutual_fund": {"segment": ["affluent", "private"], "monthly_fee": 0, "weight": 0.25},
    "stocks": {"segment": ["affluent", "private"], "monthly_fee": 0, "weight": 0.15},
    "bonds": {"segment": ["private"], "monthly_fee": 0, "weight": 0.1},
}

# =============================================================================
# EVENT DEFINITIONS (with churn-specific events)
# =============================================================================

EVENTS = {
    # Digital Banking
    "login_mobile": {"base_freq": 15, "amount_range": None, "channels": ["mobile"], "churn_multiplier": 0.1},
    "login_web": {"base_freq": 8, "amount_range": None, "channels": ["web"], "churn_multiplier": 0.1},
    
    # Transfers
    "transfer_eft": {"base_freq": 4, "amount_range": (100, 50000), "channels": ["mobile", "web"], "churn_multiplier": 0.15},
    "transfer_fast": {"base_freq": 6, "amount_range": (50, 10000), "channels": ["mobile", "web"], "churn_multiplier": 0.15},
    "transfer_internal": {"base_freq": 3, "amount_range": (100, 100000), "channels": ["mobile", "web", "branch"], "churn_multiplier": 0.2},
    
    # Payments
    "bill_payment": {"base_freq": 5, "amount_range": (50, 2000), "channels": ["mobile", "web", "atm"], "churn_multiplier": 0.3},
    "credit_card_payment": {"base_freq": 2, "amount_range": (500, 20000), "channels": ["mobile", "web"], "churn_multiplier": 0.2},
    "loan_payment": {"base_freq": 1, "amount_range": (1000, 30000), "channels": ["mobile", "web", "branch"], "churn_multiplier": 0.5},
    
    # ATM Operations
    "atm_withdrawal": {"base_freq": 4, "amount_range": (100, 5000), "channels": ["atm"], "churn_multiplier": 0.2},
    "atm_deposit": {"base_freq": 1, "amount_range": (500, 20000), "channels": ["atm"], "churn_multiplier": 0.1},
    
    # Card Transactions
    "card_transaction_pos": {"base_freq": 20, "amount_range": (10, 5000), "channels": ["pos"], "churn_multiplier": 0.1},
    "card_transaction_online": {"base_freq": 10, "amount_range": (20, 3000), "channels": ["online"], "churn_multiplier": 0.1},
    "card_contactless": {"base_freq": 15, "amount_range": (5, 500), "channels": ["pos"], "churn_multiplier": 0.1},
    
    # Customer Service - INCREASED for churners
    "customer_service_call": {"base_freq": 0.5, "amount_range": None, "channels": ["call_center"], "churn_multiplier": 8.0},
    "branch_visit": {"base_freq": 0.3, "amount_range": None, "channels": ["branch"], "churn_multiplier": 5.0},
    "complaint": {"base_freq": 0.1, "amount_range": None, "channels": ["call_center", "branch", "web"], "churn_multiplier": 15.0},
    
    # CHURN-SPECIFIC EVENTS (DISABLED for realism)
    # "account_close_inquiry": {"base_freq": 0, "amount_range": None, "channels": ["call_center", "branch"], "churn_only": True, "churn_freq": 2.0},
    # "product_cancellation": {"base_freq": 0, "amount_range": None, "channels": ["call_center", "branch", "web"], "churn_only": True, "churn_freq": 1.5},
    # "dissatisfaction_feedback": {"base_freq": 0, "amount_range": None, "channels": ["web", "call_center"], "churn_only": True, "churn_freq": 1.0},
    # "competitor_inquiry": {"base_freq": 0, "amount_range": None, "channels": ["web"], "churn_only": True, "churn_freq": 0.8},
    
    # Positive engagement (REDUCED for churners but with NOISE)
    "loan_inquiry": {"base_freq": 0.2, "amount_range": None, "channels": ["mobile", "web", "branch"], "churn_multiplier": 0.3},
    "loan_application": {"base_freq": 0.1, "amount_range": None, "channels": ["mobile", "web", "branch"], "churn_multiplier": 0.2},
    "card_application": {"base_freq": 0.1, "amount_range": None, "channels": ["mobile", "web", "branch"], "churn_multiplier": 0.2},
    "investment_buy": {"base_freq": 1, "amount_range": (1000, 100000), "channels": ["mobile", "web"], "churn_multiplier": 0.2},
    "investment_sell": {"base_freq": 0.5, "amount_range": (1000, 100000), "channels": ["mobile", "web"], "churn_multiplier": 1.5},
    "investment_inquiry": {"base_freq": 2, "amount_range": None, "channels": ["mobile", "web"], "churn_multiplier": 0.5},
}

# Segment characteristics
SEGMENTS = {
    "mass": {"activity_multiplier": 0.8, "product_count": (2, 5), "age_range": (18, 65), "churn_base": 0.12},
    "affluent": {"activity_multiplier": 1.2, "product_count": (4, 8), "age_range": (28, 70), "churn_base": 0.08},
    "private": {"activity_multiplier": 1.5, "product_count": (6, 12), "age_range": (35, 75), "churn_base": 0.05},
    "business": {"activity_multiplier": 1.3, "product_count": (3, 7), "age_range": (25, 65), "churn_base": 0.10},
}


def generate_customers(num_customers: int) -> pd.DataFrame:
    """Generate customer demographic data with segments."""
    customers = []
    
    segment_weights = {"mass": 0.65, "affluent": 0.20, "private": 0.05, "business": 0.10}
    
    for i in range(num_customers):
        segment = random.choices(
            list(segment_weights.keys()),
            weights=list(segment_weights.values())
        )[0]
        
        seg_config = SEGMENTS[segment]
        
        # PRE-DETERMINE churn status based on segment
        is_churning = random.random() < seg_config["churn_base"]
        
        customer = {
            "customer_id": f"C{i:06d}",
            "segment": segment,
            "age": random.randint(*seg_config["age_range"]),
            "risk_score": round(random.uniform(0, 100), 2),
            "tenure_months": random.randint(1, 240),
            "registration_date": fake.date_between(start_date="-20y", end_date="-1m"),
            "is_churning": is_churning,
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)


def assign_products(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Assign products to customers based on their segment."""
    product_assignments = []
    
    for _, customer in customers_df.iterrows():
        segment = customer["segment"]
        seg_config = SEGMENTS[segment]
        
        eligible_products = [
            prod for prod, config in PRODUCTS.items()
            if segment in config["segment"]
        ]
        
        num_products = random.randint(*seg_config["product_count"])
        num_products = min(num_products, len(eligible_products))
        
        weights = [PRODUCTS[p]["weight"] for p in eligible_products]
        selected_products = set()
        
        while len(selected_products) < num_products:
            product = random.choices(eligible_products, weights=weights)[0]
            selected_products.add(product)
        
        for product in selected_products:
            product_assignments.append({
                "customer_id": customer["customer_id"],
                "product_id": product,
                "acquisition_date": fake.date_between(
                    start_date=customer["registration_date"],
                    end_date="today"
                )
            })
    
    return pd.DataFrame(product_assignments)

# ... (assign_products remains same) ...

def generate_events(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    days: int = 30
) -> pd.DataFrame:
    """Generate events with REALISTIC churn behavioral signals (Statistical, not deterministic)."""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_events = []
    customer_products = products_df.groupby("customer_id")["product_id"].apply(list).to_dict()
    
    for _, customer in customers_df.iterrows():
        cust_id = customer["customer_id"]
        segment = customer["segment"]
        seg_config = SEGMENTS[segment]
        is_churning = customer["is_churning"]
        
        # Base activity with NOISE
        # Some retained customers are low activity, some churners are high activity
        base_noise = random.uniform(0.5, 1.5)
        activity_factor = seg_config["activity_multiplier"] * base_noise
        
        cust_products = customer_products.get(cust_id, [])
        
        for event_type, event_config in EVENTS.items():
            # Filtering logic (investments, loans, cards) same as before...
            if event_type.startswith("investment_") and not any(p in ["investment_account", "mutual_fund", "stocks", "bonds"] for p in cust_products): continue
            if event_type == "loan_payment" and not any("loan" in p or p == "mortgage" for p in cust_products): continue
            if "card" in event_type and event_type not in ["card_application"] and not any("credit_card" in p for p in cust_products): continue
            
            # Skip disabled events
            if event_type.startswith("#"): continue
            
            base_freq = event_config["base_freq"]
            
            if is_churning:
                # Apply multipliers but add NOISE to them
                # e.g. instead of always 0.1x, it's 0.1x * random(0.5, 2.0)
                churn_mult = event_config.get("churn_multiplier", 1.0)
                churn_noise = random.uniform(0.5, 1.5) 
                
                # Make complaint increase probabilistic, not certain
                if "complaint" in event_type or "service" in event_type:
                     # 30% of churners don't complain more than usual
                     if random.random() < 0.3:
                         churn_mult = 1.0
                
                base_freq *= (churn_mult * churn_noise)
            else:
                 # Some retained customers also complain or have reduced activity
                 if "complaint" in event_type and random.random() < 0.05:
                     base_freq *= 5.0 # Random angry retained customer
            
            expected_count = base_freq * activity_factor * (days / 30)
            event_count = np.random.poisson(max(expected_count, 0))
            
            for _ in range(event_count):
                random_seconds = random.randint(0, days * 24 * 3600)
                event_time = start_date + timedelta(seconds=random_seconds)
                
                amount = None
                if event_config["amount_range"]:
                    min_amt, max_amt = event_config["amount_range"]
                    amount = round(np.exp(np.random.uniform(np.log(min_amt), np.log(max_amt))), 2)
                
                channel = random.choice(event_config["channels"])
                
                event = {
                    "event_id": f"E{len(all_events):08d}",
                    "customer_id": cust_id,
                    "event_type": event_type,
                    "timestamp": event_time,
                    "amount": amount,
                    "channel": channel,
                }
                all_events.append(event)
    
    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    
    return events_df

# ... (activity level calculators deleted/unused in hyperedges) ...

def generate_cleora_hyperedges(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    events_df: pd.DataFrame,
    top_n_events: int = 12
) -> List[str]:
    """
    Generate Cleora hyperedge format WITHOUT explicit spoiler tokens.
    
    Format: customer_id product1 ... event1 ...
    """
    
    hyperedges = []
    
    # Product lookup
    customer_products = products_df.groupby("customer_id")["product_id"].apply(list).to_dict()
    
    # Recent events per customer
    customer_events = events_df.sort_values("timestamp", ascending=False).groupby("customer_id")
    
    for _, customer in customers_df.iterrows():
        cust_id = customer["customer_id"]
        
        # Start with customer ID
        edge_elements = [cust_id]
        
        # NO explicit activity/complaint tokens added here
        
        # Add products
        products = customer_products.get(cust_id, [])
        edge_elements.extend(products)
        
        # Add recent event types
        try:
            cust_events = customer_events.get_group(cust_id)
            recent_event_types = cust_events["event_type"].drop_duplicates().head(top_n_events).tolist()
            edge_elements.extend(recent_event_types)
        except KeyError:
            pass
        
        hyperedge = " ".join(edge_elements)
        hyperedges.append(hyperedge)
    
    return hyperedges


def save_data(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    events_df: pd.DataFrame,
    hyperedges: List[str],
    output_dir: str
):
    """Save all generated data to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Rename is_churning to churn_label
    customers_df = customers_df.copy()
    customers_df["churn_label"] = customers_df["is_churning"].astype(int)
    
    customer_cols = ["customer_id", "segment", "age", "risk_score", "tenure_months", "churn_label"]
    customers_df[customer_cols].to_csv(
        os.path.join(output_dir, "bank_customers.csv"),
        index=False
    )
    print(f"✓ Saved {len(customers_df)} customers to bank_customers.csv")
    
    products_df.to_csv(
        os.path.join(output_dir, "bank_products.csv"),
        index=False
    )
    print(f"✓ Saved {len(products_df)} product assignments to bank_products.csv")
    
    events_df.to_csv(
        os.path.join(output_dir, "bank_events.csv"),
        index=False
    )
    print(f"✓ Saved {len(events_df)} events to bank_events.csv")
    
    hyperedge_path = os.path.join(output_dir, "cleora_hyperedges.txt")
    with open(hyperedge_path, "w") as f:
        for edge in hyperedges:
            f.write(edge + "\n")
    print(f"✓ Saved {len(hyperedges)} hyperedges to cleora_hyperedges.txt")
    
    # Statistics
    print("\n" + "="*60)
    print("DATA GENERATION STATISTICS (with Churn Signals)")
    print("="*60)
    
    print(f"\nCustomers: {len(customers_df):,}")
    print(f"  - Segment Distribution:")
    for seg, count in customers_df["segment"].value_counts().items():
        print(f"    {seg}: {count:,} ({count/len(customers_df)*100:.1f}%)")
    
    print(f"\nChurn Distribution:")
    churn_counts = customers_df["churn_label"].value_counts()
    print(f"  - Churned (1): {churn_counts.get(1, 0):,} ({churn_counts.get(1, 0)/len(customers_df)*100:.1f}%)")
    print(f"  - Retained (0): {churn_counts.get(0, 0):,} ({churn_counts.get(0, 0)/len(customers_df)*100:.1f}%)")
    
    # Event stats by churn
    churned_ids = set(customers_df[customers_df["churn_label"] == 1]["customer_id"])
    churn_events = events_df[events_df["customer_id"].isin(churned_ids)]
    retained_events = events_df[~events_df["customer_id"].isin(churned_ids)]
    
    print(f"\nEvent Statistics by Churn Status:")
    print(f"  - Churned customers: avg {len(churn_events)/max(len(churned_ids),1):.1f} events/customer")
    print(f"  - Retained customers: avg {len(retained_events)/max(len(customers_df)-len(churned_ids),1):.1f} events/customer")
    
    # Churn-only events
    churn_only_events = ["account_close_inquiry", "product_cancellation", "dissatisfaction_feedback", "competitor_inquiry"]
    churn_signal_counts = events_df[events_df["event_type"].isin(churn_only_events)].groupby("event_type").size()
    print(f"\nChurn-Signal Event Counts:")
    for event_type in churn_only_events:
        count = churn_signal_counts.get(event_type, 0)
        print(f"  - {event_type}: {count:,}")


def main():
    print("="*60)
    print("BANK CHURN SYNTHETIC DATA GENERATOR v2")
    print("With Enhanced Churn Signals for Cleora + EMDE")
    print("="*60)
    print()
    
    print(f"Generating data for {NUM_CUSTOMERS:,} customers...")
    print()
    
    print("[1/5] Generating customers (with pre-determined churn)...")
    customers_df = generate_customers(NUM_CUSTOMERS)
    churn_rate = customers_df["is_churning"].mean() * 100
    print(f"      Created {len(customers_df):,} customers (churn rate: {churn_rate:.1f}%)")
    
    print("[2/5] Assigning products...")
    products_df = assign_products(customers_df)
    print(f"      Assigned {len(products_df):,} products")
    
    print("[3/5] Generating events (with churn-specific patterns)...")
    events_df = generate_events(customers_df, products_df, days=30)
    print(f"      Generated {len(events_df):,} events")
    
    print("[4/5] Generating Cleora hyperedges (with activity indicators)...")
    hyperedges = generate_cleora_hyperedges(customers_df, products_df, events_df)
    print(f"      Created {len(hyperedges):,} hyperedges")
    
    print("\n[5/5] Saving data...")
    save_data(customers_df, products_df, events_df, hyperedges, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. python bank_cleora.py        # Generate Cleora embeddings")
    print("  2. python bank_emde_session.py  # Generate EMDE sketches")
    print("  3. python future_ubr_ffn.py     # Train foundation model")
    print("  4. python churn_prediction_finetune.py  # Train churn model")


if __name__ == "__main__":
    main()
