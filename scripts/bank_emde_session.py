#!/usr/bin/env python3
"""
Bank EMDE Session Sketch Generator
===================================
Creates time-decay weighted EMDE sketches from Cleora event/product embeddings.
Splits data into Past UBR (input) and Future UBR (target) for foundation model training.

Key Design Decisions:
1. Uses Event & Product vectors from Cleora (NOT customer vectors)
2. Applies exponential time decay (recent events weighted more)
3. Splits 30-day window: Past (0-25 days) → Future (25-30 days)

"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configuration
WALK_OPTIONS = [1, 4]
EMBED_DIR = "../data/embeddings"
OUTPUT_DIR = "../data/emde"

# EMDE Configuration (Based on EMDE Literature)
# ================================================
# N (Depth/Ensemble): Number of independent random subspaces
# K (Width): Determines bucket count per subspace = 2^K
# Final sketch dimension: N × 2^K
#
# Recommended for Churn Prediction:
#   N = 32 (higher = more stability, less noise)
#   K = 7  (2^7 = 128 buckets - balanced discriminative power)
#   Result: 32 × 128 = 4,096 dimensional sketch
#
# Note: Higher K (e.g., 10+) can break Cleora's metric prior
#       (similar products should remain close in embedding space)
#
NUM_SUBSPACES = 32      # N: Number of random subspaces (depth/ensemble)
SUBSPACE_DIM = 32       # Dimension of each random projection (for LSH)
NUM_BINS = 128          # 2^K where K=7 (width parameter)
# Final sketch_dim = NUM_SUBSPACES × NUM_BINS = 32 × 128 = 4,096

# Time Configuration
PAST_DAYS = 25          # Days for Past UBR (input to model)
FUTURE_DAYS = 5         # Days for Future UBR (target)
TOTAL_DAYS = 30

# Time Decay Configuration
# -------------------------
# Using exponential decay: weight = exp(-lambda * days_ago)
# Equivalent to alpha-based: weight = alpha^(days_ago), where alpha = exp(-lambda)
#
# Recommended for Churn (30-day window):
#   alpha = 0.98 → lambda ≈ 0.0202
#   - 10 days ago: 81.7% weight
#   - 25 days ago: 60.3% weight
#   - Preserves historical signals while prioritizing recent events
#
# Other options:
#   alpha = 0.99 → lambda ≈ 0.0101 (very gentle, almost no decay)
#   alpha = 0.95 → lambda ≈ 0.0513 (moderate decay)
#   alpha = 0.90 → lambda ≈ 0.1054 (aggressive decay - old default)
#
TIME_DECAY_LAMBDA = 0.02  # Corresponds to alpha ≈ 0.98 (RECOMMENDED)

# Random seed
SEED = 42
np.random.seed(SEED)


def load_cleora_embeddings(walk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load Cleora embeddings from NPZ file."""
    filepath = os.path.join(EMBED_DIR, f"cleora_embeddings_walk{walk}.npz")
    data = np.load(filepath, allow_pickle=True)
    
    entity_ids = data["entity_ids"]
    embeddings = data["embeddings"]
    
    # Separate into customers and products/events
    customer_mask = np.array([str(e).startswith("C") for e in entity_ids])
    
    # We only need product/event embeddings (NOT customer embeddings)
    product_event_ids = entity_ids[~customer_mask]
    product_event_embeddings = embeddings[~customer_mask]
    
    print(f"✓ Loaded walk{walk}: {len(product_event_ids)} product/event embeddings")
    return product_event_ids, product_event_embeddings


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load customer, product, and event data."""
    
    customers_df = pd.read_csv("../data/bank_customers.csv")
    products_df = pd.read_csv("../data/bank_products.csv")
    events_df = pd.read_csv("../data/bank_events.csv")
    
    # Parse timestamps
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    
    print(f"✓ Loaded {len(customers_df):,} customers")
    print(f"✓ Loaded {len(products_df):,} products")
    print(f"✓ Loaded {len(events_df):,} events")
    
    return customers_df, products_df, events_df


def compute_time_decay_weight(days_ago: float, lambda_decay: float = 0.02) -> float:
    """
    Compute exponential time decay weight.
    
    Formula: weight = exp(-lambda * days_ago)
    Equivalent to: weight = alpha^(days_ago), where alpha = exp(-lambda)
    
    With lambda=0.02 (alpha≈0.98):
    - days_ago = 0  (today)    → weight = 1.000 (100%)
    - days_ago = 5  (5d ago)   → weight = 0.905 (90.5%)
    - days_ago = 10 (10d ago)  → weight = 0.819 (81.9%)
    - days_ago = 15 (15d ago)  → weight = 0.741 (74.1%)
    - days_ago = 20 (20d ago)  → weight = 0.670 (67.0%)
    - days_ago = 25 (25d ago)  → weight = 0.607 (60.7%)
    
    Args:
        days_ago: Number of days since the event
        lambda_decay: Decay rate parameter (default: 0.02 for alpha≈0.98)
        
    Returns:
        Weight factor between 0 and 1
    """
    return np.exp(-lambda_decay * days_ago)


class EMDESketchGenerator:
    """
    EMDE (Efficient Manifold Density Estimator) with Time Decay.
    
    Creates weighted sparse sketches:
    1. Projects embeddings to multiple subspaces
    2. Applies time-decay weights to each embedding
    3. Creates weighted histogram per subspace
    4. Concatenates into final sketch
    """
    
    def __init__(
        self, 
        embed_dim: int,
        num_subspaces: int = 32,
        subspace_dim: int = 32,
        num_bins: int = 16,
        seed: int = 42
    ):
        self.embed_dim = embed_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.num_bins = num_bins
        self.seed = seed
        self.sketch_dim = num_subspaces * num_bins
        
        # Initialize random projections
        np.random.seed(seed)
        self.projections = []
        for i in range(num_subspaces):
            proj = SparseRandomProjection(
                n_components=subspace_dim,
                random_state=seed + i
            )
            proj.fit(np.zeros((1, embed_dim)))
            self.projections.append(proj)
        
        self.bin_edges = None
        
    def fit(self, embeddings: np.ndarray):
        """Fit bin edges based on embedding distribution."""
        print(f"  Fitting EMDE on {len(embeddings):,} embeddings...")
        
        self.bin_edges = []
        for i, proj in enumerate(self.projections):
            projected = proj.transform(embeddings)
            subspace_edges = []
            for d in range(self.subspace_dim):
                percentiles = np.linspace(0, 100, self.num_bins + 1)
                edges = np.percentile(projected[:, d], percentiles)
                subspace_edges.append(edges)
            self.bin_edges.append(subspace_edges)
        
        print(f"  ✓ Fitted {self.num_subspaces} subspaces")
        
    def create_weighted_sketch(
        self, 
        embeddings: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Create time-decay weighted EMDE sketch.
        
        Args:
            embeddings: (N, embed_dim) array of embeddings
            weights: (N,) array of time-decay weights
            
        Returns:
            (sketch_dim,) weighted sketch vector
        """
        if len(embeddings) == 0:
            return np.zeros(self.sketch_dim)
        
        sketch = np.zeros(self.sketch_dim)
        
        for i, proj in enumerate(self.projections):
            projected = proj.transform(embeddings)
            
            for d in range(min(self.subspace_dim, projected.shape[1])):
                edges = self.bin_edges[i][d]
                bin_indices = np.digitize(projected[:, d], edges[1:-1])
                bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
                
                # Weighted count
                for j, b in enumerate(bin_indices):
                    sketch_idx = i * self.num_bins + b
                    sketch[sketch_idx] += weights[j]
        
        # Normalize
        total_weight = weights.sum()
        if total_weight > 0:
            sketch = sketch / total_weight
        
        return sketch


def create_customer_session_sketches(
    emde: EMDESketchGenerator,
    entity_to_embedding: Dict[str, np.ndarray],
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    events_df: pd.DataFrame,
    reference_date: datetime,
    time_decay_lambda: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create Past and Future UBR sketches for each customer.
    
    Past UBR (0-25 days): Input to foundation model
    Future UBR (25-30 days): Target for foundation model
    
    Returns:
        past_sketches: (N, sketch_dim) Past UBR
        future_sketches: (N, sketch_dim) Future UBR  
        portfolio_sketches: (N, sketch_dim) Static product portfolio
        customer_ids: List of customer IDs
    """
    
    print(f"\n  Creating session sketches with time decay...")
    print(f"  Reference date: {reference_date.date()}")
    print(f"  Past window: 0-{PAST_DAYS} days, Future window: {PAST_DAYS}-{TOTAL_DAYS} days")
    print(f"  Time decay lambda: {time_decay_lambda}")
    
    # Create product lookup
    customer_products = products_df.groupby("customer_id")["product_id"].apply(list).to_dict()
    
    # Cutoff date for past/future split
    cutoff_date = reference_date - timedelta(days=FUTURE_DAYS)
    start_date = reference_date - timedelta(days=TOTAL_DAYS)
    
    customer_ids = customers_df["customer_id"].tolist()
    past_sketches = []
    future_sketches = []
    portfolio_sketches = []
    
    for cust_id in customer_ids:
        # Get customer's events
        cust_events = events_df[events_df["customer_id"] == cust_id].copy()
        
        # =================================================================
        # PAST UBR: Events from day 0 to day 25 (with time decay)
        # =================================================================
        past_events = cust_events[
            (cust_events["timestamp"] >= start_date) & 
            (cust_events["timestamp"] < cutoff_date)
        ]
        
        if len(past_events) > 0:
            past_embeddings = []
            past_weights = []
            
            for _, event in past_events.iterrows():
                event_type = event["event_type"]
                if event_type in entity_to_embedding:
                    past_embeddings.append(entity_to_embedding[event_type])
                    
                    # Calculate days ago from cutoff (not reference)
                    days_ago = (cutoff_date - event["timestamp"]).total_seconds() / 86400
                    weight = compute_time_decay_weight(days_ago, time_decay_lambda)
                    past_weights.append(weight)
            
            if past_embeddings:
                past_emb = np.vstack(past_embeddings)
                past_w = np.array(past_weights)
                past_sketch = emde.create_weighted_sketch(past_emb, past_w)
            else:
                past_sketch = np.zeros(emde.sketch_dim)
        else:
            past_sketch = np.zeros(emde.sketch_dim)
        
        past_sketches.append(past_sketch)
        
        # =================================================================
        # FUTURE UBR: Events from day 25 to day 30 (uniform weight)
        # =================================================================
        future_events = cust_events[
            (cust_events["timestamp"] >= cutoff_date) & 
            (cust_events["timestamp"] <= reference_date)
        ]
        
        if len(future_events) > 0:
            future_embeddings = []
            
            for _, event in future_events.iterrows():
                event_type = event["event_type"]
                if event_type in entity_to_embedding:
                    future_embeddings.append(entity_to_embedding[event_type])
            
            if future_embeddings:
                future_emb = np.vstack(future_embeddings)
                # Uniform weights for future (no decay needed for target)
                future_w = np.ones(len(future_embeddings))
                future_sketch = emde.create_weighted_sketch(future_emb, future_w)
            else:
                future_sketch = np.zeros(emde.sketch_dim)
        else:
            future_sketch = np.zeros(emde.sketch_dim)
        
        future_sketches.append(future_sketch)
        
        # =================================================================
        # PORTFOLIO SKETCH: Static product representation
        # =================================================================
        products = customer_products.get(cust_id, [])
        if products:
            prod_embeddings = []
            for prod_id in products:
                if prod_id in entity_to_embedding:
                    prod_embeddings.append(entity_to_embedding[prod_id])
            
            if prod_embeddings:
                prod_emb = np.vstack(prod_embeddings)
                prod_w = np.ones(len(prod_embeddings))
                portfolio_sketch = emde.create_weighted_sketch(prod_emb, prod_w)
            else:
                portfolio_sketch = np.zeros(emde.sketch_dim)
        else:
            portfolio_sketch = np.zeros(emde.sketch_dim)
        
        portfolio_sketches.append(portfolio_sketch)
    
    past_sketches = np.vstack(past_sketches)
    future_sketches = np.vstack(future_sketches)
    portfolio_sketches = np.vstack(portfolio_sketches)
    
    print(f"  ✓ Created {len(customer_ids):,} customer sketches")
    print(f"    Past UBR shape: {past_sketches.shape}, sparsity: {(past_sketches == 0).mean()*100:.1f}%")
    print(f"    Future UBR shape: {future_sketches.shape}, sparsity: {(future_sketches == 0).mean()*100:.1f}%")
    print(f"    Portfolio shape: {portfolio_sketches.shape}, sparsity: {(portfolio_sketches == 0).mean()*100:.1f}%")
    
    return past_sketches, future_sketches, portfolio_sketches, customer_ids


def visualize_session_sketches(
    past_sketches: np.ndarray,
    future_sketches: np.ndarray,
    customers_df: pd.DataFrame,
    walk: int,
    output_dir: str
):
    """Create t-SNE visualization of Past and Future UBR sketches."""
    
    print(f"\n  Creating t-SNE visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample for visualization
    sample_size = min(2000, len(past_sketches))
    sample_indices = np.random.choice(len(past_sketches), sample_size, replace=False)
    
    sampled_past = past_sketches[sample_indices]
    sampled_future = future_sketches[sample_indices]
    sampled_customers = customers_df.iloc[sample_indices]
    
    # t-SNE on Past UBR
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    
    # Past UBR visualization
    past_tsne = tsne.fit_transform(sampled_past)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Past by Segment
    ax = axes[0, 0]
    segment_colors = {'mass': '#3498db', 'affluent': '#2ecc71', 'private': '#9b59b6', 'business': '#e67e22'}
    for segment in segment_colors:
        mask = sampled_customers['segment'] == segment
        if mask.sum() > 0:
            ax.scatter(past_tsne[mask, 0], past_tsne[mask, 1], c=segment_colors[segment], 
                      s=30, alpha=0.6, label=f'{segment} ({mask.sum()})')
    ax.set_title(f'Past UBR by Segment (Walk={walk})', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Past by Churn
    ax = axes[0, 1]
    churn_colors = {0: '#2ecc71', 1: '#e74c3c'}
    for churn in [0, 1]:
        mask = sampled_customers['churn_label'] == churn
        label = 'Retained' if churn == 0 else 'Churned'
        if mask.sum() > 0:
            ax.scatter(past_tsne[mask, 0], past_tsne[mask, 1], c=churn_colors[churn],
                      s=30, alpha=0.6, label=f'{label} ({mask.sum()})')
    ax.set_title(f'Past UBR by Churn (Walk={walk})', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Future UBR visualization
    future_tsne = tsne.fit_transform(sampled_future)
    
    # Future by Segment
    ax = axes[1, 0]
    for segment in segment_colors:
        mask = sampled_customers['segment'] == segment
        if mask.sum() > 0:
            ax.scatter(future_tsne[mask, 0], future_tsne[mask, 1], c=segment_colors[segment],
                      s=30, alpha=0.6, label=f'{segment} ({mask.sum()})')
    ax.set_title(f'Future UBR by Segment (Walk={walk})', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Future by Churn
    ax = axes[1, 1]
    for churn in [0, 1]:
        mask = sampled_customers['churn_label'] == churn
        label = 'Retained' if churn == 0 else 'Churned'
        if mask.sum() > 0:
            ax.scatter(future_tsne[mask, 0], future_tsne[mask, 1], c=churn_colors[churn],
                      s=30, alpha=0.6, label=f'{label} ({mask.sum()})')
    ax.set_title(f'Future UBR by Churn (Walk={walk})', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'EMDE Session Sketches - Walk {walk}\nPast UBR (Day 0-25) vs Future UBR (Day 25-30)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'emde_session_tsne_walk{walk}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: emde_session_tsne_walk{walk}.png")


def main():
    """Main execution function."""
    
    print("="*60)
    print("BANK EMDE SESSION SKETCH GENERATOR")
    print("With Time Decay + Past/Future UBR Split")
    print("="*60)
    print()
    
    # Load data
    customers_df, products_df, events_df = load_data()
    
    # Get reference date (latest event timestamp)
    reference_date = events_df["timestamp"].max()
    print(f"\nReference date: {reference_date}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each walk
    for walk in WALK_OPTIONS:
        print(f"\n{'#'*60}")
        print(f"# PROCESSING WALK = {walk}")
        print(f"{'#'*60}")
        
        # Load product/event embeddings (NOT customer embeddings!)
        entity_ids, embeddings = load_cleora_embeddings(walk)
        
        # Create entity to embedding mapping
        entity_to_embedding = {
            str(eid): emb for eid, emb in zip(entity_ids, embeddings)
        }
        
        # Initialize EMDE
        print(f"\n[1/3] Initializing EMDE...")
        emde = EMDESketchGenerator(
            embed_dim=embeddings.shape[1],
            num_subspaces=NUM_SUBSPACES,
            subspace_dim=SUBSPACE_DIM,
            num_bins=NUM_BINS,
            seed=SEED
        )
        print(f"  Sketch dimension: {emde.sketch_dim}")
        
        # Fit on product/event embeddings
        print(f"\n[2/3] Fitting EMDE on product/event embeddings...")
        emde.fit(embeddings)
        
        # Create session sketches
        print(f"\n[3/3] Creating session sketches...")
        past_sketches, future_sketches, portfolio_sketches, customer_ids = \
            create_customer_session_sketches(
                emde=emde,
                entity_to_embedding=entity_to_embedding,
                customers_df=customers_df,
                products_df=products_df,
                events_df=events_df,
                reference_date=reference_date,
                time_decay_lambda=TIME_DECAY_LAMBDA
            )
        
        # Save sketches
        filename = f"emde_session_sketches_walk{walk}.npz"
        filepath = os.path.join(OUTPUT_DIR, filename)
        np.savez_compressed(
            filepath,
            customer_ids=np.array(customer_ids),
            past_sketches=past_sketches,       # Input to foundation model
            future_sketches=future_sketches,   # Target for foundation model
            portfolio_sketches=portfolio_sketches,  # Static product info
            churn_labels=customers_df["churn_label"].values,
            segments=customers_df["segment"].values,
            walk=walk,
            sketch_dim=emde.sketch_dim,
            past_days=PAST_DAYS,
            future_days=FUTURE_DAYS,
            time_decay_lambda=TIME_DECAY_LAMBDA
        )
        print(f"\n✓ Saved to {filename}")
        
        # Visualize
        visualize_session_sketches(
            past_sketches, future_sketches, customers_df, walk, OUTPUT_DIR
        )
    
    print(f"\n{'='*60}")
    print("EMDE SESSION SKETCH GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("\nFiles:")
    for walk in WALK_OPTIONS:
        print(f"  - emde_session_sketches_walk{walk}.npz")
        print(f"  - emde_session_tsne_walk{walk}.png")
    print("\nNPZ Contents:")
    print("  - past_sketches: Input to foundation model (Day 0-25, time-decay)")
    print("  - future_sketches: Target for foundation model (Day 25-30)")
    print("  - portfolio_sketches: Static product portfolio")
    print("  - churn_labels, segments: Labels for downstream tasks")


if __name__ == "__main__":
    main()
