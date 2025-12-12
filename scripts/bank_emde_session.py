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
WALK_OPTIONS = [1, 2, 3, 4]
EMBED_DIR = "../data/embeddings"
OUTPUT_DIR = "../data/emde"

# EMDE Configuration (DLSH - Density-dependent LSH)
# ================================================
# Uses DLSH (Density-dependent Locality Sensitive Hashing) for bucket assignment.
# Unlike classic LSH which uses random bias, DLSH learns bucket boundaries
# from the data distribution using quantile functions.
#
# DLSH ADVANTAGES:
#   1. No empty buckets: Boundaries are placed where data exists
#   2. Dense regions get finer granularity
#   3. Maximum memory efficiency in sketch vector
#   4. Better similarity preservation for non-uniform data
#
# Configuration:
#   N = 64 subspaces (more subspaces = more stable representation)
#   B = 32 bins per subspace (quantile-based, guaranteed no empty bins)
#   Result: 64 × 32 = 2,048 dimensional sketch
#
# Note: DLSH requires fit() to learn quantile boundaries from data
#
NUM_SUBSPACES = 10      # N: Number of random projection subspaces
NUM_BINS = 32           # B: Number of bins per subspace (quantile-based)
# Final sketch_dim = NUM_SUBSPACES × NUM_BINS = 10 × 32 = 320
L2_NORMALIZE = False    # Disabled: Cleora embeddings already L2-normalized, DLSH doesn't require it

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
#   alpha = 0.90 → lambda ≈ 0.1054 (aggressive decay)
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


class DLSHSketchGenerator:
    """
    EMDE with DLSH (Density-dependent Locality Sensitive Hashing).
    
    DLSH uses quantile-based bucketing instead of random bias:
    1. Project embeddings onto random vectors (r_i)
    2. Learn bucket boundaries from data distribution using quantile function
    3. Assign buckets based on learned quantile thresholds
    
    Key Benefits over Classic LSH:
    - No empty buckets: boundaries are placed where data exists
    - Better representation of dense regions
    - Maximum memory efficiency in sketch vector
    
    Requires fit() before use - learns quantile boundaries from training data.
    """
    
    def __init__(
        self, 
        embed_dim: int,
        num_subspaces: int = 64,
        num_bins: int = 32,  # Direct bin count (not 2^bits)
        l2_normalize: bool = True,
        seed: int = 42
    ):
        self.embed_dim = embed_dim
        self.num_subspaces = num_subspaces
        self.num_bins = num_bins
        self.l2_normalize = l2_normalize
        self.seed = seed
        self.sketch_dim = num_subspaces * num_bins
        self.is_fitted = False
        
        # Initialize random projection vectors (one per subspace)
        np.random.seed(seed)
        self.projections = []
        for i in range(num_subspaces):
            # Random projection vector: (embed_dim,)
            r = np.random.randn(embed_dim)
            r = r / np.linalg.norm(r)  # Normalize to unit vector
            self.projections.append(r)
        
        # Quantile boundaries will be learned during fit()
        # Shape: (num_subspaces, num_bins - 1) - thresholds between bins
        self.quantile_boundaries = None
        
        print(f"  Initialized DLSH-EMDE: {num_subspaces} subspaces × {num_bins} bins = {self.sketch_dim} dim")
        print(f"  Mode: Density-dependent (quantile-based bucketing)")
        if l2_normalize:
            print(f"  L2 normalization enabled")
    
    def fit(self, embeddings: np.ndarray):
        """
        Learn quantile boundaries from training data.
        
        For each subspace:
        1. Project all embeddings onto the random vector
        2. Compute quantile thresholds that divide data into equal-sized bins
        
        Args:
            embeddings: (N, embed_dim) training embeddings to learn distribution from
        """
        if self.l2_normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
        
        # Quantile percentages for bin boundaries
        # For num_bins=32, we need 31 boundaries at 1/32, 2/32, ..., 31/32
        quantile_percentages = np.linspace(0, 1, self.num_bins + 1)[1:-1]  # Exclude 0 and 1
        
        self.quantile_boundaries = []
        
        for i in range(self.num_subspaces):
            # Project all embeddings onto this subspace's random vector
            # (N, embed_dim) @ (embed_dim,) = (N,)
            projected = embeddings @ self.projections[i]
            
            # Compute quantile boundaries from the projection distribution
            # These are the threshold values that divide data into equal-sized bins
            boundaries = np.quantile(projected, quantile_percentages)
            self.quantile_boundaries.append(boundaries)
        
        self.quantile_boundaries = np.array(self.quantile_boundaries)  # (num_subspaces, num_bins-1)
        self.is_fitted = True
        
        print(f"  ✓ DLSH fitted on {len(embeddings):,} embeddings")
        print(f"    Learned {self.num_bins - 1} quantile boundaries per subspace")
    
    def _compute_bucket(self, embeddings: np.ndarray, subspace_idx: int) -> np.ndarray:
        """
        Compute bucket indices using learned quantile boundaries.
        
        Args:
            embeddings: (N, embed_dim) array
            subspace_idx: Which subspace to use
            
        Returns:
            (N,) array of bucket indices [0, num_bins-1]
        """
        if not self.is_fitted:
            raise RuntimeError("DLSH must be fitted before use. Call fit() first.")
        
        # Project embeddings onto the random vector
        projected = embeddings @ self.projections[subspace_idx]  # (N,)
        
        # Use searchsorted to find bucket indices based on quantile boundaries
        # searchsorted returns the index where each projected value would be inserted
        # to maintain sorted order, which corresponds to the bucket index
        boundaries = self.quantile_boundaries[subspace_idx]  # (num_bins-1,)
        bucket_indices = np.searchsorted(boundaries, projected)  # (N,)
        
        return bucket_indices
        
    def create_weighted_sketch(
        self, 
        embeddings: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Create time-decay weighted EMDE sketch using DLSH.
        
        DLSH Advantages:
        - Quantile-based boundaries ensure no empty buckets
        - Dense regions get finer granularity
        - Sparse regions are grouped efficiently
        
        Args:
            embeddings: (N, embed_dim) array of embeddings
            weights: (N,) array of time-decay weights
            
        Returns:
            (sketch_dim,) weighted sketch vector
        """
        if len(embeddings) == 0:
            return np.zeros(self.sketch_dim)
        
        # L2 Normalize embeddings
        if self.l2_normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
        
        sketch = np.zeros(self.sketch_dim)
        
        for i in range(self.num_subspaces):
            # Get bucket indices using DLSH (quantile-based)
            bucket_indices = self._compute_bucket(embeddings, i)
            
            # Add weighted counts to sketch
            for j, bucket in enumerate(bucket_indices):
                sketch_idx = i * self.num_bins + bucket
                sketch[sketch_idx] += weights[j]
        
        # L1 Normalize
        total_weight = weights.sum()
        if total_weight > 0:
            sketch = sketch / total_weight
        
        return sketch


# Keep the old class as an alias for backward compatibility
EMDESketchGenerator = DLSHSketchGenerator


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
        
        # Initialize DLSH-EMDE (requires fit)
        print(f"\n[1/3] Initializing DLSH-EMDE...")
        emde = DLSHSketchGenerator(
            embed_dim=embeddings.shape[1],
            num_subspaces=NUM_SUBSPACES,
            num_bins=NUM_BINS,
            l2_normalize=L2_NORMALIZE,
            seed=SEED
        )
        
        # Fit DLSH on product/event embeddings to learn quantile boundaries
        print(f"\n[2/3] Fitting DLSH on embedding distribution...")
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
