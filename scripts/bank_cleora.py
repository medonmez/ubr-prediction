#!/usr/bin/env python3
"""
Bank Cleora Embedding Generator
===============================
Generates entity embeddings using Cleora for customers, products, and events.
Supports multiple walk configurations with t-SNE visualization.

"""

import os
import numpy as np
import pandas as pd
from pycleora import SparseMatrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Configuration
EMBED_DIM = 1024
WALK_OPTIONS = [1, 4]  # Generate embeddings for both walk counts
INPUT_FILE = "../data/cleora_hyperedges.txt"
OUTPUT_DIR = "../data/embeddings"

# t-SNE configuration
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42


def load_hyperedges(filepath: str) -> list:
    """Load hyperedges from file."""
    with open(filepath, "r") as f:
        hyperedges = [line.strip() for line in f if line.strip()]
    print(f"✓ Loaded {len(hyperedges):,} hyperedges from {filepath}")
    return hyperedges


def load_customer_data() -> pd.DataFrame:
    """Load customer data for visualization labels."""
    customers_df = pd.read_csv("../data/bank_customers.csv")
    return customers_df


def create_cleora_embeddings(hyperedges: list, embed_dim: int, num_walks: int):
    """
    Create Cleora embeddings using Markov random walks.
    """
    
    print(f"\n{'='*50}")
    print("CLEORA EMBEDDING GENERATION")
    print(f"{'='*50}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of walks: {num_walks}")
    print()
    
    print("[1/4] Building sparse matrix from hyperedges...")
    mat = SparseMatrix.from_iterator(iter(hyperedges), columns='complex::reflexive::entity')
    
    entity_ids = mat.entity_ids
    customers = [e for e in entity_ids if e.startswith("C")]
    non_customers = [e for e in entity_ids if not e.startswith("C")]
    
    print(f"      Matrix: {len(entity_ids):,} entities")
    print(f"      - Customers: {len(customers):,}")
    print(f"      - Products & Events: {len(non_customers):,}")
    
    print(f"\n[2/4] Initializing {embed_dim}-dimensional embeddings...")
    embeddings = mat.initialize_deterministically(embed_dim)
    
    print(f"\n[3/4] Performing {num_walks} Markov random walks...")
    for i in range(num_walks):
        embeddings = mat.left_markov_propagate(embeddings)
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        print(f"      Walk {i+1}/{num_walks} complete")
    
    print(f"\n[4/4] Embeddings generated! Shape: {embeddings.shape}")
    
    return mat.entity_ids, embeddings


def save_embeddings_npz(entity_ids: list, embeddings: np.ndarray, output_dir: str, num_walks: int):
    """Save embeddings to NPZ file only."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"cleora_embeddings_walk{num_walks}.npz"
    filepath = os.path.join(output_dir, filename)
    
    np.savez_compressed(
        filepath,
        entity_ids=np.array(entity_ids),
        embeddings=embeddings,
        num_walks=num_walks,
        embed_dim=embeddings.shape[1]
    )
    print(f"✓ Saved embeddings to {filename}")
    
    return filepath


def create_tsne_visualization(
    entity_ids: list, 
    embeddings: np.ndarray, 
    customers_df: pd.DataFrame,
    num_walks: int,
    output_dir: str
):
    """Create t-SNE visualization of embeddings."""
    
    print(f"\n{'='*50}")
    print(f"t-SNE VISUALIZATION (Walk {num_walks})")
    print(f"{'='*50}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate customers and products/events
    customer_mask = np.array([e.startswith("C") for e in entity_ids])
    customer_ids = [e for e in entity_ids if e.startswith("C")]
    customer_embeddings = embeddings[customer_mask]
    
    product_event_ids = [e for e in entity_ids if not e.startswith("C")]
    product_event_embeddings = embeddings[~customer_mask]
    
    # =========================================================================
    # VISUALIZATION 1: All Entities (Customers + Products + Events)
    # =========================================================================
    print("\n[1/3] Running t-SNE on all entities...")
    
    # Sample customers for faster visualization (max 2000)
    sample_size = min(2000, len(customer_ids))
    sample_indices = np.random.choice(len(customer_ids), sample_size, replace=False)
    sampled_customer_embeddings = customer_embeddings[sample_indices]
    sampled_customer_ids = [customer_ids[i] for i in sample_indices]
    
    # Combine sampled customers with all products/events
    all_embeddings = np.vstack([sampled_customer_embeddings, product_event_embeddings])
    all_ids = sampled_customer_ids + product_event_ids
    
    tsne = TSNE(
        n_components=2, 
        perplexity=min(TSNE_PERPLEXITY, len(all_embeddings) - 1),
        max_iter=TSNE_N_ITER, 
        random_state=TSNE_RANDOM_STATE,
        init='pca',
        learning_rate='auto'
    )
    all_tsne = tsne.fit_transform(all_embeddings)
    
    # Create labels
    entity_types = ['Customer'] * len(sampled_customer_ids) + ['Product/Event'] * len(product_event_ids)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {'Customer': '#3498db', 'Product/Event': '#e74c3c'}
    sizes = {'Customer': 30, 'Product/Event': 150}
    
    for etype in ['Customer', 'Product/Event']:
        mask = np.array([t == etype for t in entity_types])
        ax.scatter(
            all_tsne[mask, 0], 
            all_tsne[mask, 1], 
            c=colors[etype], 
            s=sizes[etype],
            alpha=0.6 if etype == 'Customer' else 0.9,
            label=etype,
            edgecolors='white' if etype == 'Product/Event' else 'none',
            linewidths=1
        )
    
    # Add labels for products/events
    texts = []
    for i, eid in enumerate(product_event_ids):
        idx = len(sampled_customer_ids) + i
        texts.append(ax.text(
            all_tsne[idx, 0], 
            all_tsne[idx, 1],
            eid.replace('_', '\n'),
            fontsize=8,
            ha='center',
            va='center',
            alpha=0.9,
            fontweight='bold'
        ))
    
    # Auto-adjust text positions to avoid overlap
    print("      Adjusting labels (this may take a moment)...")
    from adjustText import adjust_text
    adjust_text(
        texts, 
        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
        expand_points=(1.5, 1.5),
        force_points=(0.2, 0.2)
    )
    
    ax.set_title(f'Cleora Embeddings t-SNE (Walk={num_walks})\nCustomers + Products + Events', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    filepath1 = os.path.join(output_dir, f'tsne_all_entities_walk{num_walks}.png')
    plt.savefig(filepath1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: tsne_all_entities_walk{num_walks}.png")
    
    # =========================================================================
    # VISUALIZATION 2: Customers by Segment
    # =========================================================================
    print("[2/3] Running t-SNE on customers by segment...")
    
    # Get segment info
    segment_lookup = dict(zip(customers_df['customer_id'], customers_df['segment']))
    sampled_segments = [segment_lookup.get(cid, 'unknown') for cid in sampled_customer_ids]
    
    # t-SNE on sampled customers only
    tsne_customers = TSNE(
        n_components=2, 
        perplexity=min(TSNE_PERPLEXITY, sample_size - 1),
        max_iter=TSNE_N_ITER, 
        random_state=TSNE_RANDOM_STATE,
        init='pca',
        learning_rate='auto'
    )
    customer_tsne = tsne_customers.fit_transform(sampled_customer_embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    segment_colors = {
        'mass': '#3498db',
        'affluent': '#2ecc71', 
        'private': '#9b59b6',
        'business': '#e67e22'
    }
    
    for segment in ['mass', 'affluent', 'private', 'business']:
        mask = np.array([s == segment for s in sampled_segments])
        if mask.sum() > 0:
            ax.scatter(
                customer_tsne[mask, 0], 
                customer_tsne[mask, 1], 
                c=segment_colors[segment], 
                s=40,
                alpha=0.6,
                label=f'{segment.capitalize()} ({mask.sum():,})'
            )
    
    ax.set_title(f'Cleora Customer Embeddings by Segment (Walk={num_walks})', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    filepath2 = os.path.join(output_dir, f'tsne_customers_segment_walk{num_walks}.png')
    plt.savefig(filepath2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: tsne_customers_segment_walk{num_walks}.png")
    
    # =========================================================================
    # VISUALIZATION 3: Customers by Churn Label
    # =========================================================================
    print("[3/3] Running t-SNE on customers by churn...")
    
    churn_lookup = dict(zip(customers_df['customer_id'], customers_df['churn_label']))
    sampled_churn = [churn_lookup.get(cid, 0) for cid in sampled_customer_ids]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    churn_colors = {0: '#2ecc71', 1: '#e74c3c'}
    churn_labels = {0: 'Retained', 1: 'Churned'}
    
    for churn in [0, 1]:
        mask = np.array([c == churn for c in sampled_churn])
        if mask.sum() > 0:
            ax.scatter(
                customer_tsne[mask, 0], 
                customer_tsne[mask, 1], 
                c=churn_colors[churn], 
                s=40,
                alpha=0.6,
                label=f'{churn_labels[churn]} ({mask.sum():,})'
            )
    
    ax.set_title(f'Cleora Customer Embeddings by Churn (Walk={num_walks})', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    filepath3 = os.path.join(output_dir, f'tsne_customers_churn_walk{num_walks}.png')
    plt.savefig(filepath3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: tsne_customers_churn_walk{num_walks}.png")
    
    return [filepath1, filepath2, filepath3]


def main():
    """Main execution function."""
    
    print("="*50)
    print("BANK CLEORA EMBEDDING GENERATOR")
    print("Customer + Product + Event Graph")
    print("="*50)
    print()
    
    # Load data
    hyperedges = load_hyperedges(INPUT_FILE)
    customers_df = load_customer_data()
    
    all_visualizations = []
    
    # Generate embeddings for each walk configuration
    for num_walks in WALK_OPTIONS:
        print(f"\n{'#'*50}")
        print(f"# WALK = {num_walks}")
        print(f"{'#'*50}")
        
        # Create embeddings
        entity_ids, embeddings = create_cleora_embeddings(
            hyperedges,
            embed_dim=EMBED_DIM,
            num_walks=num_walks
        )
        
        # Save to NPZ only
        print(f"\nSaving embeddings...")
        save_embeddings_npz(entity_ids, embeddings, OUTPUT_DIR, num_walks)
        
        # Create t-SNE visualizations
        viz_files = create_tsne_visualization(
            entity_ids, embeddings, customers_df, num_walks, OUTPUT_DIR
        )
        all_visualizations.extend(viz_files)
    
    print(f"\n{'='*50}")
    print("COMPLETE!")
    print(f"{'='*50}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nEmbeddings:")
    for num_walks in WALK_OPTIONS:
        print(f"  - cleora_embeddings_walk{num_walks}.npz")
    print("\nVisualizations:")
    for num_walks in WALK_OPTIONS:
        print(f"  - tsne_all_entities_walk{num_walks}.png")
        print(f"  - tsne_customers_segment_walk{num_walks}.png")
        print(f"  - tsne_customers_churn_walk{num_walks}.png")


if __name__ == "__main__":
    main()
