#!/usr/bin/env python3
"""
Future UBR Prediction with Monad-EMDE Residual FFN
===================================================
Implements the paper-specified architecture for EMDE-based prediction.

Architecture (per Monad/EMDE papers):
1. Input: L2-Normalized sparse density vectors
2. Hidden: 3 layers × ~3000 neurons with Residual connections
   - Each block: Linear → BatchNorm → LeakyReLU → Skip Connection
3. Output: LogSoftmax for KL-Divergence training

Loss: KL-Divergence (proper for probability distribution comparison)

Author: Generated for TEB-ARF Project
Date: 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = "../data/emde"
OUTPUT_DIR = "../data/ffn_model"
WALK = 4  # Using walk4 with DLSH

# Model Configuration (Monad-EMDE Paper Specifications)
# Input: Past UBR + Portfolio (dimensions read from data)
# Output: Future UBR (dimension read from data)
HIDDEN_DIM = 3000        # ~3000 neurons per hidden layer (paper recommendation)
NUM_HIDDEN_LAYERS = 3    # 3 hidden layers with residual connections
DROPOUT = 0.1            # Light dropout for regularization
LEAKY_RELU_SLOPE = 0.01  # LeakyReLU negative slope

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class UBRDataset(Dataset):
    """Dataset for UBR prediction task with L2 normalization."""
    
    def __init__(self, past_sketches, portfolio_sketches, future_sketches):
        # Concatenate past + portfolio as input
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1).astype(np.float32)
        self.y = future_sketches.astype(np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class ResidualBlock(nn.Module):
    """
    Residual FFN Block (Monad-EMDE Paper).
    
    Structure:
        Input → Linear → BatchNorm → LeakyReLU → (Dropout) → + Input → Output
                                                              ↑
                                                    Skip Connection
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Residual connection: output = block(x) + x
        return self.block(x) + x


class MonadEMDEPredictor(nn.Module):
    """
    Monad-EMDE Residual Feed-Forward Network.
    
    Paper-specified architecture for sparse density vector prediction:
    1. L2 Input Normalization - Stabilizes training
    2. Input Projection - Projects to hidden dimension
    3. Residual Blocks - 3 layers with skip connections
    4. Output Projection - Projects to output dimension
    5. LogSoftmax - Converts to log-probabilities for KL-Divergence
    
    The residual connections allow the network to learn
    non-linear density transitions while preserving input information.
    """
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 3000,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Input projection (to hidden dimension)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)
        )
        
        # Residual blocks (3 layers as per paper)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # LogSoftmax for KL-Divergence
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ MonadEMDE Architecture:")
        print(f"  Input dim:  {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Output dim: {output_dim}")
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass with L2 input normalization.
        
        Args:
            x: (batch, input_dim) - concatenated [past_ubr, portfolio]
            
        Returns:
            (batch, output_dim) - log-probabilities of future UBR
        """
        # 1. L2 Normalize input (paper specification)
        x = F.normalize(x, p=2, dim=-1)
        
        # 2. Input projection
        h = self.input_proj(x)
        
        # 3. Residual blocks
        for block in self.residual_blocks:
            h = block(h)
        
        # 4. Output projection
        out = self.output_proj(h)
        
        # 5. LogSoftmax for KL-Divergence
        return self.log_softmax(out)


def normalize_to_distribution(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize EMDE sketches to valid probability distributions."""
    x = torch.clamp(x, min=0)
    x = x + eps
    x = x / x.sum(dim=-1, keepdim=True)
    return x


def load_data(walk: int = 4):
    """Load EMDE session sketches."""
    filepath = os.path.join(DATA_DIR, f"emde_session_sketches_walk{walk}.npz")
    data = np.load(filepath, allow_pickle=True)
    
    past_sketches = data["past_sketches"]
    future_sketches = data["future_sketches"]
    portfolio_sketches = data["portfolio_sketches"]
    customer_ids = data["customer_ids"]
    churn_labels = data["churn_labels"]
    segments = data["segments"]
    
    print(f"✓ Loaded walk{walk} session sketches:")
    print(f"  Past UBR: {past_sketches.shape}")
    print(f"  Future UBR: {future_sketches.shape}")
    print(f"  Portfolio: {portfolio_sketches.shape}")
    print(f"  Customers: {len(customer_ids)}")
    
    return past_sketches, future_sketches, portfolio_sketches, customer_ids, churn_labels, segments


def train_model(model, train_loader, val_loader, epochs, patience, device):
    """Train with KL-Divergence loss."""
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("TRAINING (Monad-EMDE Residual FFN)")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Normalize target to probability distribution
            y_prob = normalize_to_distribution(y_batch)
            
            optimizer.zero_grad()
            pred_logprob = model(X_batch)
            loss = criterion(pred_logprob, y_prob)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_prob = normalize_to_distribution(y_batch)
                pred_logprob = model(X_batch)
                loss = criterion(pred_logprob, y_prob)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {lr:.2e} | "
                  f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_logprob = model(X_batch)
            pred_prob = torch.exp(pred_logprob)
            all_preds.append(pred_prob.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Calculate cosine similarity
    similarities = []
    for i in range(len(preds)):
        sim = cosine_similarity(preds[i:i+1], targets[i:i+1])[0, 0]
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Cosine Similarity: {similarities.mean():.4f}")
    print(f"Std Cosine Similarity:  {similarities.std():.4f}")
    print(f"Min Cosine Similarity:  {similarities.min():.4f}")
    print(f"Max Cosine Similarity:  {similarities.max():.4f}")
    print(f"Median Cosine Similarity: {np.median(similarities):.4f}")
    
    return preds, targets, similarities


def plot_results(train_losses, val_losses, similarities, output_dir):
    """Plot training curves and results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.plot(train_losses, label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(val_losses, label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL-Divergence Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.hist(similarities, bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.axvline(similarities.mean(), color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {similarities.mean():.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Future UBR Prediction Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    sorted_sims = np.sort(similarities)
    cdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax.plot(sorted_sims, cdf, color='#9b59b6', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(np.median(similarities), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Median: {np.median(similarities):.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Similarity CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'future_ubr_training.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved training plot to {filepath}")


def main():
    print("="*60)
    print("FUTURE UBR PREDICTION")
    print("Monad-EMDE Residual FFN Architecture")
    print("="*60)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    past, future, portfolio, customer_ids, churn_labels, segments = load_data(WALK)
    
    # Create dataset
    dataset = UBRDataset(past, portfolio, future)
    
    # Split
    n_samples = len(dataset)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    print(f"\n✓ Dataset splits:")
    print(f"  Train: {n_train:,}")
    print(f"  Val:   {n_val:,}")
    print(f"  Test:  {n_test:,}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_dim = past.shape[1] + portfolio.shape[1]
    output_dim = future.shape[1]
    
    model = MonadEMDEPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_HIDDEN_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    
    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, EPOCHS, PATIENCE, DEVICE
    )
    
    # Evaluate
    preds, targets, similarities = evaluate_model(model, test_loader, DEVICE)
    
    # Plot
    plot_results(train_losses, val_losses, similarities, OUTPUT_DIR)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'future_ubr_model_walk{WALK}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_HIDDEN_LAYERS,
        'dropout': DROPOUT,
        'walk': WALK,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_similarities': similarities,
        'mean_similarity': similarities.mean(),
        'architecture': 'MonadEMDE_ResidualFFN'
    }, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, f'future_ubr_predictions_walk{WALK}.npz')
    np.savez_compressed(pred_path, predictions=preds, targets=targets, similarities=similarities)
    print(f"✓ Saved predictions to {pred_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nArchitecture: Monad-EMDE Residual FFN")
    print(f"  - L2 Input Normalization ✓")
    print(f"  - 3 Residual Blocks (Linear→BatchNorm→LeakyReLU→Skip) ✓")
    print(f"  - Hidden dim: {HIDDEN_DIM}")
    print(f"  - LogSoftmax output ✓")
    print(f"\nOutput: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
