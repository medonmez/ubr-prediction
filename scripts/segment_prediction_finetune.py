#!/usr/bin/env python3
"""
Segment Prediction Model (Fine-Tuning)
======================================
Predicts customer segment (mass, affluent, business, private) using
pre-trained Monad-EMDE Foundation Model backbone.

Architecture:
    Foundation Model Backbone → Classification Head → 4-class Softmax

This is a multi-class classification task with imbalanced classes.
Uses weighted CrossEntropyLoss to handle class imbalance.

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Configuration
DATA_DIR = "../data/emde"
FOUNDATION_MODEL_DIR = "../data/ffn_model"
OUTPUT_DIR = "../data/segment_model"
WALK = 4  # Using walk4 with DLSH

# Segment mapping
SEGMENT_CLASSES = ['mass', 'affluent', 'business', 'private']
NUM_CLASSES = len(SEGMENT_CLASSES)

# Fine-tuning Configuration
FREEZE_BACKBONE = False
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 10
LEAKY_RELU_SLOPE = 0.01

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class SegmentDataset(Dataset):
    """Dataset for segment prediction task."""
    
    def __init__(self, past_sketches, portfolio_sketches, segment_labels):
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1).astype(np.float32)
        
        # Convert string labels to integers
        segment_to_idx = {seg: idx for idx, seg in enumerate(SEGMENT_CLASSES)}
        self.y = np.array([segment_to_idx[s] for s in segment_labels], dtype=np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class ResidualBlock(nn.Module):
    """Residual FFN Block (Monad-EMDE Paper)."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.block(x) + x


class SegmentPredictor(nn.Module):
    """
    Segment Prediction Model with Monad-EMDE Backbone.
    
    Architecture:
        L2-Norm Input → Monad-EMDE Backbone → Classification Head → 4-class Logits
    """
    
    def __init__(self, foundation_model_path: str, freeze_backbone: bool = False):
        super().__init__()
        
        # Load foundation model checkpoint
        checkpoint = torch.load(foundation_model_path, map_location='cpu', weights_only=False)
        
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        hidden_dim = checkpoint['hidden_dim']
        num_layers = checkpoint['num_layers']
        dropout = checkpoint['dropout']
        
        print(f"Loading Monad-EMDE foundation model from {foundation_model_path}")
        print(f"  Input dim:  {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Output dim: {output_dim}")
        
        # Rebuild backbone (same architecture as MonadEMDEPredictor, without LogSoftmax)
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection (without LogSoftmax)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Load pre-trained weights
        state_dict = checkpoint['model_state_dict']
        
        # Map weights from saved model
        self.input_proj.load_state_dict({
            k.replace('input_proj.', ''): v 
            for k, v in state_dict.items() if k.startswith('input_proj.')
        })
        
        for i, block in enumerate(self.residual_blocks):
            block_prefix = f'residual_blocks.{i}.'
            block.load_state_dict({
                k.replace(block_prefix, ''): v 
                for k, v in state_dict.items() if k.startswith(block_prefix)
            })
        
        self.output_proj.load_state_dict({
            k.replace('output_proj.', ''): v 
            for k, v in state_dict.items() if k.startswith('output_proj.')
        })
        
        print("✓ Loaded pre-trained Monad-EMDE weights")
        
        # Freeze backbone if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.input_proj.parameters():
                param.requires_grad = False
            for block in self.residual_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            for param in self.output_proj.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        else:
            print("✓ Backbone unfrozen (full fine-tuning)")
        
        # Classification head for 4-class segment prediction
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)  # 4 classes
        )
        
    def forward(self, x):
        # L2 Normalize input (same as foundation model)
        x = F.normalize(x, p=2, dim=-1)
        
        # Backbone forward
        h = self.input_proj(x)
        for block in self.residual_blocks:
            h = block(h)
        features = self.output_proj(h)
        
        # Classification head
        logits = self.classifier(features)
        return logits


def load_data(walk: int = 4):
    """Load EMDE session sketches and segment labels."""
    filepath = os.path.join(DATA_DIR, f"emde_session_sketches_walk{walk}.npz")
    data = np.load(filepath, allow_pickle=True)
    
    past_sketches = data["past_sketches"]
    portfolio_sketches = data["portfolio_sketches"]
    segments = data["segments"]
    
    print(f"✓ Loaded walk{walk} data:")
    print(f"  Past UBR: {past_sketches.shape}")
    print(f"  Portfolio: {portfolio_sketches.shape}")
    print(f"  Segments: {segments.shape}")
    
    # Print segment distribution
    unique, counts = np.unique(segments, return_counts=True)
    print(f"  Segment distribution:")
    for seg, count in zip(unique, counts):
        print(f"    {seg}: {count} ({count/len(segments)*100:.1f}%)")
    
    return past_sketches, portfolio_sketches, segments


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Inverse frequency weighting
    weights = total / (len(classes) * counts)
    
    # Create weight tensor in class order
    weight_dict = {cls: w for cls, w in zip(classes, weights)}
    class_weights = torch.tensor([weight_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    
    return class_weights


def train_model(model, train_loader, val_loader, epochs, patience, device):
    """Train with weighted CrossEntropyLoss."""
    
    # Calculate class weights from training data
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy())
    all_labels = np.array(all_labels)
    
    class_weights = compute_class_weights(all_labels).to(device)
    
    print(f"\n✓ Class weights for imbalanced training:")
    for i, seg in enumerate(SEGMENT_CLASSES):
        print(f"  {seg}: {class_weights[i].item():.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Different LR for backbone and classifier
    if model.freeze_backbone:
        optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE * 10)
    else:
        backbone_params = list(model.input_proj.parameters()) + \
                         list(model.residual_blocks.parameters()) + \
                         list(model.output_proj.parameters())
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': LEARNING_RATE},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 10}
        ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    
    print(f"\n{'='*60}")
    print("FINE-TUNING FOR SEGMENT PREDICTION")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                pred_classes = torch.argmax(logits, dim=-1)
                all_preds.extend(pred_classes.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Best validation F1: {max(val_f1s):.4f}")
    
    return train_losses, val_losses, val_accs, val_f1s


def evaluate_model(model, test_loader, device):
    """Evaluate segment prediction model."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=-1)
            pred_classes = torch.argmax(logits, dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Multi-class AUC (One-vs-Rest)
    labels_binary = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try:
        auc_ovr = roc_auc_score(labels_binary, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc_ovr = 0.0
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision (M):   {precision_macro:.4f}")
    print(f"Recall (M):      {recall_macro:.4f}")
    print(f"F1 Macro:        {f1_macro:.4f}")
    print(f"F1 Weighted:     {f1_weighted:.4f}")
    print(f"ROC-AUC (OvR):   {auc_ovr:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=SEGMENT_CLASSES, zero_division=0))
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_ovr': auc_ovr
    }
    
    return preds, probs, labels, metrics


def plot_results(train_losses, val_losses, val_accs, val_f1s, preds, probs, labels, output_dir):
    """Plot training curves and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # ==========================================================================
    # FIGURE 1: Training Metrics (2x2)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training & Validation Loss
    ax = axes[0, 0]
    ax.plot(train_losses, label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(val_losses, label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CrossEntropy Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation Accuracy & F1
    ax = axes[0, 1]
    ax.plot(val_accs, label='Val Accuracy', color='#2ecc71', linewidth=2)
    ax.plot(val_f1s, label='Val F1 (Macro)', color='#9b59b6', linewidth=2)
    ax.axhline(max(val_f1s), color='#e74c3c', linestyle='--', linewidth=1, label=f'Best F1: {max(val_f1s):.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Accuracy & F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax = axes[1, 0]
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=SEGMENT_CLASSES, 
                yticklabels=SEGMENT_CLASSES,
                annot_kws={'size': 12})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # 4. Per-class Distribution
    ax = axes[1, 1]
    segment_colors = {'mass': '#3498db', 'affluent': '#2ecc71', 'private': '#9b59b6', 'business': '#e67e22'}
    
    for i, seg in enumerate(SEGMENT_CLASSES):
        mask = labels == i
        if mask.sum() > 0:
            ax.hist(probs[mask, i], bins=30, alpha=0.5, label=f'{seg}', 
                   color=segment_colors[seg], edgecolor='white')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Probability Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath1 = os.path.join(output_dir, 'segment_prediction_results.png')
    plt.savefig(filepath1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved results plot to {filepath1}")
    
    # ==========================================================================
    # FIGURE 2: Multi-class ROC Curves (2x2)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    labels_binary = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    
    for i, (seg, color) in enumerate(zip(SEGMENT_CLASSES, colors)):
        ax = axes[i // 2, i % 2]
        
        fpr, tpr, _ = roc_curve(labels_binary[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve: {seg.capitalize()}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath2 = os.path.join(output_dir, 'segment_roc_curves.png')
    plt.savefig(filepath2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved ROC curves to {filepath2}")


def main():
    print("="*60)
    print("SEGMENT PREDICTION WITH MONAD-EMDE FINE-TUNING")
    print("="*60)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    past, portfolio, segments = load_data(WALK)
    
    dataset = SegmentDataset(past, portfolio, segments)
    
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
    
    foundation_model_path = os.path.join(FOUNDATION_MODEL_DIR, f'future_ubr_model_walk{WALK}.pt')
    
    print(f"\n✓ Creating segment predictor:")
    print(f"  Freeze backbone: {FREEZE_BACKBONE}")
    
    model = SegmentPredictor(
        foundation_model_path=foundation_model_path,
        freeze_backbone=FREEZE_BACKBONE
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model, train_loader, val_loader, EPOCHS, PATIENCE, DEVICE
    )
    
    preds, probs, labels, metrics = evaluate_model(model, test_loader, DEVICE)
    
    plot_results(train_losses, val_losses, val_accs, val_f1s, preds, probs, labels, OUTPUT_DIR)
    
    model_path = os.path.join(OUTPUT_DIR, f'segment_predictor_walk{WALK}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'freeze_backbone': FREEZE_BACKBONE,
        'walk': WALK,
        'metrics': metrics,
        'segment_classes': SEGMENT_CLASSES,
        'architecture': 'MonadEMDE_SegmentPredictor'
    }, model_path)
    print(f"✓ Saved model to {model_path}")
    
    pred_path = os.path.join(OUTPUT_DIR, f'segment_predictions_walk{WALK}.npz')
    np.savez_compressed(
        pred_path, 
        predictions=preds, 
        probabilities=probs,
        labels=labels, 
        metrics=metrics,
        segment_classes=SEGMENT_CLASSES
    )
    print(f"✓ Saved predictions to {pred_path}")
    
    print(f"\n{'='*60}")
    print("SEGMENT PREDICTION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
