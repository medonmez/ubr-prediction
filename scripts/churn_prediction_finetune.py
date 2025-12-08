#!/usr/bin/env python3

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
    roc_auc_score, confusion_matrix, classification_report
)

# Configuration
DATA_DIR = "../data/emde"
FOUNDATION_MODEL_DIR = "../data/ffn_model"
OUTPUT_DIR = "../data/churn_model"
WALK = 4

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


class ChurnDataset(Dataset):
    """Dataset for churn prediction task."""
    
    def __init__(self, past_sketches, portfolio_sketches, churn_labels):
        self.X = np.concatenate([past_sketches, portfolio_sketches], axis=1).astype(np.float32)
        self.y = churn_labels.astype(np.float32)
        
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


class ChurnPredictor(nn.Module):
    """
    Churn Prediction Model with Monad-EMDE Backbone.
    
    Architecture:
        L2-Norm Input → Monad-EMDE Backbone → Classification Head → Logits
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
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
        return logits.squeeze(-1)


def load_data(walk: int = 4):
    """Load EMDE session sketches and churn labels."""
    filepath = os.path.join(DATA_DIR, f"emde_session_sketches_walk{walk}.npz")
    data = np.load(filepath, allow_pickle=True)
    
    past_sketches = data["past_sketches"]
    portfolio_sketches = data["portfolio_sketches"]
    churn_labels = data["churn_labels"]
    
    print(f"✓ Loaded walk{walk} data:")
    print(f"  Past UBR: {past_sketches.shape}")
    print(f"  Portfolio: {portfolio_sketches.shape}")
    print(f"  Churn labels: {churn_labels.shape}")
    print(f"  Churn rate: {churn_labels.mean()*100:.2f}%")
    
    return past_sketches, portfolio_sketches, churn_labels


def train_model(model, train_loader, val_loader, epochs, patience, device):
    """Train with weighted BCEWithLogitsLoss."""
    
    # Calculate class weights
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy())
    
    all_labels = np.array(all_labels)
    n_total = len(all_labels)
    n_positive = all_labels.sum()
    n_negative = n_total - n_positive
    
    pos_weight = torch.tensor([n_negative / n_positive], device=device)
    
    print(f"\n✓ Class distribution:")
    print(f"  Negative (retained): {int(n_negative)} ({n_negative/n_total*100:.1f}%)")
    print(f"  Positive (churned):  {int(n_positive)} ({n_positive/n_total*100:.1f}%)")
    print(f"  Positive weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
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
    
    train_losses, val_losses, val_aucs = [], [], []
    
    print(f"\n{'='*60}")
    print("FINE-TUNING FOR CHURN PREDICTION")
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
                
                pred_probs = torch.sigmoid(logits)
                all_preds.extend(pred_probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)
        
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
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Best validation AUC: {max(val_aucs):.4f}")
    
    return train_losses, val_losses, val_aucs


def evaluate_model(model, test_loader, device):
    """Evaluate churn prediction model."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            pred_probs = torch.sigmoid(logits)
            all_preds.extend(pred_probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    binary_preds = (preds >= 0.5).astype(int)
    
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    auc = roc_auc_score(labels, preds)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(labels, binary_preds))
    print(f"\nClassification Report:")
    print(classification_report(labels, binary_preds, target_names=['Retained', 'Churned'], zero_division=0))
    
    return preds, labels, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


def plot_results(train_losses, val_losses, val_aucs, preds, labels, output_dir):
    """Plot training curves and comprehensive evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    from sklearn.metrics import (
        roc_curve, precision_recall_curve, average_precision_score,
        confusion_matrix
    )
    from sklearn.calibration import calibration_curve
    import seaborn as sns
    
    # ==========================================================================
    # FIGURE 1: Training Metrics (2x2)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training & Validation Loss
    ax = axes[0, 0]
    ax.plot(train_losses, label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(val_losses, label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation AUC over epochs
    ax = axes[0, 1]
    ax.plot(val_aucs, color='#2ecc71', linewidth=2)
    ax.axhline(max(val_aucs), color='#e74c3c', linestyle='--', linewidth=1, label=f'Best: {max(val_aucs):.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Validation AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    ax = axes[1, 0]
    ax.hist(preds[labels == 0], bins=50, alpha=0.7, label='Retained', color='#2ecc71', edgecolor='white')
    ax.hist(preds[labels == 1], bins=50, alpha=0.7, label='Churned', color='#e74c3c', edgecolor='white')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Churn Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution by True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. ROC Curve
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    ax.plot(fpr, tpr, color='#9b59b6', linewidth=2, label=f'ROC (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath1 = os.path.join(output_dir, 'churn_prediction_results.png')
    plt.savefig(filepath1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved results plot to {filepath1}")
    
    # ==========================================================================
    # FIGURE 2: Advanced KPIs (2x3)
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Precision-Recall Curve
    ax = axes[0, 0]
    precision, recall, pr_thresholds = precision_recall_curve(labels, preds)
    avg_precision = average_precision_score(labels, preds)
    ax.plot(recall, precision, color='#e67e22', linewidth=2, label=f'AP={avg_precision:.4f}')
    ax.axhline(labels.mean(), color='gray', linestyle='--', label=f'Baseline: {labels.mean():.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix Heatmap
    ax = axes[0, 1]
    binary_preds = (preds >= 0.5).astype(int)
    cm = confusion_matrix(labels, binary_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churned'], 
                yticklabels=['Retained', 'Churned'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Threshold=0.5)')
    
    # 3. Calibration Plot
    ax = axes[0, 2]
    try:
        prob_true, prob_pred = calibration_curve(labels, preds, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, 's-', color='#3498db', linewidth=2, markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Calibration Error:\n{str(e)[:50]}', ha='center', va='center')
        ax.set_title('Calibration Plot (Error)')
    
    # 4. Metrics vs Threshold
    ax = axes[1, 0]
    thresholds = np.linspace(0.01, 0.99, 50)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred_t = (preds >= t).astype(int)
        precisions.append(precision_score(labels, pred_t, zero_division=0))
        recalls.append(recall_score(labels, pred_t, zero_division=0))
        f1s.append(f1_score(labels, pred_t, zero_division=0))
    
    ax.plot(thresholds, precisions, label='Precision', color='#3498db', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
    ax.plot(thresholds, f1s, label='F1', color='#2ecc71', linewidth=2)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    best_f1_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_f1_idx], color='#9b59b6', linestyle=':', linewidth=2, 
               label=f'Best F1 @ {thresholds[best_f1_idx]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    
    # 5. Lift Chart
    ax = axes[1, 1]
    n_samples = len(preds)
    sorted_indices = np.argsort(preds)[::-1]
    sorted_labels = labels[sorted_indices]
    cumulative_churn = np.cumsum(sorted_labels)
    total_churn = labels.sum()
    
    percentiles = np.arange(1, n_samples + 1) / n_samples * 100
    cumulative_captured = cumulative_churn / total_churn * 100
    random_baseline = percentiles
    
    ax.plot(percentiles, cumulative_captured, color='#9b59b6', linewidth=2, label='Model')
    ax.plot(percentiles, random_baseline, 'k--', linewidth=1, label='Random')
    ax.fill_between(percentiles, random_baseline, cumulative_captured, alpha=0.2, color='#9b59b6')
    ax.set_xlabel('% of Customers Contacted (Sorted by Score)')
    ax.set_ylabel('% of Churners Captured')
    ax.set_title('Cumulative Gain (Lift) Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add key lift values
    for pct in [10, 20, 30]:
        idx = int(n_samples * pct / 100)
        captured = cumulative_captured[idx-1]
        ax.annotate(f'{captured:.1f}%', (pct, captured), textcoords='offset points', 
                   xytext=(5, 5), fontsize=9, color='#9b59b6')
    
    # 6. Summary Metrics Table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate key metrics
    binary_preds = (preds >= 0.5).astype(int)
    metrics_data = [
        ['ROC-AUC', f'{roc_auc_score(labels, preds):.4f}'],
        ['Average Precision', f'{avg_precision:.4f}'],
        ['Accuracy', f'{accuracy_score(labels, binary_preds):.4f}'],
        ['Precision (Churn)', f'{precision_score(labels, binary_preds, zero_division=0):.4f}'],
        ['Recall (Churn)', f'{recall_score(labels, binary_preds, zero_division=0):.4f}'],
        ['F1 Score', f'{f1_score(labels, binary_preds, zero_division=0):.4f}'],
        ['Best F1 Threshold', f'{thresholds[best_f1_idx]:.2f}'],
        ['Best F1 Score', f'{f1s[best_f1_idx]:.4f}'],
        ['Lift @ 10%', f'{cumulative_captured[int(n_samples*0.1)-1]/10:.2f}x'],
        ['Lift @ 20%', f'{cumulative_captured[int(n_samples*0.2)-1]/20:.2f}x'],
    ]
    
    table = ax.table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Summary Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath2 = os.path.join(output_dir, 'churn_advanced_kpis.png')
    plt.savefig(filepath2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved advanced KPIs plot to {filepath2}")


def main():
    print("="*60)
    print("CHURN PREDICTION WITH MONAD-EMDE FINE-TUNING")
    print("="*60)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    past, portfolio, churn_labels = load_data(WALK)
    
    dataset = ChurnDataset(past, portfolio, churn_labels)
    
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
    
    print(f"\n✓ Creating churn predictor:")
    print(f"  Freeze backbone: {FREEZE_BACKBONE}")
    
    model = ChurnPredictor(
        foundation_model_path=foundation_model_path,
        freeze_backbone=FREEZE_BACKBONE
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    train_losses, val_losses, val_aucs = train_model(
        model, train_loader, val_loader, EPOCHS, PATIENCE, DEVICE
    )
    
    preds, labels, metrics = evaluate_model(model, test_loader, DEVICE)
    
    plot_results(train_losses, val_losses, val_aucs, preds, labels, OUTPUT_DIR)
    
    model_path = os.path.join(OUTPUT_DIR, f'churn_predictor_walk{WALK}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'freeze_backbone': FREEZE_BACKBONE,
        'walk': WALK,
        'metrics': metrics,
        'architecture': 'MonadEMDE_ChurnPredictor'
    }, model_path)
    print(f"✓ Saved model to {model_path}")
    
    pred_path = os.path.join(OUTPUT_DIR, f'churn_predictions_walk{WALK}.npz')
    np.savez_compressed(pred_path, predictions=preds, labels=labels, metrics=metrics)
    print(f"✓ Saved predictions to {pred_path}")
    
    print(f"\n{'='*60}")
    print("CHURN PREDICTION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
