"""
Training script for turn detection model. freeze_encoder
"""

import os
from networkx import config
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from tqdm import tqdm
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.turn_detector import create_model
from data.dataset import create_dataloaders


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels (0 or 1)
        labels: True labels (0 or 1)
        
    Returns:
        Dictionary of metrics
    """
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(labels_np, preds_np),
        'precision': precision_score(labels_np, preds_np, zero_division=0),
        'recall': recall_score(labels_np, preds_np, zero_division=0),
        'f1': f1_score(labels_np, preds_np, zero_division=0)
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_features = batch['input_features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits = model(input_features).squeeze(-1)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        all_predictions.append(predictions)
        all_labels.append(labels)
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_features = batch['input_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_features).squeeze(-1)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    checkpoint_path: str
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training loop."""
    print("=== Tiny Turn Detector - Training ===\n")
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create processor
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, processor)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Create optimizer and loss
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    num_epochs = config['training']['num_epochs']
    
    print("\n=== Starting Training ===\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            config['training']['gradient_clip']
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Metrics: "
              f"Acc={train_metrics['accuracy']:.4f}, "
              f"P={train_metrics['precision']:.4f}, "
              f"R={train_metrics['recall']:.4f}, "
              f"F1={train_metrics['f1']:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: "
              f"Acc={val_metrics['accuracy']:.4f}, "
              f"P={val_metrics['precision']:.4f}, "
              f"R={val_metrics['recall']:.4f}, "
              f"F1={val_metrics['f1']:.4f}")
        
        # Save best model based on F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            checkpoint_path = os.path.join(
                config['paths']['checkpoints_dir'],
                'best_model.pt'
            )
            save_checkpoint(model, optimizer, epoch, best_val_f1, checkpoint_path)
            patience_counter = 0
            print(f"[BEST] New best model! F1: {best_val_f1:.4f}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}\n")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break
    
    print("\n=== Training Complete ===")
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
