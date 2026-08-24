"""
Quick start training script that loads data on-demand.
No pre-download required - fetches samples as needed.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from tqdm import tqdm
from datasets import load_dataset
import itertools

from model.turn_detector import create_model


def load_on_demand_loader(config, processor, split_name, max_samples):
    """Create a simple dataloader that fetches data on demand."""
    from data.dataset import StreamingTurnDetectionDataset
    
    dataset = StreamingTurnDetectionDataset(
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        split_name=split_name,
        max_samples=max_samples,
        skip_samples=0
    )
    
    return DataLoader(dataset, batch_size=config['training']['batch_size'], num_workers=0)


def quick_train():
    """Quick training with minimal setup."""
    print("=== Quick Start Training ===\n")
    
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Use small sample sizes for quick start
    max_train = 100  # Start with just 100 training samples
    max_val = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training samples: {max_train}")
    print(f"Validation samples: {max_val}\n")
    
    # Load processor and model
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    
    print("Creating model...")
    model = create_model(config).to(device)
    
    # Create simple dataloaders (no download required)
    print("\nCreating dataloaders (streaming mode)...")
    print("Note: First batch may take a moment to fetch...")
    
    from data.dataset import StreamingTurnDetectionDataset
    
    train_dataset = StreamingTurnDetectionDataset(
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        max_samples=max_train
    )
    
    val_dataset = StreamingTurnDetectionDataset(
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        max_samples=max_val,
        skip_samples=max_train
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=0)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n=== Starting Training ===\n")
    
    # Train for 2 epochs as a quick test
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        print("-" * 50)
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            try:
                inputs = batch['input_features'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(inputs).squeeze(-1)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{train_correct/train_total:.4f}"})
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader) if len(list(train_loader)) > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        print(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    inputs = batch['input_features'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(inputs).squeeze(-1)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader) if len(list(val_loader)) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/quick_start_model.pt")
    print("\n✓ Model saved to checkpoints/quick_start_model.pt")
    print("\n=== Quick Training Complete ===")


if __name__ == "__main__":
    quick_train()
