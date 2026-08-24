"""
Evaluation script for turn detection model.
"""

import os
import yaml
import torch
import torch.nn as nn
from transformers import WhisperProcessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
from typing import Dict
from tqdm import tqdm

from model.turn_detector import create_model
from data.dataset import create_dataloaders


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint val F1: {checkpoint.get('val_f1', 'N/A'):.4f}\n")
    return model


def evaluate_model(model: nn.Module, dataloader, device: str) -> Dict:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with predictions, labels, and probabilities
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_features = batch['input_features'].to(device)
            labels = batch['label']
            
            # Get predictions
            logits = model(input_features).squeeze(-1)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
    
    return {
        'predictions': torch.cat(all_predictions).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'probabilities': torch.cat(all_probs).numpy()
    }


def print_metrics(results: Dict, split_name: str = "Test"):
    """Print evaluation metrics."""
    preds = results['predictions']
    labels = results['labels']
    
    # Compute metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    print(f"\n=== {split_name} Set Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Continue  End")
    print(f"Actual Continue  {cm[0][0]:6d}  {cm[0][1]:4d}")
    print(f"       End       {cm[1][0]:6d}  {cm[1][1]:4d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=['Continue', 'End'],
        zero_division=0
    ))
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def evaluate_by_language(results: Dict, dataloader):
    """
    Evaluate metrics grouped by language if metadata exists.
    
    Note: This requires the dataset to have language information.
    For simplicity, we skip this if not available.
    """
    print("\n=== Language-specific Metrics ===")
    print("(Requires language metadata in dataset)")
    
    # This would require modifying the dataset to include language info
    # For now, we'll just note it as a placeholder
    print("Not implemented - requires dataset modification to include language")


def main():
    """Main evaluation pipeline."""
    print("=== Tiny Turn Detector - Evaluation ===\n")
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load processor
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, processor)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        config['paths']['checkpoints_dir'],
        'best_model.pt'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, device)
    test_metrics = print_metrics(test_results, "Test")
    
    # Optionally evaluate on validation set
    print("\n" + "="*50)
    print("\nEvaluating on validation set...")
    val_results = evaluate_model(model, val_loader, device)
    val_metrics = print_metrics(val_results, "Validation")
    
    # Language-specific evaluation (placeholder)
    evaluate_by_language(test_results, test_loader)
    
    print("\n=== Evaluation Complete ===")
    
    # Save results
    results_path = os.path.join(config['paths']['logs_dir'], 'evaluation_results.txt')
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("=== Test Set Metrics ===\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n=== Validation Set Metrics ===\n")
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
