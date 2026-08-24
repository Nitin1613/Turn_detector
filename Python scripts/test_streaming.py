"""
Quick test to verify streaming mode works.
"""

import yaml
from transformers import WhisperProcessor
from data.dataset import create_dataloaders


def test_streaming():
    """Test that streaming mode loads data correctly."""
    print("=== Testing Streaming Mode ===\n")
    
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Streaming enabled: {config['data']['streaming']}")
    print(f"Max train samples: {config['data']['max_train_samples']}")
    print(f"Max val samples: {config['data']['max_val_samples']}")
    print(f"Batch size: {config['training']['batch_size']}\n")
    
    # Load processor
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, processor)
    
    print("\n=== Testing First Batch ===")
    
    # Test train loader
    print("\nFetching first training batch...")
    train_batch = next(iter(train_loader))
    print(f"✓ Train batch shape: {train_batch['input_features'].shape}")
    print(f"✓ Labels shape: {train_batch['label'].shape}")
    print(f"✓ Sample label values: {train_batch['label'][:5]}")
    
    # Test val loader
    print("\nFetching first validation batch...")
    val_batch = next(iter(val_loader))
    print(f"✓ Val batch shape: {val_batch['input_features'].shape}")
    print(f"✓ Labels shape: {val_batch['label'].shape}")
    
    print("\n=== Streaming Mode Test Passed! ===")
    print("You can now run training with: python train.py")


if __name__ == "__main__":
    test_streaming()
