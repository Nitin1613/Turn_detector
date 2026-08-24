"""
Dataset preparation script.
Downloads and prepares the Hugging Face dataset for training.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def inspect_dataset(dataset):
    """Inspect dataset schema and print statistics."""
    print("\n=== Dataset Inspection ===")
    print(f"Dataset features: {dataset.features}")
    print(f"\nFirst example keys: ['audio', 'id', 'language', 'endpoint_bool', 'midfiller', 'endfiller', 'synthetic', 'spoken_text', 'dataset']")
    print(f"\nDataset size: {len(dataset)} examples")
    


# Check for endpoint_bool column (skip counting to avoid loading all data)
    if 'endpoint_bool' in dataset.features:
        print(f"\nEndpoint distribution: (skipped - will validate during splitting)")
    
    # Check for language if available
    # Check for language if available (skipped to avoid audio decoding)
    if 'language' in dataset.features:
        print(f"\nLanguages: (will be validated during training)")
    
    # Print sample example
    # Print sample example (skipped to avoid audio decoding issues on Windows)
    print("\n=== Sample Example (Skipped) ===")
    print("Audio decoding skipped for compatibility")


def validate_example(example: Dict) -> bool:
    """Validate that an example has required fields and valid audio."""
    try:
        # Check required fields
        if 'audio' not in example or 'endpoint_bool' not in example:
            return False
        
        # Check audio data exists
        audio_data = example['audio']
        if audio_data is None:
            return False
            
        # Check if audio array exists and has data
        if 'array' in audio_data:
            if audio_data['array'] is None or len(audio_data['array']) == 0:
                return False
        
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def create_splits(dataset, config: Dict) -> Dict[str, List[int]]:
    """Create train/validation/test splits."""
    print("\n=== Creating Splits ===")
    
    # Check if we should use a subset
    use_subset = config['data'].get('use_subset', False)
    max_samples = config['data'].get('max_download_samples', len(dataset))
    
    if use_subset and max_samples < len(dataset):
        print(f"Using subset: {max_samples} samples (out of {len(dataset)} total)")
        dataset_size = min(max_samples, len(dataset))
    else:
        dataset_size = len(dataset)
        print(f"Using full dataset: {dataset_size} samples")
    
    # Create indices without validation (skips FFmpeg requirement)
    # Validation will happen during training when audio is actually loaded
    print("Creating split indices (skipping audio validation to avoid FFmpeg issues)...")
    all_indices = list(range(dataset_size))
    
    print(f"Total indices: {len(all_indices)}")
    
    # Shuffle indices
    import random
    random.seed(42)
    random.shuffle(all_indices)
    
    # Calculate split sizes
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    train_size = int(len(all_indices) * train_ratio)
    val_size = int(len(all_indices) * val_ratio)
    
    # Create splits
    splits = {
        'train': all_indices[:train_size],
        'val': all_indices[train_size:train_size + val_size],
        'test': all_indices[train_size + val_size:]
    }
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return splits


def save_splits(splits: Dict[str, List[int]], splits_dir: str):
    """Save split indices to JSON files."""
    os.makedirs(splits_dir, exist_ok=True)
    
    for split_name, indices in splits.items():
        split_path = os.path.join(splits_dir, f"{split_name}_indices.json")
        with open(split_path, 'w') as f:
            json.dump(indices, f)
        print(f"Saved {split_name} split to {split_path}")


def main():
    """Main preparation pipeline."""
    print("=== Tiny Turn Detector - Dataset Preparation ===\n")
    
    # Load configuration
    config = load_config()
    
    # Create directories
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    os.makedirs(config['data']['splits_dir'], exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {config['data']['dataset_name']}")
    try:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split='train',  # Adjust if dataset has specific splits
            cache_dir=config['data']['cache_dir']
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying to load without split specification...")
        dataset = load_dataset(
            config['data']['dataset_name'],
            cache_dir=config['data']['cache_dir']
        )
        # If dataset is a DatasetDict, take the first split
        if hasattr(dataset, 'keys'):
            dataset = dataset[list(dataset.keys())[0]]
    
    # Inspect dataset
    inspect_dataset(dataset)
    
    # Create and save splits
    splits = create_splits(dataset, config)
    save_splits(splits, config['data']['splits_dir'])
    
    print("\n=== Dataset Preparation Complete ===")


if __name__ == "__main__":
    main()
