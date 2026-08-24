"""
PyTorch Dataset for turn detection.
"""

import json
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import WhisperProcessor
from typing import Dict, List, Optional
from datasets import load_dataset
import itertools
import requests


class TurnDetectionDataset(Dataset):
    """Dataset for turn detection using Whisper encoder (non-streaming)."""
    
    def __init__(
        self,
        dataset_name: str,
        split_indices: List[int],
        processor: WhisperProcessor,
        cache_dir: str = None,
        max_duration_sec: float = 8.0
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            split_indices: List of indices for this split
            processor: WhisperProcessor for feature extraction
            cache_dir: Cache directory for dataset
            max_duration_sec: Maximum audio duration in seconds
        """
        self.processor = processor
        self.split_indices = split_indices
        self.max_duration_sec = max_duration_sec
        self.sample_rate = processor.feature_extractor.sampling_rate
        
        print(f"Loading dataset: {dataset_name}")
        print("Note: Loading with audio decode=False to bypass FFmpeg/torchcodec")
        
        from datasets import Audio
        
        try:
            # Load dataset WITHOUT automatic audio decoding
            self.dataset = load_dataset(
                dataset_name,
                split='train',
                cache_dir=cache_dir
            )
            # Disable automatic audio decoding - we'll decode manually with soundfile
            self.dataset = self.dataset.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"Trying alternative loading method...")
            self.dataset = load_dataset(dataset_name, cache_dir=cache_dir)
            if hasattr(self.dataset, 'keys'):
                self.dataset = self.dataset[list(self.dataset.keys())[0]]
            self.dataset = self.dataset.cast_column("audio", Audio(decode=False))
    
    def __len__(self) -> int:
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dict with 'input_features' and 'label' tensors
        """
        import soundfile as sf
        import io
        
        dataset_idx = self.split_indices[idx]
        example = self.dataset[dataset_idx]
        
        # Audio is now NOT automatically decoded, so we decode manually
        audio_data = example['audio']
        
        # audio_data is a dict with 'bytes' and 'path' keys
        if 'bytes' in audio_data and audio_data['bytes'] is not None:
            audio_bytes = audio_data['bytes']
        elif 'path' in audio_data and audio_data['path'] is not None:
            # Read from local path
            with open(audio_data['path'], 'rb') as f:
                audio_bytes = f.read()
        else:
            raise ValueError(f"Could not load audio for index {dataset_idx}")
        
        # Manually decode audio using soundfile (no FFmpeg needed!)
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Process through Whisper processor
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.squeeze(0)
        label = float(example['endpoint_bool'])
        
        return {
            'input_features': input_features,
            'label': torch.tensor(label, dtype=torch.float32)
        }


class StreamingTurnDetectionDataset(IterableDataset):
    """Streaming dataset for turn detection using Whisper encoder."""
    
    def __init__(
        self,
        dataset_name: str,
        processor: WhisperProcessor,
        split_name: str = 'train',
        max_samples: Optional[int] = None,
        skip_samples: int = 0,
        max_duration_sec: float = 8.0
    ):
        """
        Initialize streaming dataset.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            processor: WhisperProcessor for feature extraction
            split_name: Split name (train/validation/test)
            max_samples: Maximum number of samples to use (None for all)
            skip_samples: Number of samples to skip at the start
            max_duration_sec: Maximum audio duration in seconds
        """
        self.processor = processor
        self.split_name = split_name
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.max_duration_sec = max_duration_sec
        self.sample_rate = processor.feature_extractor.sampling_rate
        
        print(f"Loading dataset in streaming mode: {dataset_name}")
        try:
            # Load WITHOUT audio decoding to avoid torchcodec requirement
            self.dataset = load_dataset(
                dataset_name,
                split='train',
                streaming=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def _decode_audio_manual(self, audio_dict):
        """Manually decode audio using soundfile."""
        import soundfile as sf
        import io
        
        # Get audio bytes
        if 'bytes' in audio_dict:
            audio_bytes = audio_dict['bytes']
        elif 'path' in audio_dict:
            # Download and read the audio file
            import requests
            response = requests.get(audio_dict['path'])
            audio_bytes = response.content
        else:
            raise ValueError("No audio data found")
        
        # Decode with soundfile
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        
        return audio_array, sampling_rate
    
    def __iter__(self):
        """Iterate over the dataset."""
        dataset_iter = iter(self.dataset)
        
        if self.skip_samples > 0:
            dataset_iter = itertools.islice(dataset_iter, self.skip_samples, None)
        
        if self.max_samples is not None:
            dataset_iter = itertools.islice(dataset_iter, self.max_samples)
        
        for example in dataset_iter:
            try:
                # Manually decode audio to avoid torchcodec
                audio_dict = example['audio']
                
                # Check if already decoded (has 'array' key)
                if isinstance(audio_dict, dict) and 'array' in audio_dict:
                    audio_array = audio_dict['array']
                    sampling_rate = audio_dict['sampling_rate']
                else:
                    # Manually decode
                    audio_array, sampling_rate = self._decode_audio_manual(audio_dict)
                
                inputs = self.processor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                )
                
                input_features = inputs.input_features.squeeze(0)
                label = float(example['endpoint_bool'])
                
                yield {
                    'input_features': input_features,
                    'label': torch.tensor(label, dtype=torch.float32)
                }
            except Exception as e:
                print(f"Error processing example: {e}")
                continue


def load_split_indices(splits_dir: str, split_name: str) -> List[int]:
    """Load split indices from JSON file."""
    split_path = f"{splits_dir}/{split_name}_indices.json"
    with open(split_path, 'r') as f:
        return json.load(f)


def create_dataloaders(config: Dict, processor: WhisperProcessor):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        processor: WhisperProcessor instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    import os
    
    use_streaming = config['data'].get('streaming', False)
    batch_size = config['training']['batch_size']
    
    if use_streaming:
        print("Using streaming mode - no download required!")
        
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        max_train_samples = config['data'].get('max_train_samples', 1000)
        max_val_samples = config['data'].get('max_val_samples', 100)
        max_test_samples = config['data'].get('max_test_samples', 100)
        
        train_dataset = StreamingTurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            processor=processor,
            split_name='train',
            max_samples=max_train_samples,
            skip_samples=0,
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        val_dataset = StreamingTurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            processor=processor,
            split_name='validation',
            max_samples=max_val_samples,
            skip_samples=max_train_samples,
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        test_dataset = StreamingTurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            processor=processor,
            split_name='test',
            max_samples=max_test_samples,
            skip_samples=max_train_samples + max_val_samples,
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0
        )
        
    else:
        print("Using non-streaming mode with downloaded data")
        
        splits_dir = config['data']['splits_dir']
        if not os.path.exists(splits_dir):
            raise ValueError(
                f"Splits directory not found: {splits_dir}. "
                "Please run 'python data/prepare.py' first or enable streaming mode."
            )
        
        train_indices = load_split_indices(splits_dir, 'train')
        val_indices = load_split_indices(splits_dir, 'val')
        test_indices = load_split_indices(splits_dir, 'test')
        
        train_dataset = TurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            split_indices=train_indices,
            processor=processor,
            cache_dir=config['data']['cache_dir'],
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        val_dataset = TurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            split_indices=val_indices,
            processor=processor,
            cache_dir=config['data']['cache_dir'],
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        test_dataset = TurnDetectionDataset(
            dataset_name=config['data']['dataset_name'],
            split_indices=test_indices,
            processor=processor,
            cache_dir=config['data']['cache_dir'],
            max_duration_sec=config['audio']['max_duration_sec']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    return train_loader, val_loader, test_loader
