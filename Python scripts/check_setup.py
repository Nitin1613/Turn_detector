"""
Simple script to verify all imports work correctly.
Run this before starting training to catch any import issues.
"""

import sys
print("Python version:", sys.version)
print("\nChecking imports...\n")

# Check standard library
print("✓ Standard library imports")
import os, yaml, json, time, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Check core dependencies
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torchaudio
    print(f"✓ torchaudio")
except ImportError as e:
    print(f"✗ torchaudio import failed: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")

try:
    import datasets
    print(f"✓ datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ datasets import failed: {e}")

try:
    import librosa
    print(f"✓ librosa {librosa.__version__}")
except ImportError as e:
    print(f"✗ librosa import failed: {e}")

try:
    import soundfile
    print(f"✓ soundfile")
except ImportError as e:
    print(f"✗ soundfile import failed: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import tqdm
    print(f"✓ tqdm")
except ImportError as e:
    print(f"✗ tqdm import failed: {e}")

# Check project modules
print("\nChecking project modules...")

try:
    from model.turn_detector import TurnDetector, create_model
    print("✓ model.turn_detector")
except ImportError as e:
    print(f"✗ model.turn_detector import failed: {e}")

try:
    from data.dataset import TurnDetectionDataset, create_dataloaders
    print("✓ data.dataset")
except ImportError as e:
    print(f"✗ data.dataset import failed: {e}")

try:
    from utils.audio import load_audio, resample_audio, normalize_audio
    print("✓ utils.audio")
except ImportError as e:
    print(f"✗ utils.audio import failed: {e}")

# Check configuration loading
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration file loaded")
    print(f"  Model: {config['model']['whisper_model']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")

# Check CUDA availability
print("\nDevice information:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Will use CPU for training and inference")

print("\n" + "="*50)
print("Environment check complete!")
print("="*50)
print("\nIf all checks passed, you're ready to:")
print("1. python data/prepare.py")
print("2. python train.py")
print("3. python evaluate.py")
print("4. python inference.py <audio_file>")
