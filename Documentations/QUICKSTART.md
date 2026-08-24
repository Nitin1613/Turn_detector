# Quick Start Guide - Tiny Turn Detector

## Step-by-Step Instructions

### 1. Navigate to Project Directory
```bash
cd tiny-turn-detector
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchaudio
- Transformers (for Whisper)
- Datasets (for HuggingFace dataset loading)
- librosa and soundfile (for audio processing)
- scikit-learn (for metrics)
- PyYAML (for config)

### 4. Prepare Dataset
```bash
python data/prepare.py
```

**What happens:**
- Downloads `pipecat-ai/smart-turn-data-v3.2-train` from HuggingFace
- Inspects dataset schema and prints statistics
- Validates audio samples
- Creates train/val/test splits (80/10/10)
- Saves split indices to `data/splits/`

**Expected output:**
```
=== Tiny Turn Detector - Dataset Preparation ===

Loading dataset: pipecat-ai/smart-turn-data-v3.2-train
...
Valid examples: XXXX / XXXX

Split sizes:
  Train: XXXX
  Val: XXXX
  Test: XXXX
```

### 5. Train Model
```bash
python train.py
```

**What happens:**
- Loads Whisper Tiny encoder
- Creates MLP classification head
- Trains with BCEWithLogitsLoss and AdamW
- Validates every 0.25 epochs
- Saves best checkpoint based on validation F1
- Early stopping with patience of 3 epochs

**Expected output:**
```
=== Tiny Turn Detector - Training ===

Epoch 1/10
--------------------------------------------------
Training...
Train Loss: 0.XXXX
Train Metrics: Acc=0.XX, P=0.XX, R=0.XX, F1=0.XX
Validating...
Val Loss: 0.XXXX
Val Metrics: Acc=0.XX, P=0.XX, R=0.XX, F1=0.XX
✓ New best model! F1: 0.XXXX
```

**Checkpoint saved to:** `checkpoints/best_model.pt`

### 6. Evaluate Model
```bash
python evaluate.py
```

**What happens:**
- Loads best checkpoint
- Evaluates on test set
- Prints metrics and confusion matrix
- Saves results to `logs/evaluation_results.txt`

**Expected output:**
```
=== Test Set Metrics ===
Accuracy:  0.XXXX
Precision: 0.XXXX
Recall:    0.XXXX
F1 Score:  0.XXXX

Confusion Matrix:
                Predicted
              Continue  End
Actual Continue   XXX    XX
       End        XX     XXX
```

### 7. Run Inference
```bash
python inference.py path/to/your/audio.wav
```

**Example:**
```bash
python inference.py test_audio.wav
```

**With custom threshold:**
```bash
python inference.py test_audio.wav --threshold 0.6
```

**Expected output:**
```
=== Tiny Turn Detector - Inference ===

Device: cpu
Threshold: 0.5

Loading model...
Processing: test_audio.wav

=== Results ===
Probability (END): 0.7532
Decision: END

██████████████████████████████████████░░░░░░░░░░
0.0 ───────────────────────────────────────── 1.0
```

### 8. Benchmark Performance
```bash
python benchmark.py
```

**What happens:**
- Measures inference latency on CPU
- Reports parameter count
- Calculates model size
- Shows real-time factor

**Expected output:**
```
=== Benchmark Results ===

--- Model Parameters ---
Total parameters:          XX,XXX,XXX
Trainable parameters:      XX,XXX,XXX
Encoder parameters:        XX,XXX,XXX
Head parameters:           XX,XXX

--- Model Size ---
Total size:       XXX.XX MB

--- Inference Latency (CPU) ---
Mean:             XX.XX ms
Median:           XX.XX ms
95th percentile:  XX.XX ms
```

## Configuration Customization

Edit `configs/config.yaml` to customize:

### Model Architecture
```yaml
model:
  whisper_model: "openai/whisper-tiny"  # or "openai/whisper-base"
  freeze_encoder: false                  # Set to true to freeze encoder
  hidden_dim: 64                         # MLP hidden dimension
  pooling: "mean"                        # Options: mean, max, first
```

### Training Hyperparameters
```yaml
training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping_patience: 3
```

### Inference Settings
```yaml
inference:
  threshold: 0.5    # Adjust to balance precision/recall
  device: "cpu"     # or "cuda" if GPU available
```

## Troubleshooting

### Issue: Dataset download fails
**Solution:** Check internet connection and HuggingFace access. Try:
```bash
pip install --upgrade datasets
```

### Issue: Out of memory during training
**Solution:** Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 16  # or 8
```

### Issue: Model too slow
**Solution:** 
1. Try freezing encoder:
```yaml
model:
  freeze_encoder: true
```
2. Use GPU if available:
```yaml
inference:
  device: "cuda"
```

### Issue: Low accuracy
**Solution:** 
1. Train longer (more epochs)
2. Don't freeze encoder
3. Increase model capacity:
```yaml
model:
  hidden_dim: 128  # or 256
```

## Next Steps

### Experiment with different configurations
1. Try different pooling strategies (mean, max, first)
2. Adjust decision threshold for your use case
3. Freeze/unfreeze encoder to trade speed vs accuracy
4. Try larger Whisper models (base, small, medium)

### Integrate into your application
```python
from inference import TurnDetectorInference
from model.turn_detector import create_model
from transformers import WhisperProcessor

# Load model
config = load_config()
processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
model = create_model(config)
model = load_checkpoint(model, 'checkpoints/best_model.pt')

# Create detector
detector = TurnDetectorInference(model, processor, device='cpu', threshold=0.5)

# Predict
prob, decision = detector.predict_from_file('audio.wav')
print(f"Turn end probability: {prob:.4f}")
print(f"Decision: {decision}")
```

## File Structure Summary

```
tiny-turn-detector/
├── configs/config.yaml          # All configuration parameters
├── data/
│   ├── prepare.py              # Dataset download & splitting
│   ├── dataset.py              # PyTorch Dataset class
│   ├── cache/                  # Dataset cache (auto-created)
│   └── splits/                 # Train/val/test indices (auto-created)
├── model/
│   └── turn_detector.py        # Model architecture
├── utils/
│   └── audio.py                # Audio utilities
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── benchmark.py                 # Benchmarking script
├── checkpoints/                 # Model checkpoints (auto-created)
├── logs/                        # Logs and results (auto-created)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Full documentation
```

## Support

For issues or questions:
1. Check configuration in `configs/config.yaml`
2. Review error messages carefully
3. Verify dataset preparation completed successfully
4. Check that checkpoint exists before evaluation/inference

Happy training! 🚀
