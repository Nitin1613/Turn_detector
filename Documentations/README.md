# Tiny Turn Detector

A clean, minimal implementation of a real-time audio turn detection model using Whisper encoder and a simple MLP classifier.

## Problem

Detect whether a speaker is actually **DONE speaking** vs **PAUSING/CONTINUING** in conversational audio. This is critical for building responsive voice assistants and conversation systems.

## Architecture

```
Audio (8 sec)
    ↓
VAD (external)
    ↓
Whisper Tiny Encoder (frozen/trainable)
    ↓
Pooling (mean/max/first)
    ↓
MLP Head (embedding_dim → 64 → 1)
    ↓
Sigmoid
    ↓
P(end_turn)
    ↓
Threshold (0.5)
    ↓
Decision: END or CONTINUE
```

## Model Details

- **Encoder**: Whisper Tiny (pretrained)
- **Classifier**: Linear(384 → 64) → ReLU → Linear(64 → 1)
- **Loss**: BCEWithLogitsLoss
- **Optimizer**: AdamW
- **Target**: Binary endpoint_bool (0 = continue, 1 = end)

## Dataset

Uses `pipecat-ai/smart-turn-data-v3.2-train` from Hugging Face Datasets.

## Project Structure

```
tiny-turn-detector/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── configs/
│   └── config.yaml       # Configuration file
├── data/
│   ├── prepare.py        # Dataset preparation script
│   └── dataset.py        # PyTorch Dataset implementation
├── model/
│   └── turn_detector.py  # Model architecture
├── utils/
│   └── audio.py          # Audio utility functions
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
└── benchmark.py          # Benchmarking script
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` to adjust:
- Model parameters (whisper model, hidden dim, pooling strategy)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths
- Inference threshold

## Usage

### 1. Prepare Dataset

Download and prepare the dataset:

```bash
python data/prepare.py
```

This will:
- Download the dataset from Hugging Face
- Inspect the schema
- Create train/validation/test splits
- Save split indices

### 2. Train Model

Train the turn detection model:

```bash
python train.py
```

Training features:
- Config-driven training loop
- Validation every 0.25 epochs
- Early stopping based on validation F1
- Best model checkpoint saved
- Prints loss, accuracy, precision, recall, F1

### 3. Evaluate Model

Evaluate on test set:

```bash
python evaluate.py
```

Outputs:
- Accuracy, precision, recall, F1
- Confusion matrix
- Classification report
- Results saved to `logs/evaluation_results.txt`

### 4. Run Inference

Predict on a single audio file:

```bash
python inference.py path/to/audio.wav
```

Options:
```bash
python inference.py path/to/audio.wav --threshold 0.6 --checkpoint checkpoints/best_model.pt
```

Output:
- Probability of end turn
- Decision (END or CONTINUE)
- Visual probability bar

### 5. Benchmark Performance

Measure model performance:

```bash
python benchmark.py
```

Reports:
- Parameter count
- Model size (MB)
- Inference latency on CPU (mean, std, percentiles)
- Real-time factor

## Configuration

Key configuration options in `configs/config.yaml`:

```yaml
model:
  whisper_model: "openai/whisper-tiny"
  freeze_encoder: false
  hidden_dim: 64
  pooling: "mean"

training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.0001

inference:
  threshold: 0.5
  device: "cpu"
```

## Extension Points

This V1 implementation is intentionally minimal. Future improvements could include:

1. **Acoustic Features**: Add pause duration, speech rate, pitch features
2. **Multi-head Architecture**: Separate heads for filler detection, hesitation
3. **Language Embeddings**: Add language ID as auxiliary input
4. **Advanced Augmentation**: SpecAugment, noise injection
5. **Temporal Context**: Use sliding windows or RNN/Transformer on top
6. **Custom Architecture**: Replace MLP with attention-based head
7. **Multi-task Learning**: Joint training with ASR, speaker ID
8. **Confidence Calibration**: Improve probability estimates

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.14+
- librosa 0.10+
- scikit-learn 1.3+

## Notes

- Model runs on CPU by default
- First run will download Whisper Tiny (~150MB)
- Dataset cache is stored in `data/cache/`
- Checkpoints saved to `checkpoints/`
- Logs saved to `logs/`

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@misc{tiny-turn-detector,
  title={Tiny Turn Detector: Minimal Turn Detection with Whisper},
  author={Your Name},
  year={2024}
}
```
