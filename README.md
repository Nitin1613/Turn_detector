# Tiny Turn Detector

A lightweight real-time audio turn detection model using Whisper encoder and a simple MLP classifier.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](YOUR_HF_MODEL_URL)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problem

Detect whether a speaker is actually **DONE speaking** vs **PAUSING/CONTINUING** in conversational audio. This is critical for building responsive voice assistants and conversation systems.

## 📦 Pre-trained Model

**The trained model is available on Hugging Face:**

🔗 **https://huggingface.co/Nitinbudania/tiny-turn-detector**

You don't need to train from scratch! Download the pre-trained model and start using it immediately.

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

## 📊 Model Performance

The model was trained on [pipecat-ai/smart-turn-data-v3.2-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) dataset.

### Training Results

| Split      | Loss   | Accuracy | Precision | Recall | F1 Score |
|------------|--------|----------|-----------|--------|----------|
| **Train**  | 0.1391 | **95.50%** | 95.90%  | 95.34% | **95.62%** |
| **Val**    | 4.9075 | **73.00%** | 70.00%  | 89.09% | **78.40%** |

**Key Highlights:**
- ✅ High training accuracy (95.5%) demonstrates strong learning
- ✅ Good validation F1 score (78.4%) for real-world applicability
- ✅ High recall (89%) - conservative about marking turn endings (reduces false interruptions)
- ⚡ Fast inference (~100ms on CPU)

## 🏗️ Model Details

- **Encoder**: Whisper Tiny (pretrained from OpenAI)
- **Classifier**: Linear(384 → 64) → ReLU → Linear(64 → 1)
- **Loss**: BCEWithLogitsLoss
- **Optimizer**: AdamW
- **Target**: Binary endpoint_bool (0 = continue, 1 = end)
- **Input**: 8-second audio clips at 16kHz
- **Parameters**: ~39M (Whisper) + ~25K (Classifier)

## 📁 Project Structure

```
tiny-turn-detector/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── Configurations/                # Configuration files
│   └── config.yaml               # Main configuration
│
├── Code_folder/                   # Core implementation
│   ├── data/
│   │   ├── prepare.py            # Dataset preparation
│   │   └── dataset.py            # PyTorch Dataset
│   ├── model/
│   │   └── turn_detector.py     # Model architecture
│   └── utils/
│       └── audio.py              # Audio utilities
│
├── Python_scripts/                # Main scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Inference script
│   ├── benchmark.py              # Performance benchmarking
│   └── quick_train.py            # Quick training for testing
│
├── Notebooks/                     # Jupyter notebooks
│   ├── Run_in_Colab.ipynb        # Google Colab setup
│   └── Run_in_Colab_From_Drive.ipynb
│
└── Documentations/                # Additional docs
    ├── QUICK_START.md
    ├── DATASET_SETUP_GUIDE.md
    └── PROJECT_SUMMARY.md
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Recommended)

Download and use the trained model directly from Hugging Face:

```python
from huggingface_hub import hf_hub_download
import torch

# Download the model
model_path = hf_hub_download(
    repo_id="Nitinbudania/tiny-turn-detector",
    filename="best_model.pt"
)

# Load and use
model = torch.load(model_path, map_location='cpu')
model.eval()

# Run inference (see inference.py for full example)
```

### Option 2: Train Your Own Model

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure

Edit `Configurations/config.yaml` to adjust:
- Model parameters (whisper model, hidden dim, pooling strategy)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and streaming settings
- Inference threshold

#### 3. Prepare Dataset

**For Local Training:**
```bash
python Code_folder/data/prepare.py
```

**For Google Colab:**
Use the provided notebook `Notebooks/Run_in_Colab_From_Drive.ipynb` which handles:
- Mounting Google Drive
- Installing dependencies
- Dataset preparation with space optimization
- Training on GPU

#### 4. Train Model

```bash
python Python_scripts/train.py
```

Training features:
- ✅ Config-driven training loop
- ✅ Validation every 0.25 epochs
- ✅ Early stopping based on validation F1
- ✅ Best model checkpoint saved automatically
- ✅ Real-time metrics (loss, accuracy, precision, recall, F1)

#### 5. Evaluate Model

```bash
python Python_scripts/evaluate.py
```

Outputs:
- Accuracy, precision, recall, F1 scores
- Confusion matrix
- Classification report
- Results saved to `logs/evaluation_results.txt`

#### 6. Run Inference

Predict on a single audio file:

```bash
python Python_scripts/inference.py path/to/audio.wav
```

With custom settings:
```bash
python Python_scripts/inference.py path/to/audio.wav --threshold 0.6 --checkpoint checkpoints/best_model.pt
```

Output includes:
- Probability of end turn
- Binary decision (END or CONTINUE)
- Visual probability bar

#### 7. Benchmark Performance

Measure model speed and efficiency:

```bash
python Python_scripts/benchmark.py
```

Reports:
- Parameter count
- Model size (MB)
- Inference latency (mean, std, percentiles)
- Real-time factor

## ☁️ Google Colab Support

This project includes full Google Colab support for easy experimentation:

1. **Upload** your project ZIP to Google Drive
2. **Open** `Notebooks/Run_in_Colab_From_Drive.ipynb` in Colab
3. **Run** all cells - it handles everything automatically:
   - Mounting Drive
   - Installing dependencies
   - Preparing dataset (with space optimization)
   - Training on free GPU
   - Downloading trained model

Perfect for users without local GPU!

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

## 🎯 Use Cases

- 🎙️ **Voice Assistants**: Determine when to respond
- 📞 **Real-time Conversation Systems**: Natural turn-taking
- 🎧 **Meeting Transcription**: Identify speaker boundaries
- 🤖 **IVR Systems**: Intelligent pause detection
- 💬 **Voice Interfaces**: Responsive user interaction
- 🎮 **Voice-controlled Games**: Command completion detection

## 🔧 Configuration Options

Key settings in `Configurations/config.yaml`:

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

data:
  dataset_name: "pipecat-ai/smart-turn-data-v3.2-train"
  streaming: true           # Use streaming mode
  max_train_samples: 1000   # Limit for faster training
  max_val_samples: 100

inference:
  threshold: 0.5
  device: "cpu"  # or "cuda" for GPU
```

## 🔮 Future Improvements

This V1 implementation is intentionally minimal. Potential enhancements:

1. **Reduce Validation Gap**: Add regularization, data augmentation
2. **Acoustic Features**: Pause duration, speech rate, pitch
3. **Multi-head Architecture**: Separate heads for filler detection, hesitation
4. **Language Support**: Multi-language training and inference
5. **Advanced Augmentation**: SpecAugment, noise injection
6. **Temporal Context**: Sliding windows or RNN/Transformer layers
7. **Variable-length Audio**: Support different input durations
8. **Real-time Streaming**: Process audio in real-time
9. **Multi-task Learning**: Joint training with ASR, speaker ID
10. **Confidence Calibration**: Improve probability estimates

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.14+
- librosa 0.10+
- soundfile 0.12+
- scikit-learn 1.3+

See `requirements.txt` for complete list.

## 💡 Important Notes

- ✅ Model runs on **CPU by default** (no GPU required)
- ✅ First run downloads Whisper Tiny (~150MB) automatically
- ✅ **Pre-trained model available** - no need to train from scratch
- ✅ Supports both **full download** and **streaming mode** for large datasets
- ⚠️ Dataset cache stored in `data/cache/` (can be large)
- ⚠️ Checkpoints saved to `checkpoints/` (excluded from Git)
- ⚠️ Training logs saved to `logs/`

## 🐛 Troubleshooting

**Issue: Out of disk space during dataset preparation**
- Solution: Use streaming mode by setting `streaming: true` in config.yaml

**Issue: FFmpeg/torchcodec errors on Windows**
- Solution: The code uses `soundfile` for audio decoding (no FFmpeg needed)

**Issue: SSL certificate errors when downloading from Hugging Face**
- Solution: See the documentation in `Documentations/` for workarounds

For more help, check `Documentations/DATASET_SETUP_GUIDE.md` and `Documentations/QUICK_START.md`

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

## 🙏 Acknowledgments

- **OpenAI** for the Whisper model
- **Pipecat.ai** for the training dataset
- **Hugging Face** for hosting and tools

## 📚 Additional Resources

- 📦 **[Pre-trained Model on Hugging Face](YOUR_HF_MODEL_URL)**
- 📖 **Documentation**: See `Documentations/` folder
- 📓 **Google Colab Notebooks**: See `Notebooks/` folder
- 🔬 **Dataset**: [pipecat-ai/smart-turn-data-v3.2-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)

## 📖 Citation

If you use this code or model in your research or application, please cite:

```bibtex
@misc{tiny-turn-detector-2026,
  title={Tiny Turn Detector: Real-time Audio Turn Detection with Whisper},
  author= Nitin1613,
  year={2026},
  publisher={GitHub},
  howpublished={https://github.com/Nitin1613/Turn_detector/tree/main}
}
```

---

**⭐ If you find this project helpful, please consider giving it a star!**
