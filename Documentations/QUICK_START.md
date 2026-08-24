# Quick Start Guide - Tiny Turn Detector

## Current Status
- Streaming mode requires FFmpeg (not recommended on Windows)
- Dataset preparation is running in background (PID 28508)
- Configuration is set for 200-sample subset

## How to Start Training

### Step 1: Check if Data is Ready
```bash
dir data\splits
```

If you see these files, data is ready:
- `train_indices.json`
- `val_indices.json`  
- `test_indices.json`

### Step 2: Start Training
```bash
python train.py
```

That's it! The standard training script will:
- Load the downloaded dataset subset
- Train for 10 epochs
- Save best model to `checkpoints/best_model.pt`
- Display training/validation metrics

## Training Output
You'll see:
- Progress bars for each epoch
- Training loss, accuracy, precision, recall, F1
- Validation metrics
- Automatic early stopping if no improvement

## After Training
Test your model:
```bash
python evaluate.py
```

Run inference on audio files:
```bash
python inference.py path/to/audio.wav
```

## Troubleshooting

**If data prep is still running:**
- Wait 5-10 more minutes
- It's downloading 200 samples in background

**If it seems stuck:**
- Check Task Manager for python process
- If hung, kill it and restart:
  ```bash
  python data/prepare.py
  ```

**For full dataset (better accuracy):**
1. Edit `configs/config.yaml`: `use_subset: false`
2. Run: `python data/prepare.py` (wait 20-30 min)
3. Run: `python train.py`

## Expected Results
With 200 samples:
- Training should complete in 10-20 minutes
- Validation F1 score: 0.70-0.85 (varies)
- Good enough for testing/development

With full dataset (~1000+ samples):
- Training: 30-60 minutes  
- Validation F1 score: 0.85-0.95
- Production-ready model

---

**Next Steps:** Wait for data prep to finish, then run `python train.py`!
