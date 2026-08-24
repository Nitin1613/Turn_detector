# Dataset Setup Guide for Tiny Turn Detector

## Overview
You want to use the `pipecat-ai/smart-turn-data-v3.2-train` dataset for training. This guide explains your options.

## Current Situation
- A dataset preparation process is currently running (PID: 28508)
- It's been running for ~6 minutes and appears to be downloading data

## Option 1: Download Full Dataset (RECOMMENDED)
**Best for production training with best results**

### Pros:
- Fastest training (no network delays)
- Most reliable
- Full dataset access (~83 audio files)
- Data cached locally for future runs

### Cons:
- Takes 20-30 minutes to download
- Requires ~500MB-1GB storage

### How to use:
```bash
# If the current download process is still running, let it complete
# Or restart it:
cd tiny-turn-detector
python data/prepare.py
```

Once complete, start training:
```bash
python train.py
```

---

## Option 2: Streaming Mode
**Quick start but requires FFmpeg on Windows**

### Pros:
- Start immediately
- No storage space needed upfront

### Cons:
- **Requires FFmpeg installed on Windows** (complex setup)
- Slower training due to network overhead
- Less reliable (network issues can interrupt training)

### Setup Required:
1. Install FFmpeg full-shared version for Windows
2. Install: `pip install torchcodec`
3. Update config.yaml:
   ```yaml
   data:
     streaming: true
   ```

---

## Option 3: Small Subset Download (QUICK START)
**Best for quick testing and development**

### Pros:
- Fast download (5-10 minutes for 200 samples)
- Good for initial testing
- Can scale up later

### Cons:
- Limited data (may affect model performance)
- Need to re-download for full training

### Current Configuration:
Your config is already set for this:
- `use_subset: true`
- `max_download_samples: 200`

### How to use:
The current process should be using this approach. Wait 5-10 more minutes for it to complete.

---

## Recommended Workflow

**For Quick Start (Testing)**:
1. Let the current download complete (should finish soon with 200 samples)
2. Run training: `python train.py`
3. Verify everything works

**For Full Training (Best Results)**:
1. Update `config.yaml`: set `use_subset: false`
2. Run: `python data/prepare.py` (wait 20-30 minutes)
3. Run: `python train.py`

---

## What's Currently Running?

A data preparation script that will:
1. Download the dataset (subset of 200 samples based on current config)
2. Validate audio files
3. Create train/val/test splits
4. Save split indices to `./data/splits/`

---

## Quick Commands Reference

```bash
# Check if preparation completed
ls data/splits/

# Start training (after preparation completes)
python train.py

# Test a single batch
python test_streaming.py

# View training config
cat configs/config.yaml
```

---

## Troubleshooting

**If preparation seems stuck:**
- Check `data/splits/` folder - if files exist, it's done!
- The script may be downloading in background (Windows doesn't show progress)

**If you want to cancel and restart:**
1. Stop the process (Ctrl+C in terminal or Task Manager)
2. Delete `data/cache` and `data/splits` folders
3. Update `config.yaml` as needed
4. Run `python data/prepare.py` again

---

## My Recommendation

Since the current process is already running with the subset configuration (200 samples), I recommend:
1. ✅ **Wait 5-10 more minutes** for it to complete
2. ✅ **Start training** with this smaller dataset to verify everything works
3. ✅ **Scale up** later if needed by downloading the full dataset

This gives you the fastest path to testing your setup!
