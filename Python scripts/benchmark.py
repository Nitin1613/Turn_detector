"""
Benchmark script for turn detection model.
Measures inference latency and model size.
"""

import os
import yaml
import torch
import time
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor
from typing import Dict

from model.turn_detector import create_model


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in MB.
    
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_mb': size_mb,
        'params_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024
    }


def benchmark_inference_latency(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark inference latency.
    
    Args:
        model: Model to benchmark
        processor: Whisper processor
        device: Device to run on
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with latency statistics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input (8 seconds of audio)
    sample_rate = processor.feature_extractor.sampling_rate
    audio_duration = 8.0
    audio = np.random.randn(int(sample_rate * audio_duration)).astype(np.float32)
    
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    
    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_features)
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_features)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }


def print_benchmark_results(
    param_counts: Dict[str, int],
    model_size: Dict[str, float],
    latency_stats: Dict[str, float]
):
    """Print formatted benchmark results."""
    print("\n" + "="*60)
    print("=== Benchmark Results ===")
    print("="*60)
    
    print("\n--- Model Parameters ---")
    print(f"Total parameters:     {param_counts['total']:>15,}")
    print(f"Trainable parameters: {param_counts['trainable']:>15,}")
    print(f"Encoder parameters:   {param_counts['encoder']:>15,}")
    print(f"Head parameters:      {param_counts['head']:>15,}")
    
    print("\n--- Model Size ---")
    print(f"Total size:   {model_size['total_mb']:>10.2f} MB")
    print(f"Parameters:   {model_size['params_mb']:>10.2f} MB")
    print(f"Buffers:      {model_size['buffers_mb']:>10.2f} MB")
    
    print("\n--- Inference Latency (CPU) ---")
    print(f"Mean:         {latency_stats['mean_ms']:>10.2f} ms")
    print(f"Std:          {latency_stats['std_ms']:>10.2f} ms")
    print(f"Median:       {latency_stats['median_ms']:>10.2f} ms")
    print(f"Min:          {latency_stats['min_ms']:>10.2f} ms")
    print(f"Max:          {latency_stats['max_ms']:>10.2f} ms")
    print(f"95th percentile: {latency_stats['p95_ms']:>10.2f} ms")
    print(f"99th percentile: {latency_stats['p99_ms']:>10.2f} ms")
    
    print("\n--- Real-time Factor ---")
    audio_duration_ms = 8000  # 8 seconds
    rtf = latency_stats['mean_ms'] / audio_duration_ms
    print(f"RTF (mean):   {rtf:>10.4f}x")
    print(f"  (latency / audio_duration)")
    
    if rtf < 1.0:
        print(f"  ✓ Faster than real-time!")
    else:
        print(f"  ✗ Slower than real-time")
    
    print("\n" + "="*60)


def main():
    """Main benchmark pipeline."""
    print("=== Tiny Turn Detector - Benchmark ===\n")
    
    # Load configuration
    config = load_config()
    
    # Force CPU for benchmarking
    device = "cpu"
    print(f"Benchmarking on: {device}\n")
    
    # Load processor and model
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    model = create_model(config)
    
    # Optionally load checkpoint
    checkpoint_path = os.path.join(
        config['paths']['checkpoints_dir'],
        'best_model.pt'
    )
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = load_checkpoint(model, checkpoint_path)
    else:
        print("No checkpoint found, benchmarking untrained model")
    
    # Get parameter counts
    param_counts = model.count_parameters()
    
    # Get model size
    model_size = get_model_size(model)
    
    # Benchmark latency
    latency_stats = benchmark_inference_latency(
        model=model,
        processor=processor,
        device=device,
        num_runs=100,
        warmup_runs=10
    )
    
    # Print results
    print_benchmark_results(param_counts, model_size, latency_stats)
    
    # Save results
    results_path = os.path.join(config['paths']['logs_dir'], 'benchmark_results.txt')
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("=== Benchmark Results ===\n\n")
        
        f.write("--- Parameters ---\n")
        for k, v in param_counts.items():
            f.write(f"{k}: {v:,}\n")
        
        f.write("\n--- Model Size ---\n")
        for k, v in model_size.items():
            f.write(f"{k}: {v:.2f}\n")
        
        f.write("\n--- Latency ---\n")
        for k, v in latency_stats.items():
            f.write(f"{k}: {v:.2f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
