import os
import yaml
import torch
import argparse
from pathlib import Path
from transformers import WhisperProcessor
from typing import Dict, Tuple

from model.turn_detector import create_model
from utils.audio import load_audio


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class TurnDetectorInference:
    """Inference wrapper for turn detection model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor: WhisperProcessor,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize inference wrapper.
        
        Args:
            model: Trained turn detection model
            processor: Whisper processor
            device: Device to run inference on
            threshold: Decision threshold for classification
        """
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor
        self.device = device
        self.threshold = threshold
        self.sample_rate = processor.feature_extractor.sampling_rate
    
    def predict_from_file(self, audio_path: str) -> Tuple[float, str]:
        """
        Predict turn detection from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (probability, decision)
            - probability: P(end_turn)
            - decision: "END" or "CONTINUE"
        """
        # Load audio
        audio, sr = load_audio(audio_path, target_sr=self.sample_rate)
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(input_features)
            prob = torch.sigmoid(logits).item()
        
        # Make decision
        decision = "END" if prob >= self.threshold else "CONTINUE"
        
        return prob, decision
    
    def predict_from_array(self, audio: torch.Tensor) -> Tuple[float, str]:
        """
        Predict turn detection from audio array.
        
        Args:
            audio: Audio array (already processed)
            
        Returns:
            Tuple of (probability, decision)
        """
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(input_features)
            prob = torch.sigmoid(logits).item()
        
        # Make decision
        decision = "END" if prob >= self.threshold else "CONTINUE"
        
        return prob, decision
    
    def set_threshold(self, threshold: float):
        """Update decision threshold."""
        self.threshold = threshold


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Run turn detection inference")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: from config)"
    )
    
    args = parser.parse_args()
    
    print("=== Tiny Turn Detector - Inference ===\n")
    
    # Check audio file exists
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config['inference']['device']
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Set threshold
    threshold = args.threshold if args.threshold is not None else config['inference']['threshold']
    
    print(f"Device: {device}")
    print(f"Threshold: {threshold}\n")
    
    # Load processor and model
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(config['model']['whisper_model'])
    model = create_model(config)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config['paths']['checkpoints_dir'],
            'best_model.pt'
        )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_checkpoint(model, checkpoint_path)
    
    # Create inference wrapper
    detector = TurnDetectorInference(
        model=model,
        processor=processor,
        device=device,
        threshold=threshold
    )
    
    # Run inference
    print(f"Processing: {args.audio_path}")
    try:
        prob, decision = detector.predict_from_file(args.audio_path)
        
        print(f"\n=== Results ===")
        print(f"Probability (END): {prob:.4f}")
        print(f"Decision: {decision}")
        
        # Visual indicator
        print(f"\n{'█' * int(prob * 50)}{'░' * int((1-prob) * 50)}")
        print(f"0.0 {'─' * 45} 1.0")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return


if __name__ == "__main__":
    main()
