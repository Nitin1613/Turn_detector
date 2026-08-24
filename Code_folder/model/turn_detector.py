"""
Turn detection model using Whisper encoder.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import Dict


class TurnDetector(nn.Module):
    """
    Turn detection model using Whisper encoder + MLP head.
    
    Architecture:
        Audio → Whisper Encoder → Pooling → MLP → P(end_turn)
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        hidden_dim: int = 64,
        freeze_encoder: bool = False,
        pooling: str = "mean"
    ):
        """
        Initialize turn detector.
        
        Args:
            whisper_model_name: Pretrained Whisper model name
            hidden_dim: Hidden dimension for MLP
            freeze_encoder: Whether to freeze Whisper encoder weights
            pooling: Pooling strategy ('mean', 'max', or 'first')
        """
        super().__init__()
        
        # Load Whisper model (encoder only)
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = self.whisper.encoder
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Whisper encoder frozen")
        
        # Get encoder output dimension
        self.encoder_dim = self.whisper.config.d_model
        
        # Pooling strategy
        self.pooling = pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"TurnDetector initialized:")
        print(f"  Encoder: {whisper_model_name}")
        print(f"  Encoder dim: {self.encoder_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Pooling: {pooling}")
        print(f"  Frozen: {freeze_encoder}")
    
    def pool_encoder_output(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Pool encoder output to fixed-size representation.
        
        Args:
            encoder_output: Tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Pooled tensor of shape (batch_size, hidden_dim)
        """
        if self.pooling == "mean":
            return encoder_output.mean(dim=1)
        elif self.pooling == "max":
            return encoder_output.max(dim=1)[0]
        elif self.pooling == "first":
            return encoder_output[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_features: Log-mel spectrogram from WhisperProcessor
                           Shape: (batch_size, n_mels, n_frames)
        
        Returns:
            Logits for binary classification, shape: (batch_size, 1)
        """
        # Pass through encoder
        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        # Shape: (batch_size, seq_len, hidden_dim)
        
        # Pool encoder outputs
        pooled = self.pool_encoder_output(encoder_hidden_states)
        # Shape: (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)
        # Shape: (batch_size, 1)
        
        return logits
    
    def predict_proba(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Get probability of end turn.
        
        Args:
            input_features: Log-mel spectrogram
            
        Returns:
            Probability tensor, shape: (batch_size, 1)
        """
        logits = self.forward(input_features)
        probs = torch.sigmoid(logits)
        return probs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'head': head_params
        }


def create_model(config: Dict) -> TurnDetector:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TurnDetector instance
    """
    model = TurnDetector(
        whisper_model_name=config['model']['whisper_model'],
        hidden_dim=config['model']['hidden_dim'],
        freeze_encoder=config['model']['freeze_encoder'],
        pooling=config['model']['pooling']
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    return model
