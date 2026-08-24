"""
Audio utility functions.
"""

import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate in Hz
        mono: Convert to mono if True
        duration: Maximum duration in seconds (None for full file)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Load with librosa (handles most formats)
        audio, sr = librosa.load(
            file_path,
            sr=target_sr,
            mono=mono,
            duration=duration
        )
        return audio, target_sr
    
    except Exception as e:
        print(f"librosa failed, trying soundfile: {e}")
        try:
            # Fallback to soundfile
            audio, sr = sf.read(file_path)
            
            # Convert to mono if needed
            if mono and len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=target_sr
                )
            
            # Trim to duration if specified
            if duration is not None:
                max_samples = int(duration * target_sr)
                audio = audio[:max_samples]
            
            return audio, target_sr
        
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio from {file_path}: {e2}")


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo to mono by averaging channels.
    
    Args:
        audio: Audio array of shape (samples,) or (channels, samples)
        
    Returns:
        Mono audio array of shape (samples,)
    """
    if len(audio.shape) > 1:
        return audio.mean(axis=0)
    return audio


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(
        audio,
        orig_sr=orig_sr,
        target_sr=target_sr
    )


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dBFS level.
    
    Args:
        audio: Audio array
        target_level: Target level in dBFS
        
    Returns:
        Normalized audio array
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate current level in dB
    current_db = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_level - current_db
    gain = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized = audio * gain
    
    # Clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = 20,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        top_db: Threshold in dB below reference
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def get_audio_duration(file_path: str) -> float:
    """
    Get audio file duration in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        raise RuntimeError(f"Failed to get duration for {file_path}: {e}")


def validate_audio_file(file_path: str) -> bool:
    """
    Check if audio file is valid and readable.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not Path(file_path).exists():
            return False
        
        # Try to load just the first second
        audio, sr = load_audio(file_path, duration=1.0)
        
        # Check that we got some data
        if len(audio) == 0:
            return False
        
        return True
    
    except Exception:
        return False
