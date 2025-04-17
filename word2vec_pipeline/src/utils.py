"""
Utility functions for Word2Vec training and model management.

This module provides helper functions for:
- Setting random seeds for reproducibility
- Handling device selection (CPU/GPU)
- Checkpoint management (saving/loading)
- Timing utilities for performance monitoring
- Model export utilities
"""

import os
import json
import numpy as np
import torch
import random
import time
from typing import Tuple, Dict, Any


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all random generators.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the appropriate device (CPU/GPU) for tensor operations.

    Returns:
        torch.device: Device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    path: str) -> None:
    """
    Save model checkpoint with all information needed to resume training.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint and restore training state.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        path: Path to checkpoint

    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=get_device())

    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


def save_embeddings(model: torch.nn.Module, path: str) -> None:
    """
    Save model embeddings to file.

    Args:
        model: PyTorch model with get_in_embeddings method
        path: Path to save embeddings
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get embeddings
    embeddings = model.get_in_embeddings()

    # Save embeddings
    np.save(path, embeddings)


def export_model(model: torch.nn.Module,
                 vocab: Any,
                 config: Dict[str, Any],
                 output_dir: str,
                 version: str = "0.1.0") -> Dict[str, str]:
    """
    Export model to production-ready format with metadata.

    Creates a directory with:
    - model.pt: TorchScript model for inference
    - embeddings.npy: NumPy array of word embeddings
    - vocab.json: Vocabulary mapping in JSON format
    - metadata.json: Training parameters and configuration

    Args:
        model: Trained PyTorch model
        vocab: Vocabulary object
        config: Training configuration
        output_dir: Directory to save exports
        version: Model version string

    Returns:
        Dict[str, str]: Dictionary with paths to exported files
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    model_path = os.path.join(output_dir, "model.pt")
    embed_path = os.path.join(output_dir, "embeddings.npy")
    vocab_path = os.path.join(output_dir, "vocab.json")
    meta_path = os.path.join(output_dir, "metadata.json")

    # 1. Save TorchScript model
    model.eval()
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, model_path)

    # 2. Save embeddings
    save_embeddings(model, embed_path)

    # 3. Save vocabulary mapping (simplified format for inference)
    vocab_export = {
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_export, f)

    # 4. Save metadata
    metadata = {
        "version": version,
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": len(vocab),
        "embed_dim": config["embed_dim"],
        "window_size": config["window_size"],
        "min_count": config.get("min_count", 5),
        "neg_samples": config["neg_samples"],
        "model_type": "SkipGram"
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    return {
        "model": model_path,
        "embeddings": embed_path,
        "vocabulary": vocab_path,
        "metadata": meta_path
    }


class Timer:
    """
    Simple timer class for tracking execution time of operations.

    Provides methods to start, stop and report elapsed time in seconds.
    """

    def __init__(self):
        """Initialize timer with null values."""
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Start timer."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timer."""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            float: Elapsed time since start was called
        """
        if self.start_time is None:
            return 0

        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
