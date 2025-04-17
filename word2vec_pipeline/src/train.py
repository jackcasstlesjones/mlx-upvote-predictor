"""
Training functionality for Word2Vec models.

This module provides the core training loop and utilities for training
Skip-gram Word2Vec models with negative sampling, including:
- Training loop implementation
- Epoch-level training logic
- Checkpoint management
- Progress tracking
"""

import os
import logging
from typing import Callable, Dict, Any, Generator, List, Optional, Tuple

import torch
import torch.optim as optim

from .dataset import SkipgramDataset, create_dataloader
from .model import SkipGramModel
from .utils import (
    Timer,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_embeddings,
    set_seed,
    export_model
)

# Configure logging
logger = logging.getLogger(__name__)


def train_epoch(model: SkipGramModel,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """
    Train model for one epoch.

    Processes all batches in the dataloader, computing loss and updating
    model parameters.

    Args:
        model: SkipGramModel instance
        dataloader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to use for tensor operations

    Returns:
        float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    total_batches = 0

    # Create timer for progress tracking
    timer = Timer()
    timer.start()

    for i, (centers, contexts, negatives) in enumerate(dataloader):
        # Move data to device
        centers = centers.to(device)
        contexts = contexts.to(device)
        negatives = negatives.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(centers, contexts, negatives)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Track progress
        total_loss += loss.item()
        total_batches += 1

        if (i + 1) % 100 == 0:
            elapsed = timer.elapsed()
            logger.info(f"Batch {i+1}, Loss: {loss.item():.4f}, "
                        f"Speed: {100 / elapsed:.1f} batches/s")
            timer.start()

    # Return average loss
    return total_loss / total_batches


def train_model(
    token_stream_fn: Callable[[], Generator[List[str], None, None]],
    vocab: Any,
    embed_dim: int = 100,
    window_size: int = 5,
    neg_samples: int = 10,
    learning_rate: float = 0.002,
    batch_size: int = 512,
    epochs: int = 5,
    checkpoint_dir: str = "checkpoints",
    embeddings_dir: str = "embeddings",
    export_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    seed: int = 42
) -> Tuple[SkipGramModel, Dict[str, Any]]:
    """
    Train SkipGram Word2Vec model.

    Implements full training process including:
    - Initialization and setup
    - Epoch-level training
    - Checkpointing
    - Embedding saving
    - Optional export for production

    Args:
        token_stream_fn: Function that returns token stream generator
        vocab: Vocabulary instance
        embed_dim: Embedding dimension
        window_size: Context window size
        neg_samples: Number of negative samples per context word
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
        embeddings_dir: Directory to save embeddings
        export_dir: Directory for final model export (optional)
        resume_from: Path to checkpoint to resume from
        seed: Random seed

    Returns:
        Tuple[SkipGramModel, Dict[str, Any]]: Trained model and training info
    """
    # Set seeds for reproducibility
    set_seed(seed)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize model
    vocab_size = len(vocab)
    model = SkipGramModel(vocab_size, embed_dim)
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, resume_from)

    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Track training progress and metrics
    training_info = {
        "epochs_completed": 0,
        "best_loss": float('inf'),
        "best_model_path": None,
        "final_loss": None,
        "embed_dim": embed_dim,
        "window_size": window_size,
        "neg_samples": neg_samples,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }

    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        # Create a new dataset and dataloader for each epoch to ensure
        # we get fresh data and different negative samples
        dataset = SkipgramDataset(
            token_stream_fn,
            vocab,
            window_size=window_size,
            neg_samples=neg_samples
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=0
            # Use 0 to avoid multiprocessing issues with generators
        )

        # Train for one epoch
        logger.info(f"Epoch {epoch+1}/{epochs}")
        epoch_timer = Timer()
        epoch_timer.start()

        epoch_loss = train_epoch(model, dataloader, optimizer, device)

        # Report progress
        elapsed = epoch_timer.elapsed()
        logger.info(
            f"Epoch {epoch+1} completed in {elapsed:.2f}s, "
            f"Loss: {epoch_loss:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f"pretrain_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch+1, epoch_loss, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save embeddings if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = checkpoint_path
            training_info["best_loss"] = best_loss
            training_info["best_model_path"] = best_model_path

            # Save best embeddings
            embeddings_path = os.path.join(embeddings_dir, "word_vectors.npy")
            save_embeddings(model, embeddings_path)
            logger.info(f"Best embeddings saved to {embeddings_path}")

        # Update training info
        training_info["epochs_completed"] = epoch + 1
        training_info["final_loss"] = epoch_loss

    # Save final embeddings
    final_embeddings_path = os.path.join(
        embeddings_dir, "word_vectors_final.npy")
    save_embeddings(model, final_embeddings_path)
    logger.info(f"Final embeddings saved to {final_embeddings_path}")

    # Export model in production-ready format if requested
    if export_dir:
        config = {
            "embed_dim": embed_dim,
            "window_size": window_size,
            "neg_samples": neg_samples,
            "min_count": getattr(vocab, "min_freq", 5)
        }
        export_paths = export_model(
            model,
            vocab,
            config,
            export_dir,
            version=f"0.1.{training_info['epochs_completed']}"
        )
        training_info["export_paths"] = export_paths
        logger.info(f"Model exported to {export_dir}")

    return model, training_info
