import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import SkipgramDataset, create_dataloader
from .model import SkipGramModel
from .utils import (Timer, get_device, save_checkpoint, save_embeddings,
                    set_seed)


def train_epoch(model, dataloader, optimizer, device):
    """
    Train model for one epoch
    
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
            print(f"Batch {i+1}, Loss: {loss.item():.4f}, "
                  f"Speed: {100 / elapsed:.1f} batches/s")
            timer.start()
    
    # Return average loss
    return total_loss / total_batches

def train_model(
    token_stream_fn,
    vocab,
    embed_dim=100,
    window_size=5,
    neg_samples=10,
    learning_rate=0.002,
    batch_size=512,
    epochs=5,
    checkpoint_dir="checkpoints",
    embeddings_dir="embeddings",
    resume_from=None,
    seed=42
):
    """
    Train SkipGram Word2Vec model
    
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
        resume_from: Path to checkpoint to resume from
        seed: Random seed
        
    Returns:
        SkipGramModel: Trained model
    """
    # Set seeds for reproducibility
    set_seed(seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model
    vocab_size = len(vocab)
    model = SkipGramModel(vocab_size, embed_dim)
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, resume_from)
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        # Create a new dataset and dataloader for each epoch
        dataset = SkipgramDataset(
            token_stream_fn(),
            vocab,
            window_size=window_size,
            neg_samples=neg_samples
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=0  # Use 0 to avoid multiprocessing issues with generators
        )
        
        # Train for one epoch
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_timer = Timer()
        epoch_timer.start()
        
        epoch_loss = train_epoch(model, dataloader, optimizer, device)
        
        # Report progress
        elapsed = epoch_timer.elapsed()
        print(f"Epoch {epoch+1} completed in {elapsed:.2f}s, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"pretrain_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch+1, epoch_loss, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save embeddings if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            embeddings_path = os.path.join(embeddings_dir, "word_vectors.npy")
            save_embeddings(model, embeddings_path)
            print(f"Best embeddings saved to {embeddings_path}")
    
    # Save final embeddings
    final_embeddings_path = os.path.join(embeddings_dir, "word_vectors_final.npy")
    save_embeddings(model, final_embeddings_path)
    print(f"Final embeddings saved to {final_embeddings_path}")
    
    return model
