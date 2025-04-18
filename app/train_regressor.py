import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import psycopg
import datetime
import time

from upvote_regressor import UpvoteRegressor 
from title_embedder import TitleEmbedder

class HNPostDataset(Dataset):
    def __init__(self, titles, scores, embedder):
        """
        Dataset for Hacker News posts with titles and scores
        
        Args:
            titles: List of post titles
            scores: List of upvote scores
            embedder: TitleEmbedder instance
        """
        self.titles = titles
        self.scores = scores
        self.embedder = embedder
        
        # Pre-compute embeddings to speed up training
        self.embeddings = []
        for title in titles:
            self.embeddings.append(self.embedder.get_title_embedding(title))
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        # Get pre-computed embedding and score
        embedding = self.embeddings[idx]
        score = self.scores[idx]
        
        return {
            'embedding': torch.tensor(embedding, dtype=torch.float32),
            'score': torch.tensor(score, dtype=torch.float32)
        }

def fetch_hn_data(connection_string):
    """
    Fetch Hacker News data from PostgreSQL database
    
    Args:
        connection_string: Database connection string
        
    Returns:
        tuple: (titles, scores) lists
    """
    titles = []
    scores = []
    
    # Connect to database
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            # Query posts with title and score
            cur.execute("""
                SELECT title, score 
                FROM posts 
                WHERE score > 0 
                ORDER BY created_at DESC 
                LIMIT 10000
            """)
            
            # Fetch results
            for title, score in cur.fetchall():
                titles.append(title)
                scores.append(score)
    
    print(f"Fetched {len(titles)} posts from database")
    return titles, scores

def train_model(train_loader, val_loader, embed_dim, hidden_dims=[128, 64], 
                dropout=0.2, lr=0.001, epochs=10, device='cpu'):
    """
    Train the upvote regressor model
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        embed_dim: Dimension of word embeddings
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Trained model and training history
    """
    # Initialize model
    model = UpvoteRegressor(embed_dim, hidden_dims, dropout)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Best validation loss for model checkpoint
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        batch_times = []
        samples_processed = 0
        epoch_start_time = time.time()
        train_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Get data
            embeddings = batch['embedding'].to(device)
            scores = batch['score'].to(device)
            batch_size = embeddings.size(0)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), scores)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            train_loss += loss.item() * batch_size
            samples_processed += batch_size
            
            # Print batch progress (every 10 batches)
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                samples_per_sec = batch_size / avg_batch_time
                elapsed = time.time() - train_start_time
                progress = (batch_idx + 1) / len(train_loader) * 100
                
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) - "
                      f"Loss: {loss.item():.4f} - "
                      f"Batch time: {batch_time:.3f}s - "
                      f"Samples/sec: {samples_per_sec:.1f} - "
                      f"Elapsed: {elapsed:.1f}s")
        
        # Calculate average training loss and timing stats
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        train_time = time.time() - train_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_samples_per_sec = samples_processed / train_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                embeddings = batch['embedding'].to(device)
                scores = batch['score'].to(device)
                
                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs.squeeze(), scores)
                
                val_loss += loss.item() * embeddings.size(0)
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'  Train: {train_time:.2f}s - {len(train_loader)} batches - {avg_samples_per_sec:.1f} samples/sec - Loss: {train_loss:.4f}')
        print(f'  Val: {val_time:.2f}s - Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create model directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/upvote_regressor_best.pt')
            print(f"  Saved best model with validation loss: {val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'models/upvote_regressor_final.pt')
    print(f"Saved final model")
    
    return model, history

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train upvote regression model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='128,64', help='Hidden dimensions (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize embedder
    embedder = TitleEmbedder()
    embed_dim = embedder.word_vectors.shape[1]
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Database connection string from BRIEF.md
    connection_string = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
    
    try:
        # Fetch data from database
        titles, scores = fetch_hn_data(connection_string)
        
        # Create dataset
        dataset = HNPostDataset(titles, scores, embedder)
        
        # Split into train and validation sets (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Train model
        print(f"Starting training with configuration:")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Hidden dimensions: {hidden_dims}")
        print(f"  Dropout rate: {args.dropout}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        
        model, history = train_model(
            train_loader, 
            val_loader, 
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            device=device
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()