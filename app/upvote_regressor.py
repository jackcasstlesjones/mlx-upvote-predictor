import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import time  # For tracking training time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class UpvoteRegressor(nn.Module):
    def __init__(self, embed_dim=100, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Neural network for predicting HN post upvotes based on title embeddings

        Args:
            embed_dim: Dimension of word embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # Create layers
        layers = []
        input_dim = embed_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))  # Using LeakyReLU for better gradient flow
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(input_dim, 1))
        
        # Apply ReLU activation to ensure non-negative predictions
        layers.append(nn.ReLU())

        # Sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Tensor of title embeddings [batch_size, embed_dim]

        Returns:
            Tensor of predicted upvote scores [batch_size, 1]
        """
        return self.model(x)


class TitleVectorizer:
    """Convert titles to embeddings using Skip-gram word vectors"""

    def __init__(self, word_vectors_path=None, model_path=None):
        """
        Initialize with word vectors and vocabulary
        
        Args:
            word_vectors_path: Path to word vectors .npy file
            model_path: Path to skipgram model
        """
        # Set default paths with parent directory to access from app folder
        base_dir = Path(__file__).parent.parent
        
        if word_vectors_path is None:
            word_vectors_path = str(base_dir / 'word2vec_pipeline/embeddings/word_vectors_final.npy')
            
        if model_path is None:
            model_path = str(base_dir / 'word2vec_pipeline/models/best_skipgram_model.pth')
            
        # Load word vectors
        self.word_vectors = np.load(word_vectors_path)
        print(f"Loaded word vectors with shape: {self.word_vectors.shape}")
        
        # Use a simple approach for tokenization without requiring the full model
        # This might not match the exact tokenization used in training, but should work for simple demos
        self.embed_dim = self.word_vectors.shape[1]
        
    def get_title_vector(self, title):
        """
        Get embedding vector for a title by averaging word embeddings

        Args:
            title: Input title string

        Returns:
            ndarray: Title embedding vector
        """
        # Handle None or empty titles
        if title is None or not title:
            return np.zeros(self.embed_dim)
            
        # Simple tokenization - lowercasing and splitting by whitespace
        try:
            tokens = title.lower().split()
        except AttributeError:
            # Handle case where title is not a string
            print(f"Warning: Non-string title encountered: {title} (type: {type(title)})")
            return np.zeros(self.embed_dim)

        if not tokens:
            return np.zeros(self.embed_dim)

        # For each token, get a deterministic embedding based on the actual word vectors
        embeddings = []
        for token in tokens:
            # Simple tokenization - remove punctuation at ends of words
            token = token.strip('.,;:!?()[]{}"\'')
            
            # Skip empty tokens
            if not token:
                continue
                
            # Use a hash function to create a deterministic index into our word vectors
            # This isn't the same as using the trained vocabulary but will be consistent
            hash_value = sum(ord(c) for c in token)
            vocab_size = self.word_vectors.shape[0]
            idx = (hash_value % (vocab_size - 1)) + 1  # Avoid index 0 which is usually <UNK>
            
            embeddings.append(self.word_vectors[idx])
        
        # Handle case where no valid tokens remain
        if not embeddings:
            return np.zeros(self.embed_dim)

        # Average embeddings to get title vector
        return np.mean(embeddings, axis=0)


class HNDataset(Dataset):
    """Dataset for Hacker News titles and scores"""

    def __init__(self, titles, scores, vectorizer, precompute=False, max_precompute=100000):
        """
        Initialize dataset

        Args:
            titles: List of post titles
            scores: List of upvote scores
            vectorizer: TitleVectorizer instance
            precompute: Whether to precompute vectors (only for small datasets)
            max_precompute: Maximum number of samples to precompute
        """
        self.titles = titles
        self.scores = scores
        self.vectorizer = vectorizer
        self.title_vectors = None

        # Only precompute if explicitly requested and dataset is small enough
        if precompute and len(titles) <= max_precompute:
            print(f"Precomputing {len(titles)} title vectors...")
            self.title_vectors = []
            for i, title in enumerate(titles):
                self.title_vectors.append(self.vectorizer.get_title_vector(title))
                if i % 10000 == 0 and i > 0:
                    print(f"  Precomputed {i}/{len(titles)} vectors...")
            print("Precomputation complete.")

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        # Get title vector - either precomputed or compute on-the-fly
        if self.title_vectors is not None:
            vector = self.title_vectors[idx]
        else:
            # Compute vector on the fly
            vector = self.vectorizer.get_title_vector(self.titles[idx])
            
        score = self.scores[idx]

        return {
            'vector': torch.tensor(vector, dtype=torch.float32),
            'score': torch.tensor(score, dtype=torch.float32)
        }


def train_and_save_model(model_save_path="model.pth", force_retrain=False, two_stage=True):
    """
    Train the model and save it to disk
    
    Args:
        model_save_path: Path to save the trained model
        force_retrain: Whether to force retraining even if the model exists
        two_stage: Whether to use two-stage training (small subset first, then full dataset)
        
    Returns:
        Trained model
    """
    # Check if model already exists
    if not force_retrain and os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}")
        
        # Initialize model with correct embedding dimension
        vectorizer = TitleVectorizer()
        embed_dim = vectorizer.word_vectors.shape[1]
        model = UpvoteRegressor(embed_dim=embed_dim)
        
        # Load model weights
        model.load_state_dict(torch.load(model_save_path))
        model.eval()  # Set to evaluation mode
        return model
    
    # Model doesn't exist or retraining is forced
    print("Training new model...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Hugging Face datasets package not found. Please install with: pip install datasets")
        raise ImportError("Missing required package 'datasets'.")

    print("Loading Hacker News dataset from Hugging Face...")
    
    # Load the dataset
    ds = load_dataset("loredanagaspar/hn_title_modeling_dataset_with_tokens")
    print(f"Dataset loaded successfully: {ds}")

    # Initialize vectorizer
    vectorizer = TitleVectorizer()

    # Prepare training data
    train_data = ds['train']  # Assuming there's a 'train' split

    # Process the entire dataset but filter out problematic samples
    print(f"Processing dataset with {len(train_data)} total samples")
    
    # Display a few sample titles
    print("\nSample titles from HN dataset:")
    for i in range(5):  # Just show first 5 samples
        print(f"  - \"{train_data[i]['title']}\" (Score: {train_data[i]['score']})")
    
    # Get titles and scores from the dataset
    raw_titles = train_data['title']
    raw_scores = train_data['score']
    
    # Filter out None or invalid entries
    print("Filtering out invalid samples...")
    all_titles = []
    all_scores = []
    
    for i, (title, score) in enumerate(zip(raw_titles, raw_scores)):
        # Check for valid title and score
        if title is not None and score is not None and score > 0:
            all_titles.append(title)
            all_scores.append(score)
        
        # Show progress
        if i % 1000000 == 0 and i > 0:
            print(f"  Processed {i}/{len(raw_titles)} samples...")
    
    print(f"After filtering: {len(all_titles)}/{len(raw_titles)} valid samples remaining")

    # Define a function for the actual training process
    def train_model(titles, scores, model_path, epochs=20, lr=0.002, 
                   batch_size=256, patience=5, model_state=None, 
                   description="Training"):
        print(f"\n{description}...")
        print(f"Using {len(titles)} samples for training")

        # Create dataset
        dataset = HNDataset(titles, scores, vectorizer)

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        print(f"Created DataLoader with batch size {batch_size}")

        # Initialize model
        embed_dim = vectorizer.word_vectors.shape[1]
        model = UpvoteRegressor(embed_dim=embed_dim)
        print(f"Initialized model with embedding dimension {embed_dim}")
        
        # If we have a saved state from initial training, load it
        if model_state is not None:
            model.load_state_dict(model_state)
            print("Loaded model state from initial training")

        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Enhanced training loop with progress tracking
        
        # Track best model
        best_loss = float('inf')
        best_epoch = 0
        best_state = None
        
        # Initialize early stopping counter
        no_improve_count = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            
            total_loss = 0
            batch_count = 0
            
            # Time tracking
            start_time = time.time()
            
            for batch in dataloader:
                # Get data
                vectors = batch['vector']
                scores = batch['score']

                # Forward pass
                outputs = model(vectors)
                loss = criterion(outputs.squeeze(), scores)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                
                # Print progress every 1000 batches
                if batch_count % 1000 == 0:
                    print(f"  Batch {batch_count}/{len(dataloader)}, Current loss: {loss.item():.4f}")
            
            # Calculate epoch stats
            epoch_loss = total_loss / len(dataloader)
            epoch_time = time.time() - start_time
            
            # Print detailed epoch stats
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Check if this is the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_state = model.state_dict().copy()
                
                # Save the best model
                best_path = str(model_path) + '.best'
                try:
                    torch.save(best_state, best_path)
                    print(f"  ✓ New best model saved to {best_path} with loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"  ! Error saving best model: {e}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  - No improvement for {no_improve_count} epochs. Best loss: {best_loss:.4f} at epoch {best_epoch}")
                
            # Early stopping check
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1} as no improvement for {patience} epochs")
                break
        
        # Load the best model before returning
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Loaded best model from epoch {best_epoch} with loss {best_loss:.4f}")
        
        # Save the final trained model
        print(f"Saving model to {model_path}")
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Model successfully saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        return model, best_state
    
    # STAGE 1: Train on a small subset first
    if two_stage:
        print("\n=== STAGE 1: Initial training on subset ===")
        
        # Use a small subset for initial training
        subset_size = min(50000, len(all_titles))
        indices = np.random.choice(len(all_titles), subset_size, replace=False)
        subset_titles = [all_titles[i] for i in indices]
        subset_scores = [all_scores[i] for i in indices]
        
        # Train on the subset
        temp_path = str(model_save_path) + ".stage1"
        stage1_model, stage1_state = train_model(
            subset_titles, subset_scores, 
            temp_path, epochs=20, lr=0.002, 
            batch_size=256, patience=3,
            description="Training on initial subset"
        )
        
        print("\n=== STAGE 1 Complete: Initial model trained successfully ===")
        
        # STAGE 2: Now train on the full dataset
        print("\n=== STAGE 2: Training on full dataset ===")
        
        # Train on the full dataset, starting from the best model of stage 1
        final_model, _ = train_model(
            all_titles, all_scores, 
            model_save_path, epochs=50, lr=0.001,
            batch_size=256, patience=5, 
            model_state=stage1_state,
            description="Training on full dataset"
        )
        
        print("\n=== STAGE 2 Complete: Full model trained successfully ===")
        return final_model
    
    else:
        # Single stage training on the full dataset
        print("\n=== Single-stage training on full dataset ===")
        print(f"Training on all {len(all_titles)} samples")
        
        final_model, _ = train_model(
            all_titles, all_scores, 
            model_save_path, epochs=50, lr=0.002,
            batch_size=256, patience=5,
            description="Training on full dataset"
        )
        
        # Verify model has been saved
        if os.path.exists(model_save_path):
            model_size = os.path.getsize(model_save_path) / (1024 * 1024)  # Size in MB
            print(f"Model successfully saved to {model_save_path} ({model_size:.2f} MB)")
        else:
            print(f"WARNING: Model file not found at {model_save_path}!")
            
        # Check for best model
        best_model_path = str(model_save_path) + '.best'
        if os.path.exists(best_model_path):
            model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # Size in MB
            print(f"Best model saved to {best_model_path} ({model_size:.2f} MB)")
        
        return final_model


def demo(model_save_path="model.pth", force_retrain=False, use_best=True):
    """
    Run a demonstration of the upvote regressor
    
    Args:
        model_save_path: Path to save/load the model
        force_retrain: Whether to force retraining even if the model exists
        use_best: Whether to use the best model from training
    """
    try:
        # Get trained model (train if needed)
        model = train_and_save_model(model_save_path, force_retrain)
        
        # Load the best model if available and requested
        best_model_path = str(model_save_path) + '.best'
        if use_best and os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            vectorizer = TitleVectorizer()
            embed_dim = vectorizer.word_vectors.shape[1]
            model = UpvoteRegressor(embed_dim=embed_dim)
            model.load_state_dict(torch.load(best_model_path))
            model.eval()  # Set to evaluation mode
        
        # Initialize vectorizer for predictions
        vectorizer = TitleVectorizer()
        
        # Test prediction on a variety of sample titles
        test_titles = [
            "Steve Jobs is dead at 56",
            "Show HN: I built a tool that automatically writes code",
            "Ask HN: What technology are you excited about?",
            "Announcing Python 4.0",
            "Why I switched from React to Svelte",
            "Web Assembly is the future of web development",
            "A minimalist guide to the command line",
            "How I built a profitable SaaS in 6 months",
            "The ethics of AI in modern society",
            "Learning Rust made me a better programmer"
        ]
        
        print("\nSample predictions:")
        for test_title in test_titles:
            # Get title vector
            test_vector = vectorizer.get_title_vector(test_title)
            test_tensor = torch.tensor(
                test_vector, dtype=torch.float32).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                prediction = model(test_tensor)
                prediction = max(0, prediction.item())  # Ensure non-negative

            print(f"Title: {test_title}")
            print(f"Predicted upvotes: {prediction:.1f}")
            print("-" * 50)

    except Exception as e:
        print(f"Error loading or processing the dataset: {e}")
        # Re-raise the exception
        raise


if __name__ == "__main__":
    demo()
