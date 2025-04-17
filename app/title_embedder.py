import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add word2vec_pipeline to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from word2vec_pipeline.src.tokenize import tokenize
from word2vec_pipeline.src.vocab import Vocabulary

class TitleEmbedder:
    def __init__(self, word_vectors_path=None, vocab_path=None):
        """
        Initialize the embedder with trained Skip-gram word vectors
        
        Args:
            word_vectors_path: Path to word vectors .npy file
            vocab_path: Path to vocabulary pickle file
        """
        # Set default paths if not provided
        if word_vectors_path is None:
            word_vectors_path = Path(__file__).parent.parent / 'word2vec_pipeline/embeddings/word_vectors_final.npy'
        if vocab_path is None:
            vocab_path = Path(__file__).parent.parent / 'word2vec_pipeline/data/processed/vocab.pkl'
        
        # Load word vectors
        self.word_vectors = np.load(word_vectors_path)
        print(f"Loaded word vectors with shape: {self.word_vectors.shape}")
        
        # Load vocabulary
        self.vocab = Vocabulary.load(vocab_path)
        print(f"Loaded vocabulary with {len(self.vocab)} words")
    
    def get_title_embedding(self, title):
        """
        Convert a title to its embedding by averaging word embeddings
        
        Args:
            title: Input title string
            
        Returns:
            ndarray: Title embedding vector
        """
        # Tokenize title using the same tokenizer used in word2vec training
        tokens = tokenize(title)
        
        if not tokens:
            # Return zeros for empty titles
            return np.zeros(self.word_vectors.shape[1])
        
        # Convert tokens to indices
        token_ids = [self.vocab.get_index(token) for token in tokens]
        
        # Get embeddings for each token
        embeddings = []
        for idx in token_ids:
            if idx > 0 and idx < len(self.word_vectors):  # Skip <UNK> token (index 0)
                embeddings.append(self.word_vectors[idx])
            else:
                # For unknown words, use zero vector
                embeddings.append(np.zeros(self.word_vectors.shape[1]))
        
        # Average embeddings
        if embeddings:
            title_embedding = np.mean(embeddings, axis=0)
        else:
            title_embedding = np.zeros(self.word_vectors.shape[1])
        
        return title_embedding