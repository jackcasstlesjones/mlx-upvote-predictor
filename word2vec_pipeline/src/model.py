from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipGramModel(nn.Module):
    """
    Skip-gram model implementation with negative sampling for Word2Vec.

    This model trains word embeddings by predicting context words from
    center words using the negative sampling objective from the original
    Word2Vec paper.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize Skip-gram model with separate input and output embeddings.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
        """
        super().__init__()

        # Input (center) embeddings
        self.in_embed = nn.Embedding(vocab_size, embed_dim)

        # Output (context) embeddings
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize weights
        self._init_weights()

        # Store configuration for later reference
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim
        }

    def _init_weights(self) -> None:
        """
        Initialize embedding weights using uniform distribution.

        Uses the initialization scheme described in the original Word2Vec paper
        to improve training stability.
        """
        # Initialize embeddings to small random values
        nn.init.uniform_(
            self.in_embed.weight, -0.5 /
            self.in_embed.embedding_dim, 0.5 /
            self.in_embed.embedding_dim
        )
        nn.init.uniform_(
            self.out_embed.weight, -0.5 /
            self.out_embed.embedding_dim, 0.5 /
            self.out_embed.embedding_dim
        )

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with negative sampling loss computation.

        Implements the negative sampling objective function from the Word2Vec
        paper.

        Args:
            centers: Tensor of center word indices [batch_size]
            contexts: Tensor of context word indices [batch_size]
            negatives: Tensor of negative sample indices
            [batch_size, n_negatives]

        Returns:
            torch.Tensor: Negative sampling loss
        """
        batch_size, n_negatives = negatives.shape

        # Get embeddings for all inputs
        # [batch_size, embed_dim]
        center_emb = self.in_embed(centers)
        # [batch_size, embed_dim]
        context_emb = self.out_embed(contexts)
        # [batch_size, n_negatives, embed_dim]
        negative_emb = self.out_embed(negatives)

        # Compute positive score (dot product of center and context vectors)
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [batch_size]

        # Calculate positive sample loss
        pos_loss = F.logsigmoid(pos_score)                      # [batch_size]

        # Calculate negative sample loss
        # First reshape for batch matrix multiplication
        # [batch_size, embed_dim, 1]
        center_emb = center_emb.unsqueeze(2)

        # Compute negative scores (batch matrix multiplication)
        neg_score = torch.bmm(negative_emb, center_emb).squeeze(
            2)  # [batch_size, n_negatives]

        # Calculate negative sample loss (use negative sign
        # for correct direction)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)          # [batch_size]

        # Combine losses (negative sign as we're minimizing)
        return -(pos_loss + neg_loss).mean()

    def get_in_embeddings(self) -> np.ndarray:
        """
        Return input embeddings as NumPy array.

        Returns:
            np.ndarray: Input embeddings with shape [vocab_size, embed_dim]
        """
        return self.in_embed.weight.detach().cpu().numpy()

    def get_out_embeddings(self) -> np.ndarray:
        """
        Return output embeddings as NumPy array.

        Returns:
            np.ndarray: Output embeddings with shape [vocab_size, embed_dim]
        """
        return self.out_embed.weight.detach().cpu().numpy()

    def get_model_info(self) -> Dict[str, any]:
        """
        Return model metadata dictionary.

        Returns:
            Dict: Dictionary containing model configuration and metadata
        """
        return {
            'model_type': 'SkipGramModel',
            'vocab_size': self.config['vocab_size'],
            'embed_dim': self.config['embed_dim']
        }
