import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Skip-gram model with separate input and output embeddings

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

    def _init_weights(self):
        """Initialize embedding weights"""
        # Initialize embeddings to small random values
        nn.init.uniform_(self.in_embed.weight, -0.5 /
                         self.in_embed.embedding_dim, 0.5/self.in_embed.embedding_dim)
        nn.init.uniform_(self.out_embed.weight, -0.5 /
                         self.out_embed.embedding_dim, 0.5/self.out_embed.embedding_dim)

    def forward(self, centers, contexts, negatives):
        """
        Forward pass with negative sampling loss computation

        Args:
            centers: Tensor of center word indices [batch_size]
            contexts: Tensor of context word indices [batch_size]
            negatives: Tensor of negative sample indices [batch_size, n_negatives]

        Returns:
            loss: Negative sampling loss
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

        # Calculate negative sample loss (use negative sign for correct direction)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)          # [batch_size]

        # Combine losses (negative sign as we're minimizing)
        return -(pos_loss + neg_loss).mean()

    def get_in_embeddings(self):
        """Return input embeddings"""
        return self.in_embed.weight.detach().cpu().numpy()

    def get_out_embeddings(self):
        """Return output embeddings"""
        return self.out_embed.weight.detach().cpu().numpy()

