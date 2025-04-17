import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

class UpvoteRegressor(nn.Module):
    def __init__(self, embed_dim=100, hidden_dims=[128, 64], dropout=0.2):
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
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(input_dim, 1))
        
        # Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor of title embeddings [batch_size, embed_dim]
            
        Returns:
            Tensor of predicted upvote scores [batch_size, 1]
        """
        return self.model(x)
    
    def predict(self, title_embedding):
        """
        Make prediction for a single title
        
        Args:
            title_embedding: Embedding vector for a title
            
        Returns:
            Predicted upvote score
        """
        self.eval()
        with torch.no_grad():
            # Ensure input is a tensor with correct shape
            if not isinstance(title_embedding, torch.Tensor):
                title_embedding = torch.tensor(title_embedding, dtype=torch.float32)
            
            # Add batch dimension if needed
            if len(title_embedding.shape) == 1:
                title_embedding = title_embedding.unsqueeze(0)
            
            # Get prediction
            pred = self.forward(title_embedding)
            
            # Return as scalar
            return max(0, pred.item())  # Ensure non-negative prediction