import os
import numpy as np
import torch
import random
import time

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device():
    """
    Get the appropriate device (CPU/GPU)
    
    Returns:
        torch.device: Device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    
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
    
def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        path: Path to checkpoint
    
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    
def save_embeddings(model, path):
    """
    Save model embeddings to file
    
    Args:
        model: PyTorch model
        path: Path to save embeddings
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get embeddings
    embeddings = model.get_in_embeddings()
    
    # Save embeddings
    np.save(path, embeddings)
    
class Timer:
    """Simple timer class for tracking execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start timer"""
        self.start_time = time.time()
        
    def stop(self):
        """Stop timer"""
        self.end_time = time.time()
        
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time