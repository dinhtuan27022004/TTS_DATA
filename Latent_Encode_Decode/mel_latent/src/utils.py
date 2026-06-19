import os
import yaml
import torch
import numpy as np
import random

def load_config(config_path):
    """Loads configuration parameters from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Saves model and optimizer states to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Loads model and optimizer states from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    return epoch, loss

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    """Sets random seeds for reproducibility across random, numpy, and torch modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
