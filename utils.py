"""
Utility functions for the diffusion model project
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def save_images(images, path, nrow=8, normalize=True):
    """
    Save a grid of images
    
    Args:
        images: Tensor of images [batch, channels, height, width]
        path: Where to save
        nrow: Number of images per row
        normalize: Whether to normalize from [-1, 1] to [0, 1]
    """
    if normalize:
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np, cmap='gray' if images.shape[1] == 1 else None)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_losses(losses, save_path):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        save_path: Where to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7)
    
    # Add moving average
    if len(losses) > 100:
        window = 100
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 
                linewidth=2, label=f'Moving Average (window={window})')
        plt.legend()
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
