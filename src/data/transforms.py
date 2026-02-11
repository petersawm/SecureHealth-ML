import numpy as np
import torch
from typing import Callable, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AddGaussianNoise:
    """Add Gaussian noise for data augmentation."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        """
        Initialize noise transform.
        
        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of noise
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise to tensor."""
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class Normalize:
    """Normalize tensor to zero mean and unit variance."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize normalization.
        
        Args:
            mean: Target mean
            std: Target standard deviation
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        return (tensor - self.mean) / self.std


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: list):
        """
        Initialize composition.
        
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor