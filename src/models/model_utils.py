import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def get_model_size(model: nn.Module) -> str:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Formatted size string
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return f"{size_mb:.2f} MB"


def freeze_layers(
    model: nn.Module,
    layer_names: List[str]
) -> nn.Module:
    """
    Freeze specific layers.
    
    Args:
        model: PyTorch model
        layer_names: Names of layers to freeze
    
    Returns:
        Model with frozen layers
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            logger.debug(f"Froze layer: {name}")
    
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Unfreeze all layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model with all layers unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True
    
    logger.debug("Unfroze all layers")
    return model


def init_weights(model: nn.Module, method: str = 'xavier') -> nn.Module:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        method: Initialization method
    
    Returns:
        Model with initialized weights
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(module.weight)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    logger.info(f"Initialized weights with {method} method")
    return model