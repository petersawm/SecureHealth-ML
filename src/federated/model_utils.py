"""
Utilities for model parameter handling in federated learning.
Convert between PyTorch models and numpy arrays.

Author: Person C (FL Engineer)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, OrderedDict
from ..utils.logger import get_logger


logger = get_logger(__name__)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as list of numpy arrays.
    
    This function converts PyTorch tensors to numpy arrays,
    which is necessary for Flower's serialization.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of parameter arrays (weights and biases)
    
    Example:
        >>> model = HealthcareNet()
        >>> params = get_parameters(model)
        >>> print(len(params))  # Number of parameter tensors
    """
    logger.debug(f"Extracting parameters from {model.__class__.__name__}")
    
    parameters = [
        param.detach().cpu().numpy()
        for param in model.parameters()
    ]
    
    logger.debug(f"Extracted {len(parameters)} parameter arrays")
    
    return parameters


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray]
) -> None:
    """
    Load parameters into model from numpy arrays.
    
    This function converts numpy arrays back to PyTorch tensors
    and loads them into the model.
    
    Args:
        model: PyTorch model to update
        parameters: List of parameter arrays
    
    Raises:
        ValueError: If parameter count or shapes don't match
    
    Example:
        >>> model = HealthcareNet()
        >>> params = get_parameters(model)
        >>> # ... parameters updated by server ...
        >>> set_parameters(model, params)
    """
    logger.debug(f"Loading parameters into {model.__class__.__name__}")
    
    # Get current model parameters
    model_params = list(model.parameters())
    
    # Validation
    if len(parameters) != len(model_params):
        raise ValueError(
            f"Parameter count mismatch: model has {len(model_params)}, "
            f"got {len(parameters)}"
        )
    
    # Set each parameter
    with torch.no_grad():
        for model_param, new_param in zip(model_params, parameters):
            # Validate shape
            if model_param.shape != new_param.shape:
                raise ValueError(
                    f"Shape mismatch: model parameter {model_param.shape}, "
                    f"new parameter {new_param.shape}"
                )
            
            # Convert numpy to tensor and copy
            model_param.copy_(torch.from_numpy(new_param))
    
    logger.debug("Parameters loaded successfully")


def get_state_dict_as_numpy(model: nn.Module) -> OrderedDict:
    """
    Get model state dict with numpy arrays instead of tensors.
    
    Args:
        model: PyTorch model
    
    Returns:
        OrderedDict mapping parameter names to numpy arrays
    """
    state_dict = model.state_dict()
    numpy_state_dict = OrderedDict()
    
    for name, tensor in state_dict.items():
        numpy_state_dict[name] = tensor.detach().cpu().numpy()
    
    return numpy_state_dict


def set_state_dict_from_numpy(
    model: nn.Module,
    numpy_state_dict: OrderedDict
) -> None:
    """
    Load state dict from numpy arrays.
    
    Args:
        model: PyTorch model
        numpy_state_dict: OrderedDict with numpy arrays
    """
    state_dict = OrderedDict()
    
    for name, numpy_array in numpy_state_dict.items():
        state_dict[name] = torch.from_numpy(numpy_array)
    
    model.load_state_dict(state_dict, strict=True)


def count_parameters(parameters: List[np.ndarray]) -> int:
    """
    Count total number of elements in parameter list.
    
    Args:
        parameters: List of parameter arrays
    
    Returns:
        Total number of parameters
    """
    return sum(param.size for param in parameters)


def parameters_to_bytes(parameters: List[np.ndarray]) -> int:
    """
    Calculate approximate memory size of parameters.
    
    Args:
        parameters: List of parameter arrays
    
    Returns:
        Approximate size in bytes
    """
    total_bytes = sum(
        param.nbytes for param in parameters
    )
    return total_bytes


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def log_parameter_info(parameters: List[np.ndarray]) -> None:
    """
    Log information about parameter list.
    
    Args:
        parameters: List of parameter arrays
    """
    n_params = count_parameters(parameters)
    n_bytes = parameters_to_bytes(parameters)
    
    logger.info(
        f"Parameters: {len(parameters)} arrays, "
        f"{n_params:,} total elements, "
        f"{format_bytes(n_bytes)}"
    )
    