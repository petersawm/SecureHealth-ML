import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..utils.logger import get_logger


logger = get_logger(__name__)


class HealthcareNet(nn.Module):
    """
    Feedforward neural network for binary health classification.
    
    Architecture:
    - Input layer: patient features
    - Hidden layer 1: Linear + BatchNorm + ReLU + Dropout
    - Hidden layer 2: Linear + BatchNorm + ReLU + Dropout
    - Output layer: Linear (binary classification logits)
    
    Features:
    - Batch normalization for training stability
    - Dropout for regularization
    - Xavier initialization
    - Proper forward pass with activation functions
    
    Example:
        >>> model = HealthcareNet(input_size=10, hidden_size=64)
        >>> x = torch.randn(32, 10)  # batch_size=32, features=10
        >>> logits = model(x)  # shape: (32, 2)
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        output_size: int = 2,
        dropout_rate: float = 0.3
    ):
        """
        Initialize model architecture.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer dimension
            output_size: Number of output classes
            dropout_rate: Dropout probability (0-1)
        
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validation
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Hidden layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.debug(
            f"Initialized HealthcareNet: "
            f"in={input_size}, hidden={hidden_size}, out={output_size}, "
            f"dropout={dropout_rate}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Logits of shape (batch_size, output_size)
        """
        # Hidden layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation, returns logits)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier initialization.
        
        Xavier initialization helps with gradient flow and
        prevents vanishing/exploding gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_num_parameters(self) -> int:
        """
        Count total trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_count_by_layer(self) -> dict:
        """
        Get parameter count for each layer.
        
        Returns:
            Dictionary mapping layer names to parameter counts
        """
        counts = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                counts[name] = param.numel()
        return counts
    
    def __repr__(self) -> str:
        """String representation."""
        n_params = self.get_num_parameters()
        return (
            f"HealthcareNet(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  output_size={self.output_size},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  parameters={n_params:,}\n"
            f")"
        )


def create_model(
    input_size: int = 10,
    hidden_size: int = 64,
    output_size: int = 2,
    dropout_rate: float = 0.3,
    device: Optional[torch.device] = None
) -> HealthcareNet:
    """
    Factory function to create and initialize model.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer dimension
        output_size: Number of output classes
        dropout_rate: Dropout probability
        device: Device to move model to
    
    Returns:
        Initialized HealthcareNet model
    """
    model = HealthcareNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    )
    
    if device is not None:
        model = model.to(device)
    
    logger.info(f"Created model with {model.get_num_parameters():,} parameters")
    
    return model