import numpy as np
from typing import List, Dict, Tuple
from .base import AggregationStrategy
from ..utils.logger import get_logger


logger = get_logger(__name__)


class FederatedAveraging(AggregationStrategy):
    """
    Federated Averaging (FedAvg) algorithm.
    
    Computes weighted average of model parameters based on
    the number of training examples from each client.
    
    Paper: McMahan et al., "Communication-Efficient Learning
    of Deep Networks from Decentralized Data", 2017
    
    Formula:
        w_global = Σ(n_i * w_i) / Σ(n_i)
        where n_i is the number of examples at client i
    
    Example:
        >>> strategy = FederatedAveraging()
        >>> params = strategy.aggregate(
        ...     parameters_list=[client1_params, client2_params],
        ...     weights=[100, 200]  # num_examples
        ... )
    """
    
    def aggregate(
        self,
        parameters_list: List[List[np.ndarray]],
        weights: List[int]
    ) -> List[np.ndarray]:
        """
        Perform federated averaging on parameters.
        
        Args:
            parameters_list: List of parameter lists from each client
            weights: Number of examples per client
        
        Returns:
            Aggregated parameters (weighted average)
        
        Raises:
            ValueError: If inputs are invalid or inconsistent
        """
        # Validation
        if not parameters_list:
            raise ValueError("parameters_list cannot be empty")
        if not weights:
            raise ValueError("weights cannot be empty")
        if len(parameters_list) != len(weights):
            raise ValueError(
                f"Length mismatch: {len(parameters_list)} parameter sets "
                f"but {len(weights)} weights"
            )
        
        n_clients = len(parameters_list)
        total_examples = sum(weights)
        
        if total_examples == 0:
            raise ValueError("Total weight is zero")
        
        logger.debug(
            f"Aggregating {n_clients} clients "
            f"with {total_examples} total examples"
        )
        
        # Weighted sum of parameters
        # Initialize with zeros matching first client's structure
        aggregated = [np.zeros_like(param) for param in parameters_list[0]]
        
        for client_params, weight in zip(parameters_list, weights):
            # Validate parameter structure matches
            if len(client_params) != len(aggregated):
                raise ValueError(
                    f"Parameter count mismatch: expected {len(aggregated)}, "
                    f"got {len(client_params)}"
                )
            
            # Add weighted parameters
            for i, param in enumerate(client_params):
                if param.shape != aggregated[i].shape:
                    raise ValueError(
                        f"Shape mismatch at layer {i}: "
                        f"expected {aggregated[i].shape}, got {param.shape}"
                    )
                aggregated[i] += param * weight
        
        # Divide by total weight to get average
        aggregated = [param / total_examples for param in aggregated]
        
        logger.debug(f"Aggregation complete: {len(aggregated)} parameter arrays")
        
        return aggregated
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        weights: List[int]
    ) -> Dict[str, float]:
        """
        Aggregate metrics using weighted average.
        
        Args:
            metrics_list: List of metric dictionaries from clients
            weights: Number of examples per client
        
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
        
        if len(metrics_list) != len(weights):
            raise ValueError(
                f"Length mismatch: {len(metrics_list)} metric dicts "
                f"but {len(weights)} weights"
            )
        
        total_examples = sum(weights)
        if total_examples == 0:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        aggregated = {}
        for key in all_keys:
            weighted_sum = sum(
                metrics.get(key, 0.0) * weight
                for metrics, weight in zip(metrics_list, weights)
            )
            aggregated[key] = weighted_sum / total_examples
        
        return aggregated


def aggregate_losses(
    losses: List[float],
    num_examples: List[int]
) -> float:
    """
    Aggregate loss values using weighted average.
    
    Args:
        losses: Loss value from each client
        num_examples: Number of examples per client
    
    Returns:
        Weighted average loss
    """
    if not losses or not num_examples:
        return 0.0
    
    if len(losses) != len(num_examples):
        raise ValueError(
            f"Length mismatch: {len(losses)} losses "
            f"but {len(num_examples)} weights"
        )
    
    total_examples = sum(num_examples)
    if total_examples == 0:
        return 0.0
    
    weighted_loss = sum(
        loss * n_ex for loss, n_ex in zip(losses, num_examples)
    )
    
    return weighted_loss / total_examples


def validate_parameter_structure(
    parameters_list: List[List[np.ndarray]]
) -> Tuple[bool, str]:
    """
    Validate that all parameter lists have consistent structure.
    
    Args:
        parameters_list: List of parameter lists to validate
    
    Returns:
        (is_valid, error_message)
    """
    if not parameters_list:
        return False, "Empty parameters list"
    
    reference = parameters_list[0]
    n_layers = len(reference)
    
    for i, params in enumerate(parameters_list[1:], 1):
        # Check number of layers
        if len(params) != n_layers:
            return False, f"Client {i}: expected {n_layers} layers, got {len(params)}"
        
        # Check shapes
        for j, (ref_param, param) in enumerate(zip(reference, params)):
            if param.shape != ref_param.shape:
                return False, (
                    f"Client {i}, layer {j}: "
                    f"expected shape {ref_param.shape}, got {param.shape}"
                )
    
    return True, ""
