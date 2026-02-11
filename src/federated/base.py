from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..utils.logger import get_logger


logger = get_logger(__name__)


class FLClientInterface(ABC):
    """
    Abstract base class for federated learning clients.
    
    All client implementations must implement these methods.
    This provides a clean contract for client behavior.
    """
    
    def __init__(self, client_id: int):
        """
        Initialize client.
        
        Args:
            client_id: Unique client identifier
        """
        self.client_id = client_id
        logger.debug(f"Initialized FL client {client_id}")
    
    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays.
        
        Returns:
            List of parameter arrays
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of parameter arrays
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration (epochs, lr, etc.)
        
        Returns:
            Tuple of:
            - Updated parameters
            - Number of training examples
            - Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local test data.
        
        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration
        
        Returns:
            Tuple of:
            - Loss value
            - Number of evaluation examples
            - Metrics dictionary (e.g., accuracy)
        """
        pass


class FLServerInterface(ABC):
    """
    Abstract base class for federated learning servers.
    
    Defines the contract for server-side FL logic.
    """
    
    @abstractmethod
    def aggregate_fit(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict]]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate training results from multiple clients.
        
        Args:
            results: List of (parameters, num_examples, metrics) from clients
        
        Returns:
            Tuple of:
            - Aggregated parameters
            - Aggregated metrics
        """
        pass
    
    @abstractmethod
    def aggregate_evaluate(
        self,
        results: List[Tuple[float, int, Dict]]
    ) -> Tuple[float, Dict]:
        """
        Aggregate evaluation results from multiple clients.
        
        Args:
            results: List of (loss, num_examples, metrics) from clients
        
        Returns:
            Tuple of:
            - Aggregated loss
            - Aggregated metrics
        """
        pass
    
    @abstractmethod
    def configure_fit(self, server_round: int) -> Dict[str, Any]:
        """
        Configure parameters for training round.
        
        Args:
            server_round: Current round number
        
        Returns:
            Configuration dictionary sent to clients
        """
        pass
    
    @abstractmethod
    def configure_evaluate(self, server_round: int) -> Dict[str, Any]:
        """
        Configure parameters for evaluation round.
        
        Args:
            server_round: Current round number
        
        Returns:
            Configuration dictionary sent to clients
        """
        pass


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.
    
    Different strategies (FedAvg, FedProx, etc.) can be implemented
    by inheriting from this class.
    """
    
    @abstractmethod
    def aggregate(
        self,
        parameters_list: List[List[np.ndarray]],
        weights: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate parameters from multiple clients.
        
        Args:
            parameters_list: List of parameter lists from clients
            weights: Weight for each client (usually num_examples)
        
        Returns:
            Aggregated parameters
        """
        pass
    
    @abstractmethod
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        weights: List[int]
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple clients.
        
        Args:
            metrics_list: List of metric dictionaries
            weights: Weight for each client
        
        Returns:
            Aggregated metrics
        """
        pass
    