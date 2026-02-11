"""
Metrics tracking and calculation utilities.

"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Metrics:
    """Container for training/evaluation metrics."""
    
    loss: float
    accuracy: float
    num_examples: int
    additional: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Metrics(loss={self.loss:.4f}, "
            f"accuracy={self.accuracy:.2f}%, "
            f"n={self.num_examples})"
        )


class MetricsTracker:
    """
    Track metrics over multiple rounds.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.add_round(1, loss=0.5, accuracy=85.0)
        >>> print(tracker.get_best_accuracy())
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.rounds: List[int] = []
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.custom_metrics: Dict[str, List[float]] = {}
    
    def add_round(
        self, 
        round_num: int, 
        loss: float, 
        accuracy: float,
        **kwargs
    ) -> None:
        """
        Add metrics for a round.
        
        Args:
            round_num: Round number
            loss: Loss value
            accuracy: Accuracy percentage
            **kwargs: Additional metrics
        """
        self.rounds.append(round_num)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        for key, value in kwargs.items():
            if key not in self.custom_metrics:
                self.custom_metrics[key] = []
            self.custom_metrics[key].append(value)
    
    def get_best_accuracy(self) -> Optional[float]:
        """Get best accuracy achieved."""
        return max(self.accuracies) if self.accuracies else None
    
    def get_final_accuracy(self) -> Optional[float]:
        """Get final accuracy."""
        return self.accuracies[-1] if self.accuracies else None
    
    def get_improvement(self) -> Optional[float]:
        """Calculate accuracy improvement from first to last round."""
        if len(self.accuracies) < 2:
            return None
        return self.accuracies[-1] - self.accuracies[0]
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.accuracies:
            return {}
        
        return {
            'num_rounds': len(self.rounds),
            'best_accuracy': self.get_best_accuracy(),
            'final_accuracy': self.get_final_accuracy(),
            'improvement': self.get_improvement(),
            'avg_loss': np.mean(self.losses),
            'final_loss': self.losses[-1],
        }


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
    
    Returns:
        Accuracy percentage (0-100)
    """
    correct = (predictions == targets).sum()
    total = len(targets)
    return 100.0 * correct / total if total > 0 else 0.0


def weighted_average(
    values: List[float], 
    weights: List[int]
) -> float:
    """
    Calculate weighted average.
    
    Args:
        values: Values to average
        weights: Weights for each value
    
    Returns:
        Weighted average
    """
    if not values or not weights:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight