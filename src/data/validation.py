import numpy as np
from typing import Tuple, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


def check_data_distribution(
    features: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    Analyze data distribution.
    
    Args:
        features: Feature array
        labels: Label array
    
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'n_samples': len(labels),
        'n_features': features.shape[1],
        'label_balance': {
            'class_0': np.sum(labels == 0),
            'class_1': np.sum(labels == 1),
            'ratio': np.sum(labels == 1) / len(labels)
        },
        'feature_ranges': {
            'min': features.min(axis=0).tolist(),
            'max': features.max(axis=0).tolist(),
            'mean': features.mean(axis=0).tolist(),
            'std': features.std(axis=0).tolist()
        }
    }
    
    # Check for class imbalance
    ratio = stats['label_balance']['ratio']
    if ratio < 0.3 or ratio > 0.7:
        logger.warning(
            f"Class imbalance detected: {ratio:.2%} positive class"
        )
    
    return stats


def check_feature_correlation(
    features: np.ndarray,
    threshold: float = 0.95
) -> Dict:
    """
    Check for highly correlated features.
    
    Args:
        features: Feature array
        threshold: Correlation threshold
    
    Returns:
        Dictionary with correlation info
    """
    correlation_matrix = np.corrcoef(features.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    n_features = features.shape[1]
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature_1': i,
                    'feature_2': j,
                    'correlation': correlation_matrix[i, j]
                })
    
    if high_corr_pairs:
        logger.warning(
            f"Found {len(high_corr_pairs)} highly correlated feature pairs"
        )
    
    return {
        'correlation_matrix': correlation_matrix.tolist(),
        'high_correlation_pairs': high_corr_pairs
    }