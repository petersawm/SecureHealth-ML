from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset
import numpy as np


class HealthcareDataset(Dataset):
    """
    PyTorch Dataset for healthcare patient data.
    
    Features:
    - Type-safe tensor conversion
    - Input validation
    - Optional data transformations
    - Comprehensive statistics
    
    Attributes:
        features: Patient medical measurements
        labels: Binary diagnosis (0=healthy, 1=at-risk)
        transform: Optional data transformation callable
    
    Example:
        >>> features = np.random.randn(100, 10)
        >>> labels = np.random.randint(0, 2, 100)
        >>> dataset = HealthcareDataset(features, labels)
        >>> print(len(dataset))  # 100
        >>> sample, label = dataset[0]
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset with validation.
        
        Args:
            features: Shape (n_samples, n_features)
            labels: Shape (n_samples,)
            transform: Optional transformation function
        
        Raises:
            ValueError: If shapes are inconsistent
            TypeError: If inputs are not numpy arrays
        """
        # Type validation
        if not isinstance(features, np.ndarray):
            raise TypeError(f"features must be numpy array, got {type(features)}")
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"labels must be numpy array, got {type(labels)}")
        
        # Shape validation
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {labels.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Sample count mismatch: features={features.shape[0]}, "
                f"labels={labels.shape[0]}"
            )
        
        # Convert to tensors
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        # Cache dimensions
        self.n_samples = len(self.labels)
        self.n_features = features.shape[1]
    
    def __len__(self) -> int:
        """Get dataset size."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            (features, label) tuple
        """
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_statistics(self) -> dict:
        """
        Calculate dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        label_counts = torch.bincount(self.labels)
        
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'label_distribution': {
                'healthy (0)': label_counts[0].item() if len(label_counts) > 0 else 0,
                'at_risk (1)': label_counts[1].item() if len(label_counts) > 1 else 0,
            },
            'feature_stats': {
                'mean': self.features.mean(dim=0).tolist(),
                'std': self.features.std(dim=0).tolist(),
                'min': self.features.min(dim=0)[0].tolist(),
                'max': self.features.max(dim=0)[0].tolist(),
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HealthcareDataset(\n"
            f"  samples={self.n_samples},\n"
            f"  features={self.n_features},\n"
            f"  label_dist={dict(self.get_statistics()['label_distribution'])}\n"
            f")"
        )