import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HealthcarePreprocessor:
    """
    Preprocessing pipeline for healthcare data.
    
    Handles:
    - Missing value imputation
    - Feature scaling
    - Outlier detection
    - Data validation
    """
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        handle_outliers: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard' or 'minmax'
            handle_outliers: Whether to clip outliers
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown scaling method: {scaling_method}"
            )
        
        logger.info(
            f"Initialized preprocessor with {scaling_method} scaling"
        )
    
    def fit(self, data: np.ndarray) -> 'HealthcarePreprocessor':
        """
        Fit preprocessing parameters to data.
        
        Args:
            data: Training data to fit on
        
        Returns:
            Self for method chaining
        """
        logger.debug(f"Fitting preprocessor on data shape {data.shape}")
        self.scaler.fit(data)
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform
        
        Returns:
            Transformed data
        """
        # Handle outliers if enabled
        if self.handle_outliers:
            data = self._clip_outliers(data)
        
        # Scale features
        scaled_data = self.scaler.transform(data)
        
        logger.debug(f"Transformed data shape {data.shape}")
        return scaled_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Data to fit and transform
        
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def _clip_outliers(
        self,
        data: np.ndarray,
        n_std: float = 3.0
    ) -> np.ndarray:
        """
        Clip outliers beyond n standard deviations.
        
        Args:
            data: Input data
            n_std: Number of standard deviations
        
        Returns:
            Data with outliers clipped
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        clipped = np.clip(data, lower_bound, upper_bound)
        
        n_clipped = np.sum(
            (data < lower_bound) | (data > upper_bound)
        )
        if n_clipped > 0:
            logger.debug(f"Clipped {n_clipped} outlier values")
        
        return clipped
    
    def validate_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validate data quality.
        
        Args:
            features: Feature array
            labels: Label array
        
        Returns:
            (is_valid, error_message)
        """
        # Check for NaN values
        if np.isnan(features).any():
            return False, "Features contain NaN values"
        if np.isnan(labels).any():
            return False, "Labels contain NaN values"
        
        # Check for infinite values
        if np.isinf(features).any():
            return False, "Features contain infinite values"
        
        # Check shapes match
        if features.shape[0] != labels.shape[0]:
            return False, "Feature and label counts don't match"
        
        # Check labels are binary
        unique_labels = np.unique(labels)
        if not np.array_equal(unique_labels, [0, 1]):
            return False, f"Labels should be 0 or 1, got {unique_labels}"
        
        return True, ""