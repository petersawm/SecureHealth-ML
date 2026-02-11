import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from ..utils.logger import get_logger


logger = get_logger(__name__)


class HealthcareDataGenerator:
    """
    Generate synthetic healthcare data for federated learning.
    
    Simulates different patient populations across hospitals/clinics
    to create heterogeneous data distributions (non-IID).
    
    Features simulate:
    - Age, BMI, Blood Pressure
    - Glucose levels, Cholesterol
    - Heart rate, and other vital signs
    
    Example:
        >>> generator = HealthcareDataGenerator(random_seed=42)
        >>> features, labels = generator.generate(
        ...     n_samples=1000,
        ...     n_features=10,
        ...     client_id=0
        ... )
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        logger.debug(f"Initialized generator with seed={random_seed}")
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        client_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic patient data.
        
        Each client gets slightly different data distribution to
        simulate real-world heterogeneity across hospitals.
        
        Args:
            n_samples: Number of patient records
            n_features: Number of medical measurements
            client_id: Client identifier (affects distribution)
        
        Returns:
            Tuple of (features, labels)
            - features: shape (n_samples, n_features), normalized
            - labels: shape (n_samples,), binary (0 or 1)
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validation
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if client_id < 0:
            raise ValueError(f"client_id must be non-negative, got {client_id}")
        
        logger.info(
            f"Generating data: n_samples={n_samples}, n_features={n_features}, "
            f"client_id={client_id}"
        )
        
        # Set seed with client-specific variation
        seed = self.random_seed + client_id
        self.rng = np.random.RandomState(seed)
        
        # Generate features with client-specific distribution
        features = self._generate_features(n_samples, n_features, client_id)
        
        # Generate labels based on risk assessment
        labels = self._generate_labels(features)
        
        # Normalize features
        features = self._normalize_features(features)
        
        logger.debug(
            f"Generated data - features: {features.shape}, "
            f"labels: {np.bincount(labels)}"
        )
        
        return features, labels
    
    def _generate_features(
        self,
        n_samples: int,
        n_features: int,
        client_id: int
    ) -> np.ndarray:
        """
        Generate raw features with correlations.
        
        Client-specific mean shift simulates different patient
        populations across hospitals.
        """
        # Base distribution with client shift
        mean_shift = client_id * 0.5
        features = self.rng.randn(n_samples, n_features) + mean_shift
        
        # Add realistic correlations
        if n_features >= 3:
            # Feature 1 correlates with Feature 0 (e.g., BMI with age)
            features[:, 1] = (
                features[:, 0] * 0.5 +
                self.rng.randn(n_samples) * 0.5
            )
            
            # Feature 2 correlates with both (e.g., BP with age and BMI)
            features[:, 2] = (
                features[:, 0] * 0.3 +
                features[:, 1] * 0.3 +
                self.rng.randn(n_samples) * 0.4
            )
        
        return features
    
    def _generate_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Generate labels based on clinical risk assessment.
        
        Combines multiple features to determine disease risk.
        """
        n_samples = features.shape[0]
        
        # Calculate risk score (weighted combination of features)
        risk_score = np.zeros(n_samples)
        weights = [0.3, 0.25, 0.2, 0.15]  # Decreasing importance
        
        for i, weight in enumerate(weights):
            if i < features.shape[1]:
                risk_score += features[:, i] * weight
        
        # Add random variation (biological variability)
        risk_score += self.rng.randn(n_samples) * 0.5
        
        # Binary classification based on median risk
        threshold = np.median(risk_score)
        labels = (risk_score > threshold).astype(np.int64)
        
        return labels
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        This is crucial for neural network training.
        """
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
        return normalized


def generate_client_data(
    client_id: int,
    n_samples: int = 1000,
    n_features: int = 10,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate data for a single client.
    
    Args:
        client_id: Client identifier
        n_samples: Number of samples
        n_features: Number of features
        random_seed: Random seed
    
    Returns:
        (features, labels) tuple
    """
    generator = HealthcareDataGenerator(random_seed=random_seed)
    return generator.generate(n_samples, n_features, client_id)