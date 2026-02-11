from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration - immutable."""
    host: str = "127.0.0.1"
    port: int = 8080
    
    @property
    def address(self) -> str:
        """Get server address string."""
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    input_size: int = 10
    hidden_size: int = 64
    output_size: int = 2
    dropout_rate: float = 0.3
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""
    num_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {self.num_rounds}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass(frozen=True)
class PrivacyConfig:
    """Differential privacy parameters."""
    enabled: bool = True
    target_epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    
    def __post_init__(self):
        """Validate privacy parameters."""
        if self.enabled:
            if self.target_epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {self.target_epsilon}")
            if not 0 < self.delta < 1:
                raise ValueError(f"delta must be in (0, 1), got {self.delta}")


@dataclass(frozen=True)
class FederatedConfig:
    """Federated learning configuration."""
    num_clients: int = 3
    min_available_clients: int = 2
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    
    def __post_init__(self):
        """Validate federated settings."""
        if self.min_available_clients > self.num_clients:
            raise ValueError(
                f"min_available_clients ({self.min_available_clients}) "
                f"cannot exceed num_clients ({self.num_clients})"
            )


@dataclass(frozen=True)
class DataConfig:
    """Data configuration."""
    num_samples_per_client: int = 1000
    test_split: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate data config."""
        if not 0 < self.test_split < 1:
            raise ValueError(f"test_split must be in (0, 1), got {self.test_split}")


@dataclass
class Config:
    """Main application configuration."""
    
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging
    verbose: bool = True
    log_dir: Optional[Path] = None
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        privacy_status = "Enabled" if self.privacy.enabled else "Disabled"
        return (
            f"\n{'='*60}\n"
            f"SecureHealth-ML Configuration\n"
            f"{'='*60}\n"
            f"Server       : {self.server.address}\n"
            f"Clients      : {self.federated.num_clients}\n"
            f"Rounds       : {self.training.num_rounds}\n"
            f"Batch Size   : {self.training.batch_size}\n"
            f"Privacy      : {privacy_status}\n"
            f"Epsilon (ε)  : {self.privacy.target_epsilon}\n"
            f"Model Hidden : {self.model.hidden_size}\n"
            f"{'='*60}\n"
        )


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration parameters."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")