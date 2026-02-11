"""
Professional logging system for SecureHealth-ML.
Provides colored console output and file logging.

"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            colored_level = (
                f"{self.COLORS[record.levelname]}"
                f"{self.BOLD}"
                f"{record.levelname:8s}"
                f"{self.RESET}"
            )
            record.levelname = colored_level
        
        return super().format(record)


class AppLogger:
    """
    Application logger with console and file output.
    
    Features:
    - Colored console output for better readability
    - File logging for permanent records
    - Separate log levels for console and file
    - Thread-safe logging
    
    Example:
        >>> logger = AppLogger(__name__)
        >>> logger.info("Training started")
        >>> logger.error("Failed to load model")
    """
    
    def __init__(
        self, 
        name: str, 
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
            log_dir: Directory for log files
            console_level: Logging level for console
            file_level: Logging level for file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        self._add_console_handler(console_level)
        
        # File handler
        if log_dir:
            self._add_file_handler(log_dir, file_level)
    
    def _add_console_handler(self, level: int) -> None:
        """Add colored console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        console_formatter = ColoredFormatter(
            fmt='%(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, log_dir: Path, level: int) -> None:
        """Add file handler with rotation."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{self.logger.name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)


# Factory function
def get_logger(name: str, log_dir: Optional[Path] = None) -> AppLogger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        log_dir: Optional log directory
    
    Returns:
        AppLogger instance
    """
    return AppLogger(name, log_dir=log_dir)