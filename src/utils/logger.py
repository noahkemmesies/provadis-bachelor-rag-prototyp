"""Logging configuration and utilities"""

import logging
import logging.handlers
from pathlib import Path

from src.config import LOG_FILE, LOG_LEVEL, LOG_DIR


def setup_logger(name: str = "rag_evaluation", level: str = LOG_LEVEL) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # File Handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()
