"""Logging utilities for FinRisk Analytics."""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name='finrisk', log_dir='logs', level=logging.INFO):
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_config(logger, config):
    """Log configuration settings."""
    logger.info("=" * 60)
    logger.info("Configuration Settings")
    logger.info("=" * 60)

    for section, params in config.items():
        logger.info(f"\n{section.upper()}:")
        if isinstance(params, dict):
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {params}")

    logger.info("=" * 60)

def log_data_summary(logger, data_dict):
    """Log summary of loaded data."""
    logger.info("\n" + "=" * 60)
    logger.info("Data Summary")
    logger.info("=" * 60)

    for ticker, df in data_dict.items():
        logger.info(f"\n{ticker}:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date Range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"  Missing Values: {df.isnull().sum().sum()}")

    logger.info("=" * 60)
