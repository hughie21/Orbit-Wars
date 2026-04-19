"""
Logging system for Orbit Wars training framework.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "orbit_wars",
    log_dir: str = "./log",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    log_prefix: str = "train"
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to output to console
        file: Whether to save to file
        log_prefix: Prefix for log file name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also log to a latest.log file for easy tailing
        latest_file = os.path.join(log_dir, "latest.log")
        # Remove existing latest.log if exists
        if os.path.exists(latest_file):
            os.remove(latest_file)
        latest_fh = logging.FileHandler(latest_file, encoding='utf-8')
        latest_fh.setLevel(level)
        latest_fh.setFormatter(formatter)
        logger.addHandler(latest_fh)

    logger.info(f"Logger '{name}' initialized. Log file: {log_file if file else 'None'}")

    return logger


def get_logger(name: str = "orbit_wars") -> logging.Logger:
    """
    Get an existing logger by name.
    If logger doesn't exist, creates a basic one with console output only.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Setup a basic console logger
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# Default logger instance
logger = get_logger()