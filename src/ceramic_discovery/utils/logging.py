"""Logging utilities for research reproducibility."""

import logging
import sys
from pathlib import Path
from typing import Optional

from ceramic_discovery.config import config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level from config or parameter
    log_level = level or config.logging.level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_path = log_file or config.logging.file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def log_workflow_start(logger: logging.Logger, workflow_id: str, parameters: dict) -> None:
    """
    Log the start of a workflow for reproducibility.
    
    Args:
        logger: Logger instance
        workflow_id: Unique workflow identifier
        parameters: Workflow parameters
    """
    logger.info(f"Starting workflow: {workflow_id}")
    logger.info(f"Parameters: {parameters}")
    logger.info(f"Random seed: {config.ml.random_seed}")


def log_workflow_end(logger: logging.Logger, workflow_id: str, status: str) -> None:
    """
    Log the end of a workflow.
    
    Args:
        logger: Logger instance
        workflow_id: Unique workflow identifier
        status: Workflow status (success, failed, etc.)
    """
    logger.info(f"Workflow {workflow_id} completed with status: {status}")
