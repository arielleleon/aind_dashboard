"""
Simple logging utility for AIND Dashboard

This replaces scattered print() statements with structured logging.
Only 2 levels: DEV (shows INFO+) and PROD (shows WARNING+ only).

The goal is to REDUCE complexity, not add it.
"""

import logging
import os


class SimpleLogger:
    """Minimal logging utility that replaces print statements"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with environment-based levels"""
        if self.logger.handlers:
            return  # Already configured

        # Determine log level based on environment
        # DEV: Show INFO and above
        # PROD (default): Show WARNING and above
        # The default environment is now 'PROD' to minimize console output during normal app use.
        is_dev = os.getenv("DASH_ENV", "PROD").upper() == "DEV"
        level = logging.INFO if is_dev else logging.WARNING

        self.logger.setLevel(level)

        # Simple console handler
        handler = logging.StreamHandler()

        # Include timestamp and level for better debugging
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        """Info level - shown in DEV only"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Warning level - shown in DEV and PROD"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Error level - always shown"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.error(message)


def get_logger(name: str) -> SimpleLogger:
    """Get a logger instance for a module"""
    return SimpleLogger(name)
