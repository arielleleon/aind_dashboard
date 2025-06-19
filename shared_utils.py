"""
Shared utilities module to avoid circular imports and ensure single app_utils instance
Potentially deprecated singleton, may need to remove when data is no longer global among users.
"""

from app_utils import AppUtils

app_utils = AppUtils()


def get_shared_app_utils():
    """Get the shared app_utils instance"""
    return app_utils


def get_app_utils():
    """Get the shared app_utils instance (alias for compatibility)"""
    return app_utils
