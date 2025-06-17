"""
Shared utilities module to avoid circular imports and ensure single app_utils instance
"""

from app_utils import AppUtils

# CRITICAL FIX: Create a single shared app_utils instance
app_utils = AppUtils()


def get_app_utils():
    """Get the shared app_utils instance"""
    return app_utils
