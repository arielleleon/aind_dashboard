"""
Cache utilities for AIND Dashboard

This module provides centralized caching functionality for:
- Session-level data
- Reference distributions and quantile data
- Threshold alerts and analysis results
- UI optimization structures
- Temporal cache validation

All caching is designed to reduce computation time for expensive operations
while maintaining data consistency and freshness.
"""

import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from app_utils.simple_logger import get_logger

logger = get_logger("cache_utils")


class CacheManager:
    """
    Cache manager for AIND Dashboard providing intelligent caching, compression, and invalidation
    """

    def __init__(self):
        """Initialize cache manager with empty cache structure"""
        # Main cache structure
        self._cache = {
            # Core data caches
            "raw_session_data": None,
            "session_level_data": None,
            "reference_distributions": None,
            "quantile_data": None,
            "threshold_alerts": None,
            # Optimized structures for UI components
            "optimized_storage": None,
            "ui_structures": None,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value by key

        Parameters:
            key: str
                Cache key to retrieve
            default: Any
                Default value if key not found

        Returns:
            Any: Cached value or default
        """
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value by key

        Parameters:
            key: str
                Cache key to set
            value: Any
                Value to cache
        """
        self._cache[key] = value

    def has(self, key: str) -> bool:
        """
        Check if cache has a key with non-None value

        Parameters:
            key: str
                Cache key to check

        Returns:
            bool: True if key exists and value is not None
        """
        return key in self._cache and self._cache[key] is not None

    def invalidate_derived_caches(self) -> None:
        """Reset all derived data caches when raw data changes"""
        self._cache["session_level_data"] = None
        self._cache["optimized_storage"] = None  # Optimized storage cache
        self._cache["ui_structures"] = None  # UI-optimized structures cache
        self._cache["threshold_alerts"] = None
        self._cache["last_process_time"] = None
        self._cache["data_hash"] = None
        self._cache["last_reference_calculation"] = None
        self._cache["last_alert_calculation"] = None

    def calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate a hash for data validation

        Parameters:
            df: pd.DataFrame
                DataFrame to calculate hash for

        Returns:
            str: Data hash for validation
        """
        data_str = f"{len(df)}_{df['subject_id'].nunique()}_{df['session_date'].max()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory usage summary for optimization monitoring

        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        try:
            import sys

            import psutil

            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()

            summary = {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "cache_sizes": {},
                "optimization_status": {},
                "total_data_objects": 0,
            }

            # Calculate cache sizes
            for cache_key, cache_data in self._cache.items():
                if cache_data is not None:
                    try:
                        cache_size = sys.getsizeof(cache_data)
                        summary["cache_sizes"][cache_key] = {
                            "size_mb": cache_size / 1024 / 1024,
                            "type": type(cache_data).__name__,
                        }

                        # Add detailed size for specific caches
                        if cache_key == "session_level_data" and hasattr(
                            cache_data, "__len__"
                        ):
                            summary["cache_sizes"][cache_key]["rows"] = len(cache_data)
                        elif cache_key == "ui_structures" and isinstance(
                            cache_data, dict
                        ):
                            for struct_key, struct_data in cache_data.items():
                                if isinstance(struct_data, dict):
                                    summary["cache_sizes"][
                                        f"{cache_key}.{struct_key}"
                                    ] = {
                                        "size_mb": sys.getsizeof(struct_data)
                                        / 1024
                                        / 1024,
                                        "count": len(struct_data),
                                    }
                        elif cache_key == "optimized_storage" and isinstance(
                            cache_data, dict
                        ):
                            metadata = cache_data.get("metadata", {})
                            summary["cache_sizes"][cache_key].update(
                                {
                                    "subjects": metadata.get("total_subjects", 0),
                                    "sessions": metadata.get("total_sessions", 0),
                                    "strata": metadata.get("total_strata", 0),
                                }
                            )

                    except Exception as e:
                        summary["cache_sizes"][cache_key] = {"error": str(e)}

            # Optimization status
            summary["optimization_status"] = {
                "unified_pipeline_active": self._cache.get("session_level_data")
                is not None,
                "optimized_storage_active": self._cache.get("optimized_storage")
                is not None,
                "ui_structures_active": self._cache.get("ui_structures") is not None,
                "memory_efficient_caching": True,  # We always use efficient caching now
            }

            # Calculate total cached objects
            for cache_data in self._cache.values():
                if cache_data is not None:
                    summary["total_data_objects"] += 1

            return summary

        except Exception as e:
            return {"error": f"Memory monitoring failed: {str(e)}"}

    def compress_cache_data(self):
        """
        Compress stored cache data when memory usage is high
        """
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB

            if process_memory > 1000:  # Only compress if using more than 1GB
                logger.info(
                    f"Compressing cache data (current memory: {process_memory:.1f}MB)..."
                )

                compressed_count = 0
                total_count = len(self._cache)

                for cache_key, cache_value in self._cache.items():
                    if cache_key.endswith("_compressed"):
                        continue

                    try:
                        if (
                            isinstance(cache_value, pd.DataFrame)
                            and len(cache_value) > 1000
                        ):
                            # Compress large DataFrames
                            original_size = cache_value.memory_usage(deep=True).sum()
                            compressed_data = pickle.dumps(
                                cache_value, protocol=pickle.HIGHEST_PROTOCOL
                            )

                            # Store compressed version
                            self._cache[f"{cache_key}_compressed"] = compressed_data

                            # Calculate compression ratio
                            compressed_size = len(compressed_data)
                            compression_ratio = original_size / compressed_size

                            logger.info(
                                f"  Compressed {cache_key}: {compression_ratio:.1f}x reduction"
                            )

                            # Remove original to save memory
                            del self._cache[cache_key]
                            compressed_count += 1

                    except Exception as e:
                        logger.error(f"  Failed to compress {cache_key}: {str(e)}")

                if compressed_count > 0:
                    logger.info(
                        f"Compressed {compressed_count}/{total_count} cache entries"
                    )
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            logger.error(f"Error in cache compression: {str(e)}")

    def set_timestamp(self, key: str = "last_process_time") -> None:
        """
        Set current timestamp for cache validation

        Parameters:
            key: str
                Timestamp key to set
        """
        self.set(key, datetime.now())
