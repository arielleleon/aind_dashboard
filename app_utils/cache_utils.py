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

from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import pickle
import gzip
import os
import hashlib
from app_utils.simple_logger import get_logger

logger = get_logger('cache_utils')

class CacheManager:
    """
    Cache manager for AIND Dashboard providing intelligent caching, compression, and invalidation
    """
    
    def __init__(self):
        """Initialize cache manager with empty cache structure"""
        # Main cache structure
        self._cache = {
            # Core data caches
            'raw_session_data': None,
            'session_level_data': None,
            'reference_distributions': None,
            'quantile_data': None,
            'threshold_alerts': None,
            
            # Optimized structures for UI components
            'optimized_storage': None,
            'ui_structures': None
        }
        
        # Timestamps for cache validation
        self._timestamps = {
            'last_data_load': None,
            'last_process_time': None,
            'last_reference_calculation': None,
            'last_alert_calculation': None
        }
        
        # Cache metadata
        self._metadata = {
            'data_hash': None,
            'feature_config_hash': None,
            'cache_size_bytes': 0,
            'compression_enabled': True
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
        self._cache['session_level_data'] = None
        self._cache['optimized_storage'] = None  # Optimized storage cache
        self._cache['ui_structures'] = None      # UI-optimized structures cache
        self._cache['threshold_alerts'] = None
        self._cache['last_process_time'] = None
        self._cache['data_hash'] = None
        self._cache['last_reference_calculation'] = None
        self._cache['last_alert_calculation'] = None
    
    def clear_all(self) -> None:
        """Clear all cached data"""
        for key in self._cache:
            self._cache[key] = None
    
    def clear_key(self, key: str) -> None:
        """Clear specific cache key"""
        if key in self._cache:
            self._cache[key] = None
    
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
    
    def is_cache_valid(self, key: str, data_hash: Optional[str] = None) -> bool:
        """
        Check if cached data is still valid
        
        Parameters:
            key: str
                Cache key to validate
            data_hash: Optional[str]
                Current data hash for validation
                
        Returns:
            bool: True if cache is valid
        """
        if not self.has(key):
            return False
        
        # If data hash provided, compare with cached hash
        if data_hash is not None:
            cached_hash = self.get('data_hash')
            return cached_hash == data_hash
        
        return True
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state
        
        Returns:
            Dict[str, Any]: Cache state information
        """
        cache_info = {}
        
        for key, value in self._cache.items():
            if value is not None:
                cache_info[key] = {
                    'exists': True,
                    'type': type(value).__name__
                }
                
                # Add specific info for different cache types
                if isinstance(value, pd.DataFrame):
                    cache_info[key]['rows'] = len(value)
                    cache_info[key]['columns'] = len(value.columns)
                elif isinstance(value, dict):
                    cache_info[key]['size'] = len(value)
                elif isinstance(value, list):
                    cache_info[key]['length'] = len(value)
                elif isinstance(value, set):
                    cache_info[key]['size'] = len(value)
            else:
                cache_info[key] = {'exists': False}
        
        return cache_info
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory usage summary for optimization monitoring
        
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        try:
            import sys
            
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            summary = {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'cache_sizes': {},
                'optimization_status': {},
                'total_data_objects': 0
            }
            
            # Calculate cache sizes
            for cache_key, cache_data in self._cache.items():
                if cache_data is not None:
                    try:
                        cache_size = sys.getsizeof(cache_data)
                        summary['cache_sizes'][cache_key] = {
                            'size_mb': cache_size / 1024 / 1024,
                            'type': type(cache_data).__name__
                        }
                        
                        # Add detailed size for specific caches
                        if cache_key == 'session_level_data' and hasattr(cache_data, '__len__'):
                            summary['cache_sizes'][cache_key]['rows'] = len(cache_data)
                        elif cache_key == 'ui_structures' and isinstance(cache_data, dict):
                            for struct_key, struct_data in cache_data.items():
                                if isinstance(struct_data, dict):
                                    summary['cache_sizes'][f'{cache_key}.{struct_key}'] = {
                                        'size_mb': sys.getsizeof(struct_data) / 1024 / 1024,
                                        'count': len(struct_data)
                                    }
                        elif cache_key == 'optimized_storage' and isinstance(cache_data, dict):
                            metadata = cache_data.get('metadata', {})
                            summary['cache_sizes'][cache_key].update({
                                'subjects': metadata.get('total_subjects', 0),
                                'sessions': metadata.get('total_sessions', 0),
                                'strata': metadata.get('total_strata', 0)
                            })
                            
                    except Exception as e:
                        summary['cache_sizes'][cache_key] = {'error': str(e)}
            
            # Optimization status
            summary['optimization_status'] = {
                'unified_pipeline_active': self._cache.get('session_level_data') is not None,
                'optimized_storage_active': self._cache.get('optimized_storage') is not None,
                'ui_structures_active': self._cache.get('ui_structures') is not None,
                'memory_efficient_caching': True  # We always use efficient caching now
            }
            
            # Calculate total cached objects
            for cache_data in self._cache.values():
                if cache_data is not None:
                    summary['total_data_objects'] += 1
            
            return summary
            
        except Exception as e:
            return {'error': f"Memory monitoring failed: {str(e)}"}
    
    def compress_cache_data(self):
        """
        Compress stored cache data when memory usage is high
        """
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if process_memory > 1000:  # Only compress if using more than 1GB
            logger.info(f"Compressing cache data (current memory: {process_memory:.1f}MB)...")
            
            compressed_count = 0
            total_count = len(self._cache)
            
            for cache_key, cache_value in self._cache.items():
                if cache_key.endswith('_compressed'):
                    continue
                    
                try:
                    if isinstance(cache_value, pd.DataFrame) and len(cache_value) > 1000:
                        # Compress large DataFrames
                        original_size = cache_value.memory_usage(deep=True).sum()
                        compressed_data = pickle.dumps(cache_value, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        # Store compressed version
                        self._cache[f"{cache_key}_compressed"] = compressed_data
                        
                        # Calculate compression ratio
                        compressed_size = len(compressed_data)
                        compression_ratio = original_size / compressed_size
                        
                        logger.info(f"  Compressed {cache_key}: {compression_ratio:.1f}x reduction")
                        
                        # Remove original to save memory
                        del self._cache[cache_key]
                        compressed_count += 1
                        
                except Exception as e:
                    logger.error(f"  Failed to compress {cache_key}: {str(e)}")
            
            if compressed_count > 0:
                logger.info(f"Compressed {compressed_count}/{total_count} cache entries")
    
    def decompress_cache_data(self, cache_key: str) -> Any:
        """
        Decompress cache data if it exists in compressed form
        """
        compressed_key = f"{cache_key}_compressed"
        
        if compressed_key in self._cache:
            try:
                # Decompress the data
                decompressed_data = pickle.loads(self._cache[compressed_key])
                
                # Store decompressed version back in cache
                self._cache[cache_key] = decompressed_data
                
                logger.info(f"Decompressed {cache_key} successfully")
                return decompressed_data
                
            except Exception as e:
                logger.error(f"Failed to decompress {cache_key}: {str(e)}")
                return None
        
        return None
    
    def get_or_decompress(self, key: str, default: Any = None) -> Any:
        """
        Get cached value, decompressing if necessary
        
        Parameters:
            key: str
                Cache key to retrieve
            default: Any
                Default value if key not found
                
        Returns:
            Any: Cached value or default
        """
        # First try to get directly
        value = self.get(key, None)
        if value is not None:
            return value
        
        # Try to decompress if compressed version exists
        if self.decompress_cache_data(key):
            return self.get(key, default)
        
        return default
    
    def set_timestamp(self, key: str = 'last_process_time') -> None:
        """
        Set current timestamp for cache validation
        
        Parameters:
            key: str
                Timestamp key to set
        """
        self.set(key, datetime.now())
    
    def get_cache_age(self, key: str = 'last_process_time') -> Optional[float]:
        """
        Get age of cached data in seconds
        
        Parameters:
            key: str
                Timestamp key to check
                
        Returns:
            Optional[float]: Age in seconds or None if no timestamp
        """
        timestamp = self.get(key)
        if timestamp is not None and isinstance(timestamp, datetime):
            return (datetime.now() - timestamp).total_seconds()
        return None
    
    def is_cache_fresh(self, key: str = 'last_process_time', max_age_seconds: float = 3600) -> bool:
        """
        Check if cached data is fresh (within max age)
        
        Parameters:
            key: str
                Timestamp key to check
            max_age_seconds: float
                Maximum age in seconds before cache is stale
                
        Returns:
            bool: True if cache is fresh
        """
        age = self.get_cache_age(key)
        if age is None:
            return False
        return age <= max_age_seconds 