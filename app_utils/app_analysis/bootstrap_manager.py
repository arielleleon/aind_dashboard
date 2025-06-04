import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
from .statistical_utils import StatisticalUtils


class BootstrapManager:
    """
    Bootstrap Manager for Phase 3 Enhanced Reference Distributions
    
    Orchestrates bootstrap generation with:
    - Tiered approach based on strata size (large/medium/small)
    - 30-day update detection based on session_date
    - Integration with existing strata processing and caching
    - Validation and quality control with graceful degradation
    """
    
    def __init__(self, bootstrap_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Bootstrap Manager
        
        Parameters:
            bootstrap_config: Optional[Dict[str, Any]]
                Bootstrap configuration (uses defaults if None)
        """
        # Default configuration following the tiered strategy
        self.default_config = {
            'large_strata': {
                'threshold': 80,  # subjects
                'bootstrap_samples': 2000,
                'update_frequency_days': 30
            },
            'medium_strata': {
                'threshold': 20,  # subjects  
                'bootstrap_samples': 1000,
                'update_frequency_days': 30
            },
            'small_strata': {
                'threshold': 0,   # subjects
                'bootstrap_samples': 0,  # Skip bootstrap
                'update_frequency_days': None
            },
            'validation': {
                'enable_monotonicity_check': True,
                'enable_quality_comparison': True,
                'strict_validation': False,  # Log warnings, don't block
                'min_agreement_threshold': 0.10,  # Max relative difference acceptable
                'fallback_to_standard': True
            },
            'performance': {
                'random_state': 42,  # For reproducibility
                'percentile_grid': None,  # Will use default 1-99
                'enable_caching': True,
                'compress_storage': True
            }
        }
        
        # Override with provided config
        self.config = self.default_config.copy()
        if bootstrap_config:
            self._update_config_recursive(self.config, bootstrap_config)
        
        # Initialize statistical utilities
        self.statistical_utils = StatisticalUtils()
        
        # Bootstrap cache for generated distributions
        self._bootstrap_cache = {}
        self._last_update_check = None
        
        print(f"BootstrapManager initialized with tiered strategy:")
        print(f"  Large strata (≥{self.config['large_strata']['threshold']} subjects): {self.config['large_strata']['bootstrap_samples']} samples")
        print(f"  Medium strata (≥{self.config['medium_strata']['threshold']} subjects): {self.config['medium_strata']['bootstrap_samples']} samples") 
        print(f"  Small strata (<{self.config['medium_strata']['threshold']} subjects): Skip bootstrap")
    
    def _update_config_recursive(self, base_config: Dict, update_config: Dict) -> None:
        """Recursively update nested configuration dictionaries"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def should_update_bootstrap(self, strata_data: Dict[str, pd.DataFrame], session_dates: Optional[pd.Series] = None) -> Dict[str, bool]:
        """
        Determine which strata need bootstrap updates based on 30-day rule
        
        Parameters:
            strata_data: Dict[str, pd.DataFrame]
                Dictionary of strata dataframes (from ReferenceProcessor)
            session_dates: Optional[pd.Series]
                Recent session dates for update detection
                
        Returns:
            Dict[str, bool]
                Dictionary mapping strata names to whether they need updates
        """
        update_needed = {}
        current_time = pd.Timestamp.now()
        
        # Get most recent session date from data if not provided
        if session_dates is not None:
            most_recent_session = session_dates.max()
        else:
            most_recent_session = current_time  # Fallback to current time
        
        for strata_name, strata_df in strata_data.items():
            # Determine strata tier
            subject_count = len(strata_df)
            strata_tier = self._get_strata_tier(subject_count)
            
            # Skip small strata (no bootstrap)
            if strata_tier == 'small_strata':
                update_needed[strata_name] = False
                continue
            
            # Check if we have cached bootstrap for this strata
            cached_bootstrap = self._bootstrap_cache.get(strata_name)
            
            if cached_bootstrap is None:
                # No cached bootstrap - need to generate
                update_needed[strata_name] = True
                continue
            
            # Check timestamp of cached bootstrap
            generation_time = cached_bootstrap.get('generation_timestamp')
            if generation_time is None:
                # Invalid cached bootstrap - need to regenerate
                update_needed[strata_name] = True
                continue
            
            # Calculate days since last bootstrap generation
            days_since_generation = (current_time - generation_time).days
            update_frequency = self.config[strata_tier]['update_frequency_days']
            
            # Check if bootstrap is older than update frequency
            if days_since_generation >= update_frequency:
                update_needed[strata_name] = True
            else:
                update_needed[strata_name] = False
        
        return update_needed
    
    def _get_strata_tier(self, subject_count: int) -> str:
        """Determine strata tier based on subject count"""
        if subject_count >= self.config['large_strata']['threshold']:
            return 'large_strata'
        elif subject_count >= self.config['medium_strata']['threshold']:
            return 'medium_strata'
        else:
            return 'small_strata'
    
    def generate_bootstrap_for_strata(self, 
                                    strata_name: str,
                                    strata_data: pd.DataFrame,
                                    force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate bootstrap reference distribution for a specific strata
        
        Parameters:
            strata_name: str
                Name of the strata
            strata_data: pd.DataFrame
                Strata dataframe with processed features and outlier weights
            force_regenerate: bool
                Force regeneration even if cached version exists
                
        Returns:
            Dict[str, Any]
                Bootstrap distribution results with validation metadata
        """
        result = {
            'strata_name': strata_name,
            'subject_count': len(strata_data),
            'bootstrap_enabled': False,
            'bootstrap_distributions': {},
            'validation_results': {},
            'quality_metrics': {},
            'generation_timestamp': pd.Timestamp.now(),
            'warnings': []
        }
        
        # Determine strata tier and configuration
        subject_count = len(strata_data)
        strata_tier = self._get_strata_tier(subject_count)
        tier_config = self.config[strata_tier]
        
        result['strata_tier'] = strata_tier
        result['tier_config'] = tier_config.copy()
        
        # Skip bootstrap for small strata
        if strata_tier == 'small_strata':
            result['bootstrap_enabled'] = False
            result['reason'] = f'Small strata ({subject_count} < {self.config["medium_strata"]["threshold"]} subjects) - skipping bootstrap'
            return result
        
        # Check cache if not forcing regeneration
        if not force_regenerate and strata_name in self._bootstrap_cache:
            cached_result = self._bootstrap_cache[strata_name]
            if cached_result.get('bootstrap_enabled', False):
                print(f"Using cached bootstrap for strata '{strata_name}'")
                return cached_result
        
        print(f"Generating bootstrap for strata '{strata_name}' ({strata_tier}, {subject_count} subjects)")
        
        # Get processed feature columns
        processed_features = [col for col in strata_data.columns if col.endswith('_processed')]
        
        if not processed_features:
            result['warnings'].append('No processed features found for bootstrap generation')
            return result
        
        # Generate bootstrap for each processed feature
        successful_features = 0
        
        for feature_col in processed_features:
            feature_name = feature_col.replace('_processed', '')
            
            # Get feature data and weights
            feature_values = strata_data[feature_col].values
            feature_weights_data = strata_data.get('outlier_weight', np.ones(len(feature_values)))
            
            # Handle both pandas Series and numpy array cases
            if hasattr(feature_weights_data, 'values'):
                feature_weights = feature_weights_data.values
            else:
                feature_weights = feature_weights_data
            
            # Remove NaN values
            valid_mask = ~np.isnan(feature_values)
            clean_values = feature_values[valid_mask]
            clean_weights = feature_weights[valid_mask]
            
            if len(clean_values) < 3:
                result['warnings'].append(f'Insufficient data for {feature_name} ({len(clean_values)} valid values)')
                continue
            
            try:
                # Generate bootstrap distribution
                bootstrap_dist = self.statistical_utils.generate_bootstrap_reference_distribution(
                    reference_data=clean_values,
                    reference_weights=clean_weights,
                    n_bootstrap=tier_config['bootstrap_samples'],
                    percentile_grid=self.config['performance']['percentile_grid'],
                    random_state=self.config['performance']['random_state']
                )
                
                if bootstrap_dist.get('bootstrap_enabled', False):
                    # Validate bootstrap distribution
                    validation_result = self._validate_bootstrap_distribution(
                        feature_name=feature_name,
                        reference_data=clean_values,
                        reference_weights=clean_weights,
                        bootstrap_distribution=bootstrap_dist
                    )
                    
                    # Store results
                    result['bootstrap_distributions'][feature_name] = bootstrap_dist
                    result['validation_results'][feature_name] = validation_result
                    successful_features += 1
                    
                    print(f" {feature_name}: {bootstrap_dist['bootstrap_samples']} samples, validation={'passed' if validation_result['recommend_bootstrap'] else '⚠️ warnings'}")
                else:
                    error_msg = bootstrap_dist.get('error', 'Unknown bootstrap generation error')
                    result['warnings'].append(f'{feature_name}: {error_msg}')
                    print(f" {feature_name}: {error_msg}")
                    
            except Exception as e:
                error_msg = f'Bootstrap generation failed for {feature_name}: {str(e)}'
                result['warnings'].append(error_msg)
                print(f" {feature_name}: {str(e)}")
        
        # Determine overall success
        if successful_features > 0:
            result['bootstrap_enabled'] = True
            result['successful_features'] = successful_features
            result['total_features'] = len(processed_features)
            
            # Cache the result
            if self.config['performance']['enable_caching']:
                self._bootstrap_cache[strata_name] = result
            
            print(f" Bootstrap generation complete: {successful_features}/{len(processed_features)} features successful")
        else:
            result['reason'] = 'No features successfully generated bootstrap distributions'
            print(f" Bootstrap generation failed for strata '{strata_name}'")
        
        return result
    
    def _validate_bootstrap_distribution(self,
                                       feature_name: str,
                                       reference_data: np.ndarray,
                                       reference_weights: np.ndarray,
                                       bootstrap_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate bootstrap distribution quality and make recommendation
        
        Parameters:
            feature_name: str
                Name of the feature being validated
            reference_data: np.ndarray
                Original reference data
            reference_weights: np.ndarray
                Reference weights from outlier detection
            bootstrap_distribution: Dict[str, Any]
                Generated bootstrap distribution
                
        Returns:
            Dict[str, Any]
                Validation results with recommendation
        """
        validation_result = {
            'feature_name': feature_name,
            'recommend_bootstrap': False,
            'monotonicity_valid': True,
            'quality_acceptable': True,
            'warnings': [],
            'validation_timestamp': pd.Timestamp.now()
        }
        
        try:
            # 1. Monotonicity validation
            if self.config['validation']['enable_monotonicity_check']:
                monotonicity = self.statistical_utils.validate_percentile_monotonicity(
                    percentile_values=bootstrap_distribution['percentile_values'],
                    percentile_grid=bootstrap_distribution['percentile_grid']
                )
                
                validation_result['monotonicity_check'] = monotonicity
                validation_result['monotonicity_valid'] = monotonicity.get('monotonicity_valid', False)
                
                if not validation_result['monotonicity_valid']:
                    warning_msg = f'Monotonicity violations detected for {feature_name}'
                    validation_result['warnings'].append(warning_msg)
                    if not self.config['validation']['strict_validation']:
                        print(f" {warning_msg} (continuing with degraded CIs)")
            
            # 2. Quality comparison with standard method
            if self.config['validation']['enable_quality_comparison']:
                quality_comparison = self.statistical_utils.compare_bootstrap_vs_standard(
                    reference_data=reference_data,
                    bootstrap_distribution=bootstrap_distribution,
                    reference_weights=reference_weights
                )
                
                validation_result['quality_comparison'] = quality_comparison
                
                # Check agreement threshold
                quality_metrics = quality_comparison.get('quality_metrics', {})
                max_rel_diff = quality_metrics.get('max_relative_difference', float('inf'))
                
                if max_rel_diff > self.config['validation']['min_agreement_threshold']:
                    validation_result['quality_acceptable'] = False
                    warning_msg = f'Poor agreement with standard method for {feature_name} (max_rel_diff={max_rel_diff:.3f})'
                    validation_result['warnings'].append(warning_msg)
                    if not self.config['validation']['strict_validation']:
                        print(f" {warning_msg} (continuing with degraded CIs)")
            
            # 3. Make final recommendation
            if self.config['validation']['strict_validation']:
                # Strict mode: require both monotonicity and quality
                validation_result['recommend_bootstrap'] = (
                    validation_result['monotonicity_valid'] and 
                    validation_result['quality_acceptable']
                )
            else:
                # Lenient mode: recommend bootstrap if generated successfully
                # Warnings are logged but don't block usage
                validation_result['recommend_bootstrap'] = bootstrap_distribution.get('bootstrap_enabled', False)
            
        except Exception as e:
            validation_result['recommend_bootstrap'] = False
            validation_result['error'] = str(e)
            validation_result['warnings'].append(f'Validation failed for {feature_name}: {str(e)}')
        
        return validation_result
    
    def generate_bootstrap_for_all_strata(self,
                                        strata_data: Dict[str, pd.DataFrame],
                                        session_dates: Optional[pd.Series] = None,
                                        force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate bootstrap distributions for all eligible strata
        
        Parameters:
            strata_data: Dict[str, pd.DataFrame]
                Dictionary of strata dataframes
            session_dates: Optional[pd.Series]
                Recent session dates for update detection
            force_regenerate: bool
                Force regeneration of all bootstrap distributions
                
        Returns:
            Dict[str, Any]
                Summary of bootstrap generation results
        """
        print(f" Starting bootstrap generation for {len(strata_data)} strata")
        
        # Determine which strata need updates
        if force_regenerate:
            strata_to_update = {name: True for name in strata_data.keys()}
            print("  Force regeneration requested - updating all strata")
        else:
            strata_to_update = self.should_update_bootstrap(strata_data, session_dates)
            update_count = sum(strata_to_update.values())
            print(f"  Update check complete: {update_count}/{len(strata_data)} strata need updates")
        
        # Generate bootstrap for each strata that needs updating
        generation_results = {}
        summary = {
            'total_strata': len(strata_data),
            'strata_processed': 0,
            'bootstrap_enabled_count': 0,
            'bootstrap_skipped_count': 0,
            'bootstrap_failed_count': 0,
            'generation_timestamp': pd.Timestamp.now(),
            'warnings': []
        }
        
        for strata_name, strata_df in strata_data.items():
            if strata_to_update.get(strata_name, False):
                print(f"\n Processing strata: {strata_name}")
                result = self.generate_bootstrap_for_strata(
                    strata_name=strata_name,
                    strata_data=strata_df,
                    force_regenerate=force_regenerate
                )
                generation_results[strata_name] = result
                summary['strata_processed'] += 1
                
                if result.get('bootstrap_enabled', False):
                    summary['bootstrap_enabled_count'] += 1
                elif result.get('reason') and 'Small strata' in result.get('reason', ''):
                    summary['bootstrap_skipped_count'] += 1
                else:
                    summary['bootstrap_failed_count'] += 1
                    
                # Collect warnings
                if result.get('warnings'):
                    summary['warnings'].extend(result['warnings'])
            else:
                print(f"  ⏭️ Skipping {strata_name} (no update needed)")
        
        summary['generation_results'] = generation_results
        
        # Print final summary
        print(f"\n Bootstrap generation complete:")
        print(f"  Total strata: {summary['total_strata']}")
        print(f"  Processed: {summary['strata_processed']}")
        print(f"  Bootstrap enabled: {summary['bootstrap_enabled_count']}")
        print(f"  Skipped (small): {summary['bootstrap_skipped_count']}")  
        print(f"  Failed: {summary['bootstrap_failed_count']}")
        print(f"  Warnings: {len(summary['warnings'])}")
        
        return summary
    
    def get_bootstrap_distribution(self, strata_name: str, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached bootstrap distribution for a specific strata and feature
        
        Parameters:
            strata_name: str
                Name of the strata
            feature_name: str
                Name of the feature
                
        Returns:
            Optional[Dict[str, Any]]
                Bootstrap distribution or None if not available
        """
        cached_result = self._bootstrap_cache.get(strata_name)
        if cached_result and cached_result.get('bootstrap_enabled', False):
            return cached_result.get('bootstrap_distributions', {}).get(feature_name)
        return None
    
    def is_bootstrap_available(self, strata_name: str, feature_name: str) -> bool:
        """
        Check if bootstrap distribution is available for a strata and feature
        
        Parameters:
            strata_name: str
                Name of the strata
            feature_name: str
                Name of the feature
                
        Returns:
            bool
                True if bootstrap distribution is available and valid
        """
        bootstrap_dist = self.get_bootstrap_distribution(strata_name, feature_name)
        return bootstrap_dist is not None and bootstrap_dist.get('bootstrap_enabled', False)
    
    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get summary of cached bootstrap distributions
        
        Returns:
            Dict[str, Any]
                Summary of bootstrap cache status
        """
        summary = {
            'total_strata_cached': len(self._bootstrap_cache),
            'bootstrap_enabled_strata': 0,
            'total_features_cached': 0,
            'cache_size_mb': 0,
            'oldest_generation': None,
            'newest_generation': None,
            'strata_details': {}
        }
        
        generation_times = []
        
        for strata_name, cached_result in self._bootstrap_cache.items():
            strata_detail = {
                'bootstrap_enabled': cached_result.get('bootstrap_enabled', False),
                'feature_count': len(cached_result.get('bootstrap_distributions', {})),
                'generation_timestamp': cached_result.get('generation_timestamp'),
                'subject_count': cached_result.get('subject_count', 0),
                'strata_tier': cached_result.get('strata_tier', 'unknown')
            }
            
            if strata_detail['bootstrap_enabled']:
                summary['bootstrap_enabled_strata'] += 1
                summary['total_features_cached'] += strata_detail['feature_count']
                
                if strata_detail['generation_timestamp']:
                    generation_times.append(strata_detail['generation_timestamp'])
            
            summary['strata_details'][strata_name] = strata_detail
        
        # Calculate timestamp range
        if generation_times:
            summary['oldest_generation'] = min(generation_times)
            summary['newest_generation'] = max(generation_times)
        
        return summary
    
    def clear_cache(self, strata_names: Optional[List[str]] = None) -> None:
        """
        Clear bootstrap cache for specified strata or all strata
        
        Parameters:
            strata_names: Optional[List[str]]
                List of strata names to clear (None = clear all)
        """
        if strata_names is None:
            self._bootstrap_cache.clear()
            print("Bootstrap cache cleared for all strata")
        else:
            for strata_name in strata_names:
                if strata_name in self._bootstrap_cache:
                    del self._bootstrap_cache[strata_name]
                    print(f"Bootstrap cache cleared for strata: {strata_name}") 