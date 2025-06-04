import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
import warnings

class StatisticalUtils:
    """
    Centralized statistical utilities for robust analysis pipeline
    
    Provides:
    - Confidence interval calculations for percentiles
    - Bootstrap methods for uncertainty quantification
    - Outlier detection using multiple methods
    - Weighted statistical operations
    """
    
    @staticmethod
    def calculate_percentile_confidence_interval(
        values: np.ndarray, 
        percentile: float, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a percentile using Wilson score interval
        
        This method is robust for percentile confidence intervals and works well
        with the sample sizes typically seen in behavioral analysis.
        
        Parameters:
            values: np.ndarray
                Array of values from reference distribution
            percentile: float
                The percentile value (0-100) for which to calculate CI
            confidence_level: float
                Confidence level (default: 0.95 for 95% CI)
                
        Returns:
            Tuple[float, float]
                (lower_bound, upper_bound) of confidence interval
        """
        if len(values) < 3:
            # Not enough data for meaningful CI
            return (np.nan, np.nan)
        
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        n = len(clean_values)
        
        if n < 3:
            return (np.nan, np.nan)
        
        # Convert percentile to proportion
        p = percentile / 100.0
        
        # Calculate z-score for confidence level
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha/2)
        
        # Wilson score interval for percentile confidence
        # This accounts for the binomial nature of percentile estimation
        denominator = 1 + (z**2 / n)
        
        center = (p + (z**2)/(2*n)) / denominator
        margin = (z * np.sqrt((p*(1-p) + (z**2)/(4*n)) / n)) / denominator
        
        lower_bound = max(0, center - margin) * 100
        upper_bound = min(100, center + margin) * 100
        
        return (lower_bound, upper_bound)
    
    @staticmethod
    def calculate_bootstrap_ci(
        data: np.ndarray, 
        statistic_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap resampling
        
        Parameters:
            data: np.ndarray
                Input data for bootstrap sampling
            statistic_func: callable
                Function to calculate statistic (e.g., np.mean, np.median)
            n_bootstrap: int
                Number of bootstrap samples
            confidence_level: float
                Confidence level (default: 0.95)
            random_state: Optional[int]
                Random seed for reproducibility
                
        Returns:
            Tuple[float, float]
                (lower_bound, upper_bound) of confidence interval
        """
        if len(data) < 3:
            return (np.nan, np.nan)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            return (np.nan, np.nan)
        
        # Perform bootstrap sampling
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
            
            try:
                stat = statistic_func(bootstrap_sample)
                if not np.isnan(stat):
                    bootstrap_stats.append(stat)
            except:
                continue
        
        if len(bootstrap_stats) < 10:  # Need minimum successful bootstrap samples
            return (np.nan, np.nan)
        
        # Calculate percentiles for CI
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    @staticmethod
    def detect_outliers_iqr(
        data: np.ndarray, 
        factor: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Parameters:
            data: np.ndarray
                Input data for outlier detection
            factor: float
                IQR multiplier factor (default: 1.5 for standard outlier detection)
                
        Returns:
            Tuple[np.ndarray, np.ndarray]
                (outlier_mask, data_weights) where outlier_mask is boolean array
                and data_weights are suggested weights (0.5 for outliers, 1.0 for normal)
        """
        if len(data) < 4:  # Need at least 4 points for IQR calculation
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        
        # Remove NaN values for quartile calculation
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 4:
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        
        # Calculate quartiles
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        
        # Calculate outlier bounds
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Create outlier mask for all data (including NaN)
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        # Create weights array (0.5 for outliers, 1.0 for normal, 0.0 for NaN)
        weights = np.ones(len(data))
        weights[outlier_mask] = 0.5  # Reduce weight for outliers
        weights[np.isnan(data)] = 0.0  # Zero weight for NaN values
        
        return outlier_mask, weights
    
    @staticmethod
    def detect_outliers_modified_zscore(
        data: np.ndarray, 
        threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Modified Z-score method (using median absolute deviation)
        More robust than standard Z-score for non-normal distributions
        
        Parameters:
            data: np.ndarray
                Input data for outlier detection
            threshold: float
                Modified Z-score threshold (default: 3.5)
                
        Returns:
            Tuple[np.ndarray, np.ndarray]
                (outlier_mask, data_weights)
        """
        if len(data) < 3:
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        
        # Remove NaN values for calculation
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        
        # Calculate median and median absolute deviation
        median = np.median(clean_data)
        mad = np.median(np.abs(clean_data - median))
        
        if mad == 0:
            # If MAD is 0, all values are the same, no outliers
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        
        # Calculate modified Z-scores for all data
        modified_z_scores = 0.6745 * (data - median) / mad
        
        # Create outlier mask
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        # Create weights array
        weights = np.ones(len(data))
        weights[outlier_mask] = 0.5  # Reduce weight for outliers
        weights[np.isnan(data)] = 0.0  # Zero weight for NaN values
        
        return outlier_mask, weights
    
    @staticmethod
    def calculate_weighted_percentile_rank(
        reference_values: np.ndarray, 
        reference_weights: np.ndarray,
        target_value: float
    ) -> float:
        """
        Calculate percentile rank using weighted observations
        
        Parameters:
            reference_values: np.ndarray
                Reference distribution values
            reference_weights: np.ndarray
                Weights for each reference value
            target_value: float
                Value for which to calculate percentile rank
                
        Returns:
            float
                Percentile rank (0-100)
        """
        if len(reference_values) == 0 or np.isnan(target_value):
            return np.nan
        
        # Remove NaN values and corresponding weights
        valid_mask = ~np.isnan(reference_values)
        clean_values = reference_values[valid_mask]
        clean_weights = reference_weights[valid_mask]
        
        if len(clean_values) == 0:
            return np.nan
        
        # Add target value with weight 1.0
        all_values = np.append(clean_values, target_value)
        all_weights = np.append(clean_weights, 1.0)
        
        # Sort by values
        sort_indices = np.argsort(all_values)
        sorted_values = all_values[sort_indices]
        sorted_weights = all_weights[sort_indices]
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        # Find the position of target value
        target_index = np.where(sorted_values == target_value)[0][-1]  # Get last occurrence
        
        # Calculate percentile rank
        if target_index == 0:
            percentile_rank = 0.0
        else:
            # Weight below target value
            weight_below = cumulative_weights[target_index - 1]
            # Weight of target value itself
            weight_at = sorted_weights[target_index]
            
            # Use midpoint method for tied values
            percentile_rank = (weight_below + weight_at / 2.0) / total_weight * 100
        
        return percentile_rank
    
    @staticmethod
    def validate_ci_coverage(
        true_percentiles: np.ndarray,
        estimated_percentiles: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Validate confidence interval coverage for testing purposes
        
        Parameters:
            true_percentiles: np.ndarray
                True percentile values (for validation)
            estimated_percentiles: np.ndarray
                Estimated percentile values
            ci_lower: np.ndarray
                Lower CI bounds
            ci_upper: np.ndarray
                Upper CI bounds
            confidence_level: float
                Expected confidence level
                
        Returns:
            Dict[str, float]
                Coverage statistics
        """
        # Remove NaN values
        valid_mask = (
            ~np.isnan(true_percentiles) & 
            ~np.isnan(estimated_percentiles) & 
            ~np.isnan(ci_lower) & 
            ~np.isnan(ci_upper)
        )
        
        if np.sum(valid_mask) == 0:
            return {"coverage_rate": np.nan, "avg_ci_width": np.nan, "n_valid": 0}
        
        true_vals = true_percentiles[valid_mask]
        ci_low = ci_lower[valid_mask]
        ci_high = ci_upper[valid_mask]
        
        # Check if true values fall within CI bounds
        within_ci = (true_vals >= ci_low) & (true_vals <= ci_high)
        coverage_rate = np.mean(within_ci)
        
        # Calculate average CI width
        avg_ci_width = np.mean(ci_high - ci_low)
        
        return {
            "coverage_rate": coverage_rate,
            "expected_coverage": confidence_level,
            "coverage_difference": coverage_rate - confidence_level,
            "avg_ci_width": avg_ci_width,
            "n_valid": np.sum(valid_mask)
        }

    # PHASE 3: Bootstrap Enhancement Methods
    
    @staticmethod
    def generate_bootstrap_reference_distribution(
        reference_data: np.ndarray,
        reference_weights: Optional[np.ndarray] = None,
        n_bootstrap: int = 2000,
        percentile_grid: Optional[np.ndarray] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate bootstrap reference distribution using pre-computed percentile grids
        
        This creates a more robust reference distribution for percentile calculations
        by generating bootstrap samples and computing percentile statistics.
        
        Parameters:
            reference_data: np.ndarray
                Original reference distribution values
            reference_weights: Optional[np.ndarray]
                Weights for reference data (from outlier detection)
            n_bootstrap: int
                Number of bootstrap samples to generate
            percentile_grid: Optional[np.ndarray]
                Percentiles to compute (default: 1, 2, 3, ..., 99)
            random_state: Optional[int]
                Random seed for reproducibility
                
        Returns:
            Dict[str, Any]
                Bootstrap reference distribution with percentile grid and metadata
        """
        if len(reference_data) < 3:
            return {
                'percentile_grid': np.array([]),
                'percentile_values': np.array([]),
                'bootstrap_enabled': False,
                'error': 'Insufficient reference data'
            }
        
        # Set default percentile grid (1st to 99th percentiles)
        if percentile_grid is None:
            percentile_grid = np.arange(1, 100)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Clean reference data
        valid_mask = ~np.isnan(reference_data)
        clean_data = reference_data[valid_mask]
        
        if reference_weights is not None:
            clean_weights = reference_weights[valid_mask]
        else:
            clean_weights = np.ones(len(clean_data))
        
        if len(clean_data) < 3:
            return {
                'percentile_grid': np.array([]),
                'percentile_values': np.array([]),
                'bootstrap_enabled': False,
                'error': 'Insufficient clean reference data'
            }
        
        # Generate bootstrap samples and compute percentile grid
        bootstrap_percentiles = []
        successful_samples = 0
        
        for _ in range(n_bootstrap):
            try:
                # Weighted bootstrap sampling
                if np.any(clean_weights != 1.0):
                    # Use weighted sampling
                    normalized_weights = clean_weights / np.sum(clean_weights)
                    bootstrap_indices = np.random.choice(
                        len(clean_data), 
                        size=len(clean_data), 
                        replace=True, 
                        p=normalized_weights
                    )
                    bootstrap_sample = clean_data[bootstrap_indices]
                else:
                    # Standard bootstrap sampling
                    bootstrap_sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
                
                # Compute percentiles for this bootstrap sample
                sample_percentiles = np.percentile(bootstrap_sample, percentile_grid)
                bootstrap_percentiles.append(sample_percentiles)
                successful_samples += 1
                
            except Exception as e:
                # Skip failed bootstrap samples
                continue
        
        if successful_samples < max(10, n_bootstrap * 0.1):  # Need at least 10% successful samples
            return {
                'percentile_grid': np.array([]),
                'percentile_values': np.array([]),
                'bootstrap_enabled': False,
                'error': f'Too few successful bootstrap samples: {successful_samples}/{n_bootstrap}'
            }
        
        # Convert to numpy array and compute statistics
        bootstrap_percentiles = np.array(bootstrap_percentiles)
        
        # Compute mean percentile values across all bootstrap samples
        mean_percentiles = np.mean(bootstrap_percentiles, axis=0)
        
        # Compute confidence intervals for each percentile
        percentile_cis = {}
        for i, p in enumerate(percentile_grid):
            ci_lower = np.percentile(bootstrap_percentiles[:, i], 2.5)
            ci_upper = np.percentile(bootstrap_percentiles[:, i], 97.5)
            percentile_cis[int(p)] = (ci_lower, ci_upper)
        
        return {
            'percentile_grid': percentile_grid,
            'percentile_values': mean_percentiles,
            'percentile_cis': percentile_cis,
            'bootstrap_samples': successful_samples,
            'bootstrap_enabled': True,
            'original_data_size': len(clean_data),
            'generation_timestamp': pd.Timestamp.now(),
            'random_state': random_state
        }
    
    @staticmethod
    def calculate_bootstrap_raw_value_ci(
        reference_data: np.ndarray,
        target_value: float,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a raw rolling average value
        
        This calculates uncertainty in the actual measurement value itself,
        not the percentile ranking. Uses bootstrap resampling of reference
        distribution to estimate CI bounds for the target value.
        
        Parameters:
            reference_data: np.ndarray
                Reference distribution of raw values (same feature, same strata)
            target_value: float
                The rolling average value for which to calculate CI
            confidence_level: float
                Confidence level (default: 0.95)
            n_bootstrap: int
                Number of bootstrap samples
            random_state: Optional[int]
                Random seed for reproducibility
                
        Returns:
            Tuple[float, float]
                (lower_bound, upper_bound) for the raw value uncertainty
        """
        if len(reference_data) < 5 or pd.isna(target_value):
            return (np.nan, np.nan)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Clean reference data
        clean_reference = reference_data[~np.isnan(reference_data)]
        
        if len(clean_reference) < 5:
            return (np.nan, np.nan)
        
        # Calculate the sample standard deviation as a proxy for uncertainty
        sample_std = np.std(clean_reference, ddof=1)
        sample_mean = np.mean(clean_reference)
        
        # For bootstrap CI of raw values, we estimate the uncertainty
        # in our target value based on the reference distribution variance
        
        # Bootstrap the reference distribution to estimate sampling variability
        bootstrap_means = []
        bootstrap_stds = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample from reference
            bootstrap_sample = np.random.choice(clean_reference, size=len(clean_reference), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))
        
        # Estimate the standard error of the measurement
        mean_of_bootstrap_stds = np.mean(bootstrap_stds)
        
        # Calculate CI bounds for the target value using the estimated uncertainty
        # This represents "how uncertain are we about this measurement value"
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Use the bootstrap-estimated standard error
        margin_of_error = z_score * mean_of_bootstrap_stds / np.sqrt(len(clean_reference))
        
        lower_bound = target_value - margin_of_error
        upper_bound = target_value + margin_of_error
        
        return (lower_bound, upper_bound)
    
    @staticmethod
    def validate_percentile_monotonicity(
        percentile_values: np.ndarray,
        percentile_grid: np.ndarray,
        key_percentiles: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Validate percentile monotonicity (25th < 50th < 75th percentiles)
        
        Ensures that percentile distributions maintain proper ordering,
        which is critical for statistical validity.
        
        Parameters:
            percentile_values: np.ndarray
                Computed percentile values
            percentile_grid: np.ndarray
                Corresponding percentile ranks
            key_percentiles: Optional[List[float]]
                Key percentiles to check (default: [25, 50, 75])
                
        Returns:
            Dict[str, Any]
                Validation results with pass/fail status and details
        """
        if key_percentiles is None:
            key_percentiles = [25.0, 50.0, 75.0]
        
        validation_result = {
            'monotonicity_valid': True,
            'violations': [],
            'key_percentile_values': {},
            'overall_monotonic': True
        }
        
        try:
            # Check overall monotonicity
            if len(percentile_values) > 1:
                is_monotonic = np.all(np.diff(percentile_values) >= 0)
                validation_result['overall_monotonic'] = is_monotonic
                
                if not is_monotonic:
                    # Find specific violations
                    violations_indices = np.where(np.diff(percentile_values) < 0)[0]
                    for idx in violations_indices:
                        validation_result['violations'].append({
                            'percentile_1': float(percentile_grid[idx]),
                            'value_1': float(percentile_values[idx]),
                            'percentile_2': float(percentile_grid[idx + 1]),
                            'value_2': float(percentile_values[idx + 1]),
                            'violation_type': 'general_monotonicity'
                        })
            
            # Check key percentiles specifically
            key_values = []
            for p in key_percentiles:
                # Find closest percentile in grid
                if len(percentile_grid) > 0:
                    closest_idx = np.argmin(np.abs(percentile_grid - p))
                    key_value = percentile_values[closest_idx]
                    key_values.append(key_value)
                    validation_result['key_percentile_values'][f'p{int(p)}'] = float(key_value)
            
            # Check key percentile ordering
            if len(key_values) >= 2:
                for i in range(len(key_values) - 1):
                    if key_values[i] >= key_values[i + 1]:
                        validation_result['monotonicity_valid'] = False
                        validation_result['violations'].append({
                            'percentile_1': key_percentiles[i],
                            'value_1': float(key_values[i]),
                            'percentile_2': key_percentiles[i + 1],
                            'value_2': float(key_values[i + 1]),
                            'violation_type': 'key_percentile_ordering'
                        })
        
        except Exception as e:
            validation_result.update({
                'monotonicity_valid': False,
                'overall_monotonic': False,
                'error': str(e)
            })
        
        return validation_result
    
    @staticmethod
    def compare_bootstrap_vs_standard(
        reference_data: np.ndarray,
        bootstrap_distribution: Dict[str, Any],
        test_percentiles: Optional[List[float]] = None,
        reference_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare bootstrap-enhanced vs standard percentile calculations
        
        Provides quality metrics to assess the value of bootstrap enhancement
        and detect potential issues with bootstrap distributions.
        
        Parameters:
            reference_data: np.ndarray
                Original reference distribution values
            bootstrap_distribution: Dict[str, Any]
                Bootstrap reference distribution
            test_percentiles: Optional[List[float]]
                Percentiles to compare (default: [10, 25, 50, 75, 90])
            reference_weights: Optional[np.ndarray]
                Weights for reference data
                
        Returns:
            Dict[str, Any]
                Comparison statistics and quality metrics
        """
        if test_percentiles is None:
            test_percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]
        
        comparison_result = {
            'bootstrap_available': bootstrap_distribution.get('bootstrap_enabled', False),
            'percentile_comparisons': {},
            'quality_metrics': {},
            'recommendation': 'standard'  # Default to standard method
        }
        
        if not comparison_result['bootstrap_available']:
            comparison_result['error'] = 'Bootstrap distribution not available'
            return comparison_result
        
        try:
            # Clean reference data
            valid_mask = ~np.isnan(reference_data)
            clean_data = reference_data[valid_mask]
            
            if reference_weights is not None:
                clean_weights = reference_weights[valid_mask]
            else:
                clean_weights = np.ones(len(clean_data))
            
            # Calculate standard percentiles
            if np.any(clean_weights != 1.0):
                # Weighted percentiles (more complex calculation)
                standard_percentiles = {}
                for p in test_percentiles:
                    # Simple weighted percentile approximation
                    sorted_indices = np.argsort(clean_data)
                    sorted_data = clean_data[sorted_indices]
                    sorted_weights = clean_weights[sorted_indices]
                    cumsum_weights = np.cumsum(sorted_weights)
                    total_weight = cumsum_weights[-1]
                    target_weight = (p / 100.0) * total_weight
                    percentile_idx = np.searchsorted(cumsum_weights, target_weight)
                    percentile_idx = min(percentile_idx, len(sorted_data) - 1)
                    standard_percentiles[p] = sorted_data[percentile_idx]
            else:
                # Standard percentiles
                standard_percentiles = {p: np.percentile(clean_data, p) for p in test_percentiles}
            
            # Get bootstrap percentiles
            bootstrap_grid = bootstrap_distribution.get('percentile_grid', np.array([]))
            bootstrap_values = bootstrap_distribution.get('percentile_values', np.array([]))
            
            # Compare percentiles
            differences = []
            relative_differences = []
            
            for p in test_percentiles:
                standard_val = standard_percentiles[p]
                
                # Find closest bootstrap percentile
                if len(bootstrap_grid) > 0:
                    closest_idx = np.argmin(np.abs(bootstrap_grid - p))
                    bootstrap_val = bootstrap_values[closest_idx]
                    
                    diff = abs(bootstrap_val - standard_val)
                    rel_diff = diff / abs(standard_val) if standard_val != 0 else 0
                    
                    differences.append(diff)
                    relative_differences.append(rel_diff)
                    
                    comparison_result['percentile_comparisons'][f'p{int(p)}'] = {
                        'standard': float(standard_val),
                        'bootstrap': float(bootstrap_val),
                        'absolute_diff': float(diff),
                        'relative_diff': float(rel_diff)
                    }
            
            # Calculate quality metrics
            if differences:
                comparison_result['quality_metrics'] = {
                    'mean_absolute_difference': float(np.mean(differences)),
                    'max_absolute_difference': float(np.max(differences)),
                    'mean_relative_difference': float(np.mean(relative_differences)),
                    'max_relative_difference': float(np.max(relative_differences)),
                    'bootstrap_samples': bootstrap_distribution.get('bootstrap_samples', 0),
                    'original_data_size': len(clean_data)
                }
                
                # Make recommendation based on quality metrics
                max_rel_diff = np.max(relative_differences)
                mean_rel_diff = np.mean(relative_differences)
                
                if max_rel_diff < 0.05 and mean_rel_diff < 0.02:
                    # Good agreement, bootstrap provides value
                    comparison_result['recommendation'] = 'bootstrap'
                elif max_rel_diff < 0.10 and mean_rel_diff < 0.05:
                    # Reasonable agreement, bootstrap acceptable
                    comparison_result['recommendation'] = 'bootstrap_with_caution'
                else:
                    # Poor agreement, stick with standard
                    comparison_result['recommendation'] = 'standard'
                    comparison_result['warning'] = f'High disagreement: max_rel_diff={max_rel_diff:.3f}'
        
        except Exception as e:
            comparison_result.update({
                'error': str(e),
                'recommendation': 'standard'
            })
        
        return comparison_result 