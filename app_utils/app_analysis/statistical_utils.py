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