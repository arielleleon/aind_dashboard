import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler

class StatisticalUtils:
    """
    Centralized statistical utilities for robust analysis pipeline
    
    Provides:
    - Confidence interval calculations for percentiles
    - Bootstrap methods for uncertainty quantification
    - Outlier detection using multiple methods
    - Weighted statistical operations
    - Session-level bootstrap CI calculations
    - Bootstrap distribution generation and management
    - Coverage statistics and enhancement summaries
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
    
    @staticmethod
    def calculate_session_bootstrap_cis(session_data: pd.DataFrame, 
                                      bootstrap_manager=None,
                                      reference_processor=None,
                                      quantile_analyzer=None) -> pd.DataFrame:
        """
        Calculate bootstrap CIs for all session rolling averages during pipeline processing
        
        This pre-computes bootstrap CIs once and stores them in session data to avoid
        expensive real-time calculations during UI creation.
        
        Parameters:
            session_data: pd.DataFrame
                Session-level data with rolling averages and percentiles
            bootstrap_manager: BootstrapManager
                Bootstrap manager instance for statistical utilities
            reference_processor: ReferenceProcessor
                Reference processor for feature configuration
            quantile_analyzer: QuantileAnalyzer
                Quantile analyzer for reference data access
                
        Returns:
            pd.DataFrame
                Session data with bootstrap CI columns added
        """
        print("Pre-computing bootstrap CIs during pipeline...")
        
        if bootstrap_manager is None:
            print("Bootstrap manager not available - skipping bootstrap CI calculation")
            return session_data
        
        if reference_processor is None:
            print("Reference processor not available - skipping bootstrap CI calculation")
            return session_data
        
        result_df = session_data.copy()
        features = list(reference_processor.features_config.keys())
        
        # Counters for reporting
        total_ci_calculations = 0
        successful_ci_calculations = 0
        strata_processed = set()
        
        # Process by strata for efficiency (shared reference data)
        for strata, strata_sessions in result_df.groupby('strata'):
            strata_processed.add(strata)
            
            # Get reference data for this strata once
            if quantile_analyzer is not None:
                strata_data = quantile_analyzer.percentile_data.get(strata)
                
                if strata_data is None:
                    print(f"  No reference data for strata {strata} - skipping")
                    continue
                
                print(f"  Processing {len(strata_sessions)} sessions for strata {strata}")
                
                # Process each feature for this strata
                for feature in features:
                    reference_col = f"{feature}_processed"
                    rolling_avg_col = f"{feature}_processed_rolling_avg"
                    ci_lower_col = f"{feature}_bootstrap_ci_lower"
                    ci_upper_col = f"{feature}_bootstrap_ci_upper"
                    
                    # Get reference values for this feature/strata combination
                    if reference_col not in strata_data.columns:
                        continue
                    
                    reference_values = strata_data[reference_col].dropna().values
                    
                    if len(reference_values) < 5:
                        # Not enough reference data for bootstrap CI
                        continue
                    
                    # Calculate bootstrap CI for each session in this strata
                    for idx, row in strata_sessions.iterrows():
                        total_ci_calculations += 1
                        
                        rolling_avg_value = row.get(rolling_avg_col)
                        
                        if pd.isna(rolling_avg_value):
                            result_df.loc[idx, ci_lower_col] = np.nan
                            result_df.loc[idx, ci_upper_col] = np.nan
                            continue
                        
                        # Calculate bootstrap CI with reduced samples for efficiency
                        ci_lower, ci_upper = StatisticalUtils.calculate_bootstrap_raw_value_ci(
                            reference_data=reference_values,
                            target_value=rolling_avg_value,
                            confidence_level=0.95,
                            n_bootstrap=150,  # Reduced from 500 for better performance
                            random_state=42
                        )
                        
                        if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                            result_df.loc[idx, ci_lower_col] = ci_lower
                            result_df.loc[idx, ci_upper_col] = ci_upper
                            successful_ci_calculations += 1
                        else:
                            result_df.loc[idx, ci_lower_col] = np.nan
                            result_df.loc[idx, ci_upper_col] = np.nan
                
                # Calculate overall bootstrap CIs for this strata
                # Collect all processed feature values for overall calculation
                all_processed_values = []
                for feature in features:
                    reference_col = f"{feature}_processed"
                    if reference_col in strata_data.columns:
                        feature_values = strata_data[reference_col].dropna().values
                        all_processed_values.extend(feature_values)
                
                if len(all_processed_values) >= 10:
                    # Calculate overall bootstrap CI for each session
                    for idx, row in strata_sessions.iterrows():
                        overall_rolling_avg = row.get('session_overall_rolling_avg')
                        
                        if pd.isna(overall_rolling_avg):
                            result_df.loc[idx, 'session_overall_bootstrap_ci_lower'] = np.nan
                            result_df.loc[idx, 'session_overall_bootstrap_ci_upper'] = np.nan
                            continue
                        
                        ci_lower, ci_upper = StatisticalUtils.calculate_bootstrap_raw_value_ci(
                            reference_data=np.array(all_processed_values),
                            target_value=overall_rolling_avg,
                            confidence_level=0.95,
                            n_bootstrap=150,  # Reduced from 500
                            random_state=42
                        )
                        
                        if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                            result_df.loc[idx, 'session_overall_bootstrap_ci_lower'] = ci_lower
                            result_df.loc[idx, 'session_overall_bootstrap_ci_upper'] = ci_upper
                        else:
                            result_df.loc[idx, 'session_overall_bootstrap_ci_lower'] = np.nan
                            result_df.loc[idx, 'session_overall_bootstrap_ci_upper'] = np.nan
        
        # Report results
        print(f"Bootstrap CI pre-computation complete:")
        if total_ci_calculations > 0:
            success_rate = (successful_ci_calculations/total_ci_calculations*100)
            print(f"   - {successful_ci_calculations}/{total_ci_calculations} successful calculations ({success_rate:.1f}%)")
        else:
            print(f"   - No CI calculations attempted (no suitable reference data)")
        print(f"   - {len(strata_processed)} strata processed")
        print(f"   - Reduced bootstrap samples from 500 to 150 for better performance")
        
        return result_df
    
    @staticmethod
    def generate_bootstrap_distributions(bootstrap_manager=None,
                                       quantile_analyzer=None,
                                       reference_processor=None,
                                       cache_manager=None,
                                       force_regenerate: bool = False,
                                       strata_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate bootstrap distributions for eligible strata using the current data
        
        Parameters:
            bootstrap_manager: BootstrapManager
                Bootstrap manager instance
            quantile_analyzer: QuantileAnalyzer  
                Quantile analyzer for stratified data access
            reference_processor: ReferenceProcessor
                Reference processor for validation
            cache_manager: CacheManager
                Cache manager for session data access
            force_regenerate: bool
                Force regeneration of all bootstrap distributions
            strata_filter: Optional[List[str]]
                List of specific strata to process (None = all strata)
                
        Returns:
            Dict[str, Any]: Bootstrap generation results
        """
        if bootstrap_manager is None:
            print("Bootstrap manager not available for distribution generation")
            return {'error': 'Bootstrap manager not available', 'bootstrap_enabled_count': 0}
        
        if reference_processor is None:
            print("Reference processor not available for bootstrap distribution generation")
            return {'error': 'Reference processor not available', 'bootstrap_enabled_count': 0}
        
        # Get current session data for session date checking
        session_data = None
        if cache_manager and cache_manager.has('session_level_data'):
            session_data = cache_manager.get('session_level_data')
        
        session_dates = None
        if session_data is not None:
            session_dates = session_data['session_date']
        
        # Get stratified data directly from quantile analyzer
        if quantile_analyzer is not None:
            strata_data = getattr(quantile_analyzer, 'stratified_data', None)
            if strata_data is None:
                print("No stratified data found in quantile analyzer - bootstrap generation skipped")
                return {'error': 'No stratified data available for bootstrap generation', 'bootstrap_enabled_count': 0}
        else:
            print("Quantile analyzer not available - bootstrap generation skipped")
            return {'error': 'Quantile analyzer not available', 'bootstrap_enabled_count': 0}
        
        # Filter strata if requested
        if strata_filter is not None:
            strata_data = {k: v for k, v in strata_data.items() if k in strata_filter}
        
        # Generate bootstrap distributions
        print(f"Generating bootstrap distributions for {len(strata_data)} strata...")
        result = bootstrap_manager.generate_bootstrap_for_all_strata(
            strata_data=strata_data,
            session_dates=session_dates,
            force_regenerate=force_regenerate
        )
        
        return result
    
    @staticmethod
    def get_bootstrap_coverage_stats(cache_manager=None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get bootstrap coverage statistics for all strata
        
        Parameters:
            cache_manager: CacheManager
                Cache manager for accessing stored data
            use_cache: bool
                Whether to use cached coverage statistics
                
        Returns:
            Dict[str, Any]: Bootstrap coverage statistics by strata
        """
        if cache_manager is None:
            return {}
            
        # Check cache first
        if use_cache and cache_manager.has('bootstrap_coverage_stats'):
            return cache_manager.get('bootstrap_coverage_stats')
        
        # Check optimized storage
        if cache_manager.has('optimized_storage'):
            optimized_storage = cache_manager.get('optimized_storage')
            bootstrap_coverage = optimized_storage.get('bootstrap_coverage', {})
            if bootstrap_coverage:
                cache_manager.set('bootstrap_coverage_stats', bootstrap_coverage)
                return bootstrap_coverage
        
        # No coverage statistics available
        return {}
    
    @staticmethod
    def get_bootstrap_enabled_strata(cache_manager=None, use_cache: bool = True) -> set:
        """
        Get set of strata names that have bootstrap enhancement enabled
        
        Parameters:
            cache_manager: CacheManager
                Cache manager for accessing stored data
            use_cache: bool
                Whether to use cached strata set
                
        Returns:
            set: Set of strata names with bootstrap enhancement
        """
        if cache_manager is None:
            return set()
            
        # Check cache first  
        if use_cache and cache_manager.has('bootstrap_enabled_strata'):
            return cache_manager.get('bootstrap_enabled_strata')
        
        # Check optimized storage metadata
        if cache_manager.has('optimized_storage'):
            optimized_storage = cache_manager.get('optimized_storage')
            metadata = optimized_storage.get('metadata', {})
            strata_list = metadata.get('bootstrap_enabled_strata_list', [])
            if strata_list:
                strata_set = set(strata_list)
                cache_manager.set('bootstrap_enabled_strata', strata_set)
                return strata_set
        
        # No enabled strata found
        return set()
    
    @staticmethod
    def get_bootstrap_enhancement_summary(cache_manager=None,
                                        bootstrap_manager=None,
                                        reference_processor=None,
                                        use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of bootstrap enhancement coverage
        
        Parameters:
            cache_manager: CacheManager
                Cache manager for session data access
            bootstrap_manager: BootstrapManager
                Bootstrap manager instance for validation
            reference_processor: ReferenceProcessor
                Reference processor for feature configuration
            use_cache: bool
                Whether to use cached data if available
                
        Returns:
            Dict[str, Any]
                Bootstrap enhancement summary with statistics and subject details
        """
        # Get session-level data
        session_data = None
        if use_cache and cache_manager and cache_manager.has('session_level_data'):
            session_data = cache_manager.get('session_level_data')
        
        if session_data is None or session_data.empty:
            return {'error': 'No session data available'}
        
        summary = {
            'total_sessions': len(session_data),
            'total_subjects': session_data['subject_id'].nunique(),
            'total_strata': session_data['strata'].nunique(),
            'bootstrap_manager_available': bootstrap_manager is not None,
            'feature_enhancement': {},
            'overall_enhancement': {},
            'strata_breakdown': {},
            'subject_breakdown': {},
            'enhancement_statistics': {}
        }
        
        if not summary['bootstrap_manager_available']:
            summary['error'] = 'Bootstrap manager not available'
            return summary
        
        if reference_processor is None:
            summary['error'] = 'Reference processor not available'
            return summary
        
        # Get feature list
        feature_list = list(reference_processor.features_config.keys())
        
        # Analyze feature-specific bootstrap enhancement
        for feature in feature_list:
            bootstrap_col = f"{feature}_bootstrap_enhanced"
            percentile_col = f"{feature}_session_percentile"
            
            if bootstrap_col in session_data.columns and percentile_col in session_data.columns:
                # Count sessions with valid percentiles
                valid_percentiles = session_data[percentile_col].notna().sum()
                
                # Count bootstrap enhanced sessions
                bootstrap_enhanced = session_data[bootstrap_col].sum()
                
                # Calculate coverage rate
                coverage_rate = (bootstrap_enhanced / valid_percentiles * 100) if valid_percentiles > 0 else 0
                
                summary['feature_enhancement'][feature] = {
                    'valid_percentiles': int(valid_percentiles),
                    'bootstrap_enhanced': int(bootstrap_enhanced),
                    'coverage_rate': round(coverage_rate, 1),
                    'subjects_with_enhancement': int(session_data[session_data[bootstrap_col] == True]['subject_id'].nunique()),
                    'strata_with_enhancement': list(session_data[session_data[bootstrap_col] == True]['strata'].unique())
                }
        
        # Analyze overall percentile bootstrap enhancement
        overall_bootstrap_col = "session_overall_bootstrap_enhanced"
        overall_percentile_col = "session_overall_percentile"
        
        if overall_bootstrap_col in session_data.columns and overall_percentile_col in session_data.columns:
            valid_overall = session_data[overall_percentile_col].notna().sum()
            bootstrap_overall = session_data[overall_bootstrap_col].sum()
            overall_coverage = (bootstrap_overall / valid_overall * 100) if valid_overall > 0 else 0
            
            summary['overall_enhancement'] = {
                'valid_percentiles': int(valid_overall),
                'bootstrap_enhanced': int(bootstrap_overall),
                'coverage_rate': round(overall_coverage, 1),
                'subjects_with_enhancement': int(session_data[session_data[overall_bootstrap_col] == True]['subject_id'].nunique()),
                'strata_with_enhancement': list(session_data[session_data[overall_bootstrap_col] == True]['strata'].unique())
            }
        
        # Strata-level breakdown
        for strata, strata_sessions in session_data.groupby('strata'):
            strata_summary = {
                'total_sessions': len(strata_sessions),
                'subjects': strata_sessions['subject_id'].nunique(),
                'features_with_bootstrap': {},
                'overall_bootstrap_sessions': 0,
                'any_bootstrap_sessions': 0
            }
            
            # Check each feature for this strata
            for feature in feature_list:
                bootstrap_col = f"{feature}_bootstrap_enhanced"
                if bootstrap_col in strata_sessions.columns:
                    bootstrap_count = strata_sessions[bootstrap_col].sum()
                    strata_summary['features_with_bootstrap'][feature] = int(bootstrap_count)
            
            # Overall bootstrap count for this strata
            if overall_bootstrap_col in strata_sessions.columns:
                strata_summary['overall_bootstrap_sessions'] = int(strata_sessions[overall_bootstrap_col].sum())
            
            # Any bootstrap enhancement for this strata
            bootstrap_cols = [f"{feature}_bootstrap_enhanced" for feature in feature_list]
            bootstrap_cols = [col for col in bootstrap_cols if col in strata_sessions.columns]
            
            if bootstrap_cols:
                any_bootstrap = strata_sessions[bootstrap_cols].any(axis=1).sum()
                strata_summary['any_bootstrap_sessions'] = int(any_bootstrap)
            
            summary['strata_breakdown'][strata] = strata_summary
        
        # Subject-level breakdown (most recent session only)
        most_recent = session_data.sort_values('session_date').groupby('subject_id').last().reset_index()
        
        for _, row in most_recent.iterrows():
            subject_id = row['subject_id']
            subject_summary = {
                'strata': row['strata'],
                'session_date': row['session_date'],
                'features_with_bootstrap': [],
                'overall_bootstrap_enhanced': False
            }
            
            # Check each feature
            for feature in feature_list:
                bootstrap_col = f"{feature}_bootstrap_enhanced"
                if bootstrap_col in row and row[bootstrap_col]:
                    subject_summary['features_with_bootstrap'].append(feature)
            
            # Check overall
            if overall_bootstrap_col in row:
                subject_summary['overall_bootstrap_enhanced'] = bool(row[overall_bootstrap_col])
            
            summary['subject_breakdown'][subject_id] = subject_summary
        
        # Enhancement statistics
        total_possible_enhancements = len(session_data) * len(feature_list)
        total_actual_enhancements = sum(
            session_data[f"{feature}_bootstrap_enhanced"].sum() 
            for feature in feature_list 
            if f"{feature}_bootstrap_enhanced" in session_data.columns
        )
        
        summary['enhancement_statistics'] = {
            'total_possible_feature_enhancements': total_possible_enhancements,
            'total_actual_feature_enhancements': int(total_actual_enhancements),
            'overall_enhancement_rate': round((total_actual_enhancements / total_possible_enhancements * 100) if total_possible_enhancements > 0 else 0, 1),
            'strata_with_any_enhancement': len([s for s in summary['strata_breakdown'].values() if s['any_bootstrap_sessions'] > 0]),
            'subjects_with_any_enhancement': len([s for s in summary['subject_breakdown'].values() if s['features_with_bootstrap'] or s['overall_bootstrap_enhanced']])
        }
        
        return summary 