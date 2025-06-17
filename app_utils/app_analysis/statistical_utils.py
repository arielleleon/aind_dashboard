"""
Statistical utilities and analysis functions for AIND Dashboard

This module provides statistical processing and confidence interval calculations
for session data using Wilson Score intervals for robust uncertainty quantification.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from app_utils.simple_logger import get_logger

logger = get_logger("statistical_utils")


class StatisticalUtils:
    """
    Centralized statistical utilities for robust analysis pipeline

    Provides:
    - Wilson Score confidence interval calculations for percentiles
    - Outlier detection using multiple methods
    - Weighted statistical operations
    - Percentile data validation and processing for heatmaps
    """

    @staticmethod
    def calculate_percentile_confidence_interval(
        values: np.ndarray, percentile: float, confidence_level: float = 0.95
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
        z = stats.norm.ppf(1 - alpha / 2)

        # Wilson score interval for percentile confidence
        # This accounts for the binomial nature of percentile estimation
        denominator = 1 + (z**2 / n)

        center = (p + (z**2) / (2 * n)) / denominator
        margin = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denominator

        lower_bound = max(0, center - margin) * 100
        upper_bound = min(100, center + margin) * 100

        return (lower_bound, upper_bound)

    @staticmethod
    def detect_outliers_iqr(
        data: np.ndarray, factor: float = 3.0
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
        data: np.ndarray, threshold: float = 3.5
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
        reference_values: np.ndarray, reference_weights: np.ndarray, target_value: float
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
        target_index = np.where(sorted_values == target_value)[0][
            -1
        ]  # Get last occurrence

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
    def validate_percentile_data(
        percentiles: List[Union[float, int]], invalid_marker: float = -1
    ) -> List[float]:
        """
        Validate and clean percentile data, converting invalid markers to NaN

        This function processes raw percentile data by converting invalid markers
        (typically -1) to NaN values for proper statistical handling.

        Parameters:
            percentiles: List[Union[float, int]]
                Raw percentile data that may contain invalid markers
            invalid_marker: float
                Value used to mark invalid/missing data (default: -1)

        Returns:
            List[float]: Cleaned percentile data with NaN for invalid values
        """
        return [p if p != invalid_marker else np.nan for p in percentiles]

    @staticmethod
    def process_heatmap_matrix_data(
        time_series_data: Dict[str, Any], features_config: Dict[str, bool]
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Process time series data into heatmap matrix format with feature names

        This function extracts percentile data for configured features and prepares
        it for heatmap visualization, including data validation and display name formatting.

        Parameters:
            time_series_data: Dict[str, Any]
                Time series data containing percentile information for features
            features_config: Dict[str, bool]
                Configuration dict mapping feature names to whether they should be included

        Returns:
            Tuple[List[List[float]], List[str]]:
                - Matrix data for heatmap (list of feature percentile arrays)
                - Feature display names for heatmap labels
        """
        heatmap_data = []
        feature_names = []

        # Process each configured feature
        for feature in features_config.keys():
            percentile_key = f"{feature}_percentiles"

            if percentile_key in time_series_data:
                percentiles = time_series_data[percentile_key]

                # Validate and clean percentile data
                valid_percentiles = StatisticalUtils.validate_percentile_data(
                    percentiles
                )

                # Only include features with at least some valid data
                if any(not np.isnan(p) for p in valid_percentiles):
                    heatmap_data.append(valid_percentiles)

                    # Format display name for visualization
                    display_name = StatisticalUtils.format_feature_display_name(feature)
                    feature_names.append(display_name)

        # Add overall percentile row if available
        if "overall_percentiles" in time_series_data:
            overall_percentiles = time_series_data["overall_percentiles"]
            valid_overall = StatisticalUtils.validate_percentile_data(
                overall_percentiles
            )

            if any(not np.isnan(p) for p in valid_overall):
                heatmap_data.append(valid_overall)
                feature_names.append("Overall Percentile")

        return heatmap_data, feature_names

    @staticmethod
    def format_feature_display_name(feature: str) -> str:
        """
        Convert feature key to human-readable display name

        Transforms technical feature names into properly formatted display names
        for visualization components.

        Parameters:
            feature: str
                Technical feature name (e.g., "finished_trials", "abs(bias_naive)")

        Returns:
            str: Human-readable display name (e.g., "Finished Trials", "|Bias Naive|")
        """
        return feature.replace("_", " ").replace("abs(", "|").replace(")", "|").title()

    @staticmethod
    def calculate_session_highlighting_coordinates(
        sessions: List[int], highlighted_session: int
    ) -> Optional[int]:
        """
        Calculate coordinates for highlighting a specific session in visualizations

        Parameters:
            sessions: List[int]
                List of all session numbers in display order
            highlighted_session: int
                Session number to highlight

        Returns:
            Optional[int]: Index of the highlighted session, or None if not found
        """
        try:
            return sessions.index(highlighted_session)
        except ValueError:
            logger.info(f"Session {highlighted_session} not found in session list")
            return None
