from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .statistical_utils import StatisticalUtils


class OverallPercentileCalculator:
    """
    Dedicated class for calculating overall percentiles across features.
    This centralizes percentile calculation logic used across the application.
    """

    def __init__(self):
        """Initialize the percentile calculator with caching support"""
        self._cache = {
            "overall_percentiles": None,
            "session_overall_percentiles": None,
            "last_update_time": None,
        }
        self.statistical_utils = StatisticalUtils()

    def calculate_session_overall_percentile(
        self,
        session_data: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate overall percentile performance for each session via a simple average across feature percentiles.
        Enhanced with confidence interval calculations using error propagation.

        Parameters:
            session_data: pd.DataFrame
                DataFrame containing session-level data with percentile values and CI bounds
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features

        Returns:
            pd.DataFrame
                DataFrame with added session_overall_percentile and CI columns
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()

        # Get all session-level percentile columns
        percentile_cols = [
            col
            for col in result_df.columns
            if col.endswith("_session_percentile")
            and not col.endswith("_ci_lower")
            and not col.endswith("_ci_upper")
        ]

        if not percentile_cols:
            print("No session-level percentile columns found in data")
            return result_df

        print(
            f"Calculating overall percentiles with CI for {len(percentile_cols)} features"
        )

        # Process each session
        for idx, row in result_df.iterrows():
            # Extract all percentile values that are not NaN
            percentile_values = []
            ci_lower_values = []
            ci_upper_values = []
            feature_names = []
            weights = []

            for col in percentile_cols:
                if not pd.isna(row[col]):
                    feature_name = col.replace("_session_percentile", "")

                    # Get CI columns for this feature
                    ci_lower_col = f"{feature_name}_session_percentile_ci_lower"
                    ci_upper_col = f"{feature_name}_session_percentile_ci_upper"

                    # Only include if we have valid CI data
                    if (
                        ci_lower_col in row
                        and ci_upper_col in row
                        and not pd.isna(row[ci_lower_col])
                        and not pd.isna(row[ci_upper_col])
                    ):

                        feature_names.append(feature_name)
                        percentile_values.append(row[col])
                        ci_lower_values.append(row[ci_lower_col])
                        ci_upper_values.append(row[ci_upper_col])

                        # Get weight for this feature
                        weight = (
                            feature_weights.get(feature_name, 1.0)
                            if feature_weights
                            else 1.0
                        )
                        weights.append(weight)

            # Skip if no valid percentile values
            if not percentile_values:
                result_df.loc[idx, "session_overall_percentile"] = np.nan
                result_df.loc[idx, "session_overall_percentile_ci_lower"] = np.nan
                result_df.loc[idx, "session_overall_percentile_ci_upper"] = np.nan
                continue

            # Convert to numpy arrays for easier calculation
            percentiles = np.array(percentile_values)
            ci_lower = np.array(ci_lower_values)
            ci_upper = np.array(ci_upper_values)
            weights_array = np.array(weights)

            # Normalize weights
            total_weight = np.sum(weights_array)
            normalized_weights = (
                weights_array / total_weight
                if total_weight > 0
                else np.ones_like(weights_array) / len(weights_array)
            )

            # Calculate weighted average for overall percentile
            overall_percentile = np.sum(percentiles * normalized_weights)

            # Calculate confidence interval for overall percentile using error propagation
            # For weighted average: CI_overall = sqrt(sum(w_i^2 * CI_width_i^2)) / 2
            # where CI_width_i = (upper_i - lower_i)

            ci_widths = ci_upper - ci_lower
            # Calculate error propagation (assuming independence of errors)
            variance_sum = np.sum((normalized_weights * ci_widths / 2) ** 2)
            overall_ci_half_width = np.sqrt(variance_sum)

            # Calculate overall CI bounds
            overall_ci_lower = max(0, overall_percentile - overall_ci_half_width)
            overall_ci_upper = min(100, overall_percentile + overall_ci_half_width)

            # Store in result dataframe
            result_df.loc[idx, "session_overall_percentile"] = overall_percentile
            result_df.loc[idx, "session_overall_percentile_ci_lower"] = overall_ci_lower
            result_df.loc[idx, "session_overall_percentile_ci_upper"] = overall_ci_upper

        # Cache the result
        self._cache["session_overall_percentiles"] = result_df

        print(f"Calculated overall percentiles with CI for {len(result_df)} sessions")
        return result_df

    def clear_cache(self) -> None:
        """Clear the cache"""
        self._cache["overall_percentiles"] = None
        self._cache["session_overall_percentiles"] = None
        self._cache["last_update_time"] = None

    def calculate_session_overall_rolling_average(
        self,
        session_data: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate overall rolling average for each session by averaging the feature rolling averages.
        This provides the "rolling average value" for overall percentile hover information.

        Parameters:
            session_data: pd.DataFrame
                DataFrame containing session-level data with rolling average values
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features

        Returns:
            pd.DataFrame
                DataFrame with added session_overall_rolling_avg column
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()

        # Get all session-level rolling average columns
        rolling_avg_cols = [
            col for col in result_df.columns if col.endswith("_processed_rolling_avg")
        ]

        if not rolling_avg_cols:
            print("No session-level rolling average columns found in data")
            return result_df

        # Process each session
        for idx, row in result_df.iterrows():
            # Extract all rolling average values that are not NaN
            rolling_avg_values = []
            feature_names = []

            for col in rolling_avg_cols:
                if not pd.isna(row[col]):
                    # Extract feature name from column (remove _processed_rolling_avg suffix)
                    feature_name = col.replace("_processed_rolling_avg", "")
                    feature_names.append(feature_name)
                    rolling_avg_values.append(row[col])

            # Skip if no valid rolling average values
            if not rolling_avg_values:
                result_df.loc[idx, "session_overall_rolling_avg"] = np.nan
                continue

            # Apply weights if provided
            if feature_weights is not None:
                weighted_values = []
                total_weight = 0

                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(rolling_avg_values[i] * weight)
                    total_weight += weight

                # Calculate weighted average
                overall_rolling_avg = (
                    sum(weighted_values) / total_weight if total_weight > 0 else np.nan
                )
            else:
                # Calculate simple average
                overall_rolling_avg = sum(rolling_avg_values) / len(rolling_avg_values)

            # Store in result dataframe
            result_df.loc[idx, "session_overall_rolling_avg"] = overall_rolling_avg

        return result_df
