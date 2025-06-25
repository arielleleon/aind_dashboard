"""
Quantile analyzer for session-level percentile calculations and statistical analysis
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app_utils.simple_logger import get_logger

from .overall_percentile_calculator import OverallPercentileCalculator
from .statistical_utils import StatisticalUtils

logger = get_logger("quantile_analyzer")


class QuantileAnalyzer:
    """
    Analyzer for calculating and retrieving quantile-based metrics for subject performance
    """

    def __init__(
        self,
        stratified_data: Dict[str, pd.DataFrame],
        historical_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the QuantileAnalyzer with stratified subject data

        Parameters:
            stratified_data: Dict[str, pd.DataFrame]
                Dictionary of dataframes with subject averages, keyed by strata
            historical_data: Optional[pd.DataFrame]
                DataFrame containing historical strata data for all subjects
        """
        self.stratified_data = stratified_data
        self.historical_data = historical_data
        self.percentile_data = {}
        self.historical_percentile_data = None
        self.percentile_calculator = OverallPercentileCalculator()
        self.statistical_utils = StatisticalUtils()
        self.calculate_percentiles()

    def calculate_percentiles(self):
        """
        Calculate percentile ranks for each subject within each stratified group
        using combined current and historical data for more robust distributions
        """
        # Calculate percentiles for strata
        for strata, df in self.stratified_data.items():
            # Log strata size for debugging
            logger.info("Processing strata " + strata + " with " + str(len(df)))

            # Skip strata with too few subjects (consider increasing minimum)
            if len(df) < 10:  # Minimum number for meaningful percentiles
                logger.info(
                    "  Skipping strata "
                    + strata
                    + " - too few subjects ("
                    + str(len(df))
                    + " < 10)"
                )
                continue

            # Get processed feature columns
            feature_cols = [col for col in df.columns if col.endswith("_processed")]

            # Create a copy to store percentiles
            percentile_df = df.copy()

            # Check if we have outlier weights for weighted percentile calculation
            has_outlier_weights = "outlier_weight" in df.columns

            if has_outlier_weights:
                logger.info(
                    "  Using weighted percentile ranking (outlier weights detected)"
                )
                outlier_count = (df["outlier_weight"] < 1.0).sum()
                logger.info(f"    Found {outlier_count} sessions with outlier weights")

            # Calculate percentile for each feature - ranking all subjects together
            for feature in feature_cols:
                feature_values = df[feature].values

                if has_outlier_weights:
                    outlier_weights = df["outlier_weight"].values

                    # Calculate weighted percentiles for each subject
                    percentiles = []
                    for i, target_value in enumerate(feature_values):
                        if pd.isna(target_value):
                            percentiles.append(np.nan)
                        else:
                            # Use weighted percentile ranking
                            percentile = self.statistical_utils.calculate_weighted_percentile_rank(
                                reference_values=feature_values,
                                reference_weights=outlier_weights,
                                target_value=target_value,
                            )
                            percentiles.append(percentile)

                    percentile_df[f"{feature.replace('_processed', '_percentile')}"] = (
                        percentiles
                    )

                else:
                    # Traditional percentile ranking **error check
                    percentile_df[f"{feature.replace('_processed', '_percentile')}"] = (
                        df[feature].rank(pct=True) * 100
                    )

            # Store percentile data with an indicator of data source
            self.percentile_data[strata] = percentile_df

    def _create_subject_data_dict(
        self, row: pd.Series, strata: str, is_current: bool = True
    ) -> Dict:
        """
        Create a dictionary containing subject data with percentiles and processed values

        Parameters:
            row: pd.Series - Row data for the subject
            strata: str - Strata identifier
            is_current: bool - Whether this is current or historical data

        Returns:
            Dict containing subject data
        """
        subject_data = {
            "subject_id": row["subject_id"],
            "strata": strata,
            "is_current": is_current,
        }

        # Add percentile values
        percentile_cols = [col for col in row.index if col.endswith("_percentile")]
        for col in percentile_cols:
            feature = col.replace("_percentile", "")
            subject_data[f"{feature}_percentile"] = row[col]

            # Also add the processed feature value
            processed_col = f"{feature}_processed"
            if processed_col in row.index:
                subject_data[processed_col] = row[processed_col]

        # Add session count if available
        if "session_count" in row.index:
            subject_data["session_count"] = row["session_count"]

        # Add date information for historical data
        if not is_current:
            for date_col in ["first_date", "last_date"]:
                if date_col in row.index:
                    subject_data[date_col] = row[date_col]

        return subject_data

    def _process_current_strata_data(self) -> list:
        """
        Process current strata data and return list of subject data dictionaries

        Returns:
            List of dictionaries containing current subject data
        """
        all_data = []

        # Process each strata for current data
        for strata, df in self.percentile_data.items():
            # For each subject in this strata
            for _, row in df.iterrows():
                subject_data = self._create_subject_data_dict(
                    row, strata, is_current=True
                )
                all_data.append(subject_data)

        return all_data

    def _process_historical_data(self) -> list:
        """
        Process historical data and return list of subject data dictionaries

        Returns:
            List of dictionaries containing historical subject data
        """
        all_data = []

        if self.historical_percentile_data is not None:
            # For each historical subject-strata combination
            for _, row in self.historical_percentile_data.iterrows():
                # Skip if this is a current strata (already included above)
                if row.get("is_current", False):
                    continue

                subject_data = self._create_subject_data_dict(
                    row, row["strata"], is_current=False
                )
                all_data.append(subject_data)

        return all_data

    def create_comprehensive_dataframe(
        self, include_history: bool = False
    ) -> pd.DataFrame:
        """
        Create a comprehensive dataframe with all subjects, their strata, and feature percentile ranks

        Parameters:
            include_history: bool
                Whether to include historical strata data for subjects

        Returns:
            pd.DataFrame
                DataFrame containing subject_id, strata, and percentile ranks for all features
        """
        # Initialize an empty list to store data for each subject
        all_data = []

        # Process current data
        all_data.extend(self._process_current_strata_data())

        # Add historical data if requested
        if include_history:
            all_data.extend(self._process_historical_data())

        # Convert to DataFrame
        if all_data:
            return pd.DataFrame(all_data)
        else:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=["subject_id", "strata", "is_current"])

    def _get_rolling_average_columns(self, result_df: pd.DataFrame) -> list:
        """
        Get all rolling average columns from the dataframe

        Parameters:
            result_df: pd.DataFrame - Input dataframe

        Returns:
            List of rolling average column names
        """
        return [col for col in result_df.columns if col.endswith("_rolling_avg")]

    def _calculate_session_percentile_for_feature(
        self,
        rolling_value: float,
        reference_values: np.ndarray,
        reference_weights: np.ndarray,
        clean_reference_values: np.ndarray,
        clean_reference_weights: np.ndarray,
    ) -> tuple:
        """
        Calculate session percentile and confidence interval for a single feature

        Parameters:
            rolling_value: float - The rolling average value to calculate percentile for
            reference_values: np.ndarray - All reference values including NaN
            reference_weights: np.ndarray - Weights for reference values
            clean_reference_values: np.ndarray - Reference values without NaN
            clean_reference_weights: np.ndarray - Weights without NaN entries

        Returns:
            tuple: (percentile, ci_lower, ci_upper)
        """
        if pd.isna(rolling_value):
            return np.nan, np.nan, np.nan

        # Check if we have outlier weights for weighted calculation
        has_weights = not np.array_equal(
            reference_weights, np.ones(len(reference_weights))
        )

        if has_weights:
            # Use weighted percentile ranking
            percentile = self.statistical_utils.calculate_weighted_percentile_rank(
                reference_values=clean_reference_values,
                reference_weights=clean_reference_weights,
                target_value=rolling_value,
            )
        else:
            # Traditional percentile calculation
            temp_values = pd.Series(list(clean_reference_values) + [rolling_value])
            temp_values = temp_values[~temp_values.isna()]  # Remove NaN values

            # Calculate percentile using rank method for consistency
            ranks = temp_values.rank(pct=True)
            percentile = ranks.iloc[-1] * 100  # Get percentile of the last value

        # Calculate confidence interval using Wilson Score method
        ci_lower, ci_upper = (
            self.statistical_utils.calculate_percentile_confidence_interval(
                clean_reference_values, percentile, confidence_level=0.95
            )
        )

        return percentile, ci_lower, ci_upper

    def _process_strata_percentiles(
        self,
        strata_df: pd.DataFrame,
        strata: str,
        rolling_avg_cols: list,
        result_df: pd.DataFrame,
    ) -> tuple:
        """
        Process session-level percentiles for a single strata

        Parameters:
            strata_df: pd.DataFrame - Data for the current strata
            strata: str - Strata identifier
            rolling_avg_cols: list - List of rolling average columns to process
            result_df: pd.DataFrame - Result dataframe to update

        Returns:
            tuple: (created_columns, created_ci_columns)
        """
        created_columns = 0
        created_ci_columns = 0

        # Check if we have percentile data for this strata
        if strata not in self.percentile_data:
            return created_columns, created_ci_columns

        # Get reference distribution for this strata
        reference_df = self.percentile_data[strata]

        # For each rolling average feature, calculate percentile using reference distribution
        for rolling_col in rolling_avg_cols:
            # Extract the base feature name from the rolling_avg column
            feature_name = rolling_col.replace("_rolling_avg", "")
            processed_col = feature_name

            # Get reference column (processed feature values)
            if processed_col not in reference_df.columns:
                logger.info(
                    f"No reference data found for {processed_col} in strata '{strata}'"
                )
                continue

            # Get reference values for this feature
            reference_values = reference_df[processed_col].values

            # Get reference weights if available
            if "outlier_weight" in reference_df.columns:
                reference_weights = reference_df["outlier_weight"].values
            else:
                reference_weights = np.ones(len(reference_values))  # Equal weights

            # Remove NaN values from reference for CI calculation
            valid_mask = ~np.isnan(reference_values)
            clean_reference_values = reference_values[valid_mask]
            clean_reference_weights = reference_weights[valid_mask]

            if len(clean_reference_values) < 3:
                logger.info(
                    f"Insufficient reference data for CI calculation in {processed_col}, strata '{strata}'"
                )
                continue

            # For each session in this strata
            for idx, row in strata_df.iterrows():
                rolling_value = row[rolling_col]

                percentile, ci_lower, ci_upper = (
                    self._calculate_session_percentile_for_feature(
                        rolling_value,
                        reference_values,
                        reference_weights,
                        clean_reference_values,
                        clean_reference_weights,
                    )
                )

                if not pd.isna(percentile):
                    # Store in result dataframe with correct column name
                    clean_feature_name = feature_name.replace("_processed", "")
                    result_df.loc[idx, f"{clean_feature_name}_session_percentile"] = (
                        percentile
                    )
                    result_df.loc[
                        idx, f"{clean_feature_name}_session_percentile_ci_lower"
                    ] = ci_lower
                    result_df.loc[
                        idx, f"{clean_feature_name}_session_percentile_ci_upper"
                    ] = ci_upper

                    created_columns += 1
                    created_ci_columns += 2  # Lower and upper bounds

        return created_columns, created_ci_columns

    def calculate_session_level_percentiles(
        self, session_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate session-level percentile ranks for each subject in each strata
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()

        # Get all rolling average columns
        rolling_avg_cols = self._get_rolling_average_columns(result_df)

        if not rolling_avg_cols:
            logger.info("No rolling average columns found in data")
            return result_df

        # Track how many session-level percentiles we create
        total_created_columns = 0
        total_created_ci_columns = 0

        # Check if session data has outlier weights
        session_has_weights = "outlier_weight" in session_data.columns
        if session_has_weights:
            logger.info(
                "Session data contains outlier weights - will use weighted percentiles where available"
            )

        # Process each strata separately
        for strata, strata_df in session_data.groupby("strata"):
            created_columns, created_ci_columns = self._process_strata_percentiles(
                strata_df, strata, rolling_avg_cols, result_df
            )

            total_created_columns += created_columns
            total_created_ci_columns += created_ci_columns

        logger.info(f"Created {total_created_columns} session-level percentile columns")
        logger.info(f"Created {total_created_ci_columns} confidence interval columns")
        if session_has_weights or any(
            "outlier_weight" in self.percentile_data[strata].columns
            for strata in self.percentile_data
        ):
            logger.info(
                "Used weighted percentile ranking where outlier weights were available"
            )

        return result_df

    def calculate_session_overall_percentile(
        self,
        session_data: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate overall percentile for each session by taking the arithmetic mean of feature percentiles.

        Parameters:
            session_data: pd.DataFrame
                DataFrame with session-level percentiles
            feature_weights: Optional[Dict[str, float]]
                Optional weights for features

        Returns:
            pd.DataFrame
                DataFrame with calculated overall percentiles for each session
        """
        result_df = session_data.copy()

        # Get all session percentile columns
        session_percentile_cols = [
            col for col in result_df.columns if col.endswith("_session_percentile")
        ]

        if not session_percentile_cols:
            logger.info("No session percentile columns found")
            return result_df

        # Process each session
        for idx, row in result_df.iterrows():
            # Extract percentile values for this session
            percentile_values = []
            feature_names = []

            for col in session_percentile_cols:
                if pd.notna(row[col]):
                    feature_name = col.replace("_session_percentile", "")
                    percentile_values.append(row[col])
                    feature_names.append(feature_name)

            # Skip if no percentile values
            if not percentile_values:
                continue

            # Apply weights if provided
            if feature_weights is not None:
                weighted_values = []
                total_weight = 0

                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(percentile_values[i] * weight)
                    total_weight += weight

                # Calculate weighted average
                overall_percentile = (
                    sum(weighted_values) / total_weight if total_weight > 0 else np.nan
                )
            else:
                # Calculate simple average
                overall_percentile = sum(percentile_values) / len(percentile_values)

            # Store in result dataframe
            result_df.loc[idx, "session_overall_percentile"] = overall_percentile

        return result_df
