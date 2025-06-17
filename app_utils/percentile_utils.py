"""
Percentile Calculation Coordination Module

This module coordinates percentile calculations between statistical analysis components
and application needs. It bridges the gap between pipeline processing and percentile
calculation requirements.

REFACTORING: Extracted from app_utils.py to achieve single responsibility principle
for percentile coordination logic.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .app_analysis.overall_percentile_calculator import OverallPercentileCalculator


class PercentileCoordinator:
    """
    Coordinator for percentile calculations across the application

    This class orchestrates percentile calculations by:
    1. Coordinating with OverallPercentileCalculator for core calculations
    2. Working with pipeline_manager for data processing
    3. Managing percentile result caching
    4. Providing percentile data to UI components

    REFACTORING: Extracted to separate percentile coordination concerns
    from the main AppUtils class.
    """

    def __init__(self, cache_manager=None, pipeline_manager=None):
        """
        Initialize percentile coordinator

        Parameters:
            cache_manager: CacheManager instance for result caching
            pipeline_manager: DataPipelineManager for data processing coordination
        """
        self.cache_manager = cache_manager
        self.pipeline_manager = pipeline_manager

        # Initialize the core percentile calculator
        self.percentile_calculator = OverallPercentileCalculator()

    def get_session_overall_percentiles(
        self,
        subject_ids: Optional[List[str]] = None,
        use_cache: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Get overall percentile scores for subjects using the session - level pipeline

        This method coordinates between the data pipeline and percentile calculation
        to provide session - level percentile scores for specified subjects.

        Parameters:
            subject_ids: Optional[List[str]]
                List of specific subjects to calculate for (None = all subjects)
            use_cache: bool
                Whether to use cached results if available
            feature_weights: Optional[Dict[str, float]]
                Optional weights for features in percentile calculation

        Returns:
            pd.DataFrame: DataFrame with session - level overall percentile scores
        """
        # Get session - level data from pipeline or cache
        session_data = self._get_session_level_data(use_cache=use_cache)

        # Return empty DataFrame if no data available
        if session_data.empty:
            return pd.DataFrame()

        # Filter for specific subjects if requested
        if subject_ids is not None:
            session_data = session_data[session_data["subject_id"].isin(subject_ids)]

        # Return empty DataFrame if no subjects match filter
        if session_data.empty:
            return pd.DataFrame()

        # Get the most recent session for each subject (for summary views)
        # Only attempt sorting if required columns exist
        if (
            "subject_id" in session_data.columns
            and "session_date" in session_data.columns
        ):
            most_recent = (
                session_data.sort_values(["subject_id", "session_date"])
                .groupby("subject_id")
                .last()
                .reset_index()
            )
        else:
            # Fallback: return the data as - is if sorting columns are missing
            most_recent = session_data

        return most_recent

    def _get_session_level_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get session - level data from pipeline manager or cache

        Parameters:
            use_cache: bool
                Whether to use cached session data

        Returns:
            pd.DataFrame: Session - level data with percentiles
        """
        # Check cache first if available and enabled
        if (
            use_cache
            and self.cache_manager
            and self.cache_manager.has("session_level_data")
        ):
            return self.cache_manager.get("session_level_data")

        # If no cache or pipeline manager, return empty DataFrame
        # The actual data processing should be handled by AppUtils
        # This method is primarily for coordination, not data processing
        if not self.pipeline_manager or not self.cache_manager:
            return pd.DataFrame()

        # Check if raw data exists in cache
        if self.cache_manager.has("raw_data"):
            raw_data = self.cache_manager.get("raw_data")
            if raw_data is not None and not raw_data.empty:
                # Process through pipeline manager
                return self.pipeline_manager.process_data_pipeline(
                    raw_data, use_cache=use_cache
                )

        # Return empty DataFrame if no data available
        return pd.DataFrame()

    def calculate_percentiles_for_sessions(
        self,
        session_data: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate percentiles for a set of sessions using the core calculator

        This method delegates to the OverallPercentileCalculator for the actual
        percentile calculations while coordinating the data flow.

        Parameters:
            session_data: pd.DataFrame
                Session data to calculate percentiles for
            feature_weights: Optional[Dict[str, float]]
                Optional weights for features in calculation

        Returns:
            pd.DataFrame: Session data with percentile calculations added
        """
        if session_data.empty:
            return session_data

        # Delegate to the core percentile calculator
        # Note: The actual calculation methods would depend on the
        # OverallPercentileCalculator's interface
        try:
            # Calculate session overall percentiles using the calculator
            enhanced_data = (
                self.percentile_calculator.calculate_session_overall_percentile(
                    session_data
                )
            )

            # Calculate rolling averages if available
            if hasattr(
                self.percentile_calculator, "calculate_session_overall_rolling_average"
            ):
                enhanced_data = self.percentile_calculator.calculate_session_overall_rolling_average(
                    enhanced_data
                )

            return enhanced_data

        except Exception as e:
            print(f"Error calculating percentiles: {e}")
            return session_data

    def get_percentiles_by_strata(
        self, strata_name: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get percentile distributions for a specific strata

        Parameters:
            strata_name: str
                Name of the strata to get percentiles for
            use_cache: bool
                Whether to use cached percentile data

        Returns:
            Optional[pd.DataFrame]: Percentile data for the strata
        """
        # Get session - level data
        session_data = self._get_session_level_data(use_cache=use_cache)

        if session_data.empty:
            return None

        # Filter for the specific strata
        strata_data = session_data[session_data["strata"] == strata_name]

        if strata_data.empty:
            return None

        return strata_data

    def get_subject_percentile_history(
        self, subject_id: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get percentile history for a specific subject across all their sessions

        Parameters:
            subject_id: str
                Subject ID to get percentile history for
            use_cache: bool
                Whether to use cached data

        Returns:
            Optional[pd.DataFrame]: Subject's percentile history
        """
        # Get session - level data
        session_data = self._get_session_level_data(use_cache=use_cache)

        if session_data.empty:
            return None

        # Filter for the specific subject
        subject_data = session_data[session_data["subject_id"] == subject_id]

        if subject_data.empty:
            return None

        # Sort by session date for chronological history
        subject_history = subject_data.sort_values("session_date")

        return subject_history

    def clear_percentile_cache(self):
        """Clear percentile calculation cache"""
        if hasattr(self.percentile_calculator, "clear_cache"):
            self.percentile_calculator.clear_cache()

    def get_percentile_summary_stats(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get summary statistics for percentile calculations

        Parameters:
            use_cache: bool
                Whether to use cached data

        Returns:
            Dict[str, Any]: Summary statistics for percentiles
        """
        session_data = self._get_session_level_data(use_cache=use_cache)

        if session_data.empty:
            return {}

        # Calculate summary statistics for numeric percentile columns only
        percentile_cols = [
            col for col in session_data.columns if "percentile" in col.lower()
        ]

        summary = {}
        for col in percentile_cols:
            if col in session_data.columns:
                # Only process numeric columns
                try:
                    series = pd.to_numeric(session_data[col], errors="coerce")
                    if not series.isna().all():  # If column has any numeric values
                        summary[col] = {
                            "mean": series.mean(),
                            "median": series.median(),
                            "std": series.std(),
                            "min": series.min(),
                            "max": series.max(),
                            "count": series.count(),
                        }
                except (TypeError, ValueError):
                    # Skip non - numeric columns
                    continue

        return summary

    def validate_percentile_calculations(
        self, session_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate percentile calculations for consistency and accuracy

        Parameters:
            session_data: pd.DataFrame
                Session data with percentile calculations

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {"valid": True, "issues": [], "warnings": []}

        # Check for percentile columns
        percentile_cols = [
            col for col in session_data.columns if "percentile" in col.lower()
        ]

        if not percentile_cols:
            validation_results["issues"].append("No percentile columns found")
            validation_results["valid"] = False
            return validation_results

        # Validate percentile ranges (should be 0 - 100) for numeric columns only
        for col in percentile_cols:
            if col in session_data.columns:
                try:
                    # Convert to numeric, coercing non - numeric values to NaN
                    numeric_series = pd.to_numeric(session_data[col], errors="coerce")

                    # Skip validation if column is entirely non - numeric
                    if numeric_series.isna().all():
                        continue

                    # Check for values outside 0 - 100 range
                    invalid_values = numeric_series[
                        (numeric_series < 0) | (numeric_series > 100)
                    ]

                    if not invalid_values.empty:
                        validation_results["issues"].append(
                            f"Invalid percentile values in {col}: {len(invalid_values)} rows"
                        )
                        validation_results["valid"] = False

                    # Check for excessive NaN values
                    nan_count = numeric_series.isna().sum()
                    total_count = len(session_data)

                    if nan_count > total_count * 0.5:  # More than 50% NaN
                        validation_results["warnings"].append(
                            f"High NaN rate in {col}: {nan_count}/{total_count} ({100 * nan_count / total_count:.1f}%)"
                        )

                except (TypeError, ValueError) as e:
                    validation_results["warnings"].append(
                        f"Could not validate column {col}: {str(e)}"
                    )

        return validation_results


def calculate_heatmap_colorscale(mode: str = "binned") -> List[Tuple[float, str]]:
    """
    Calculate colorscale for percentile heatmaps with alert category mapping

    This function provides colorscales that match the alert category system used
    throughout the dashboard for consistent visual representation.

    Parameters:
        mode: str
            Colorscale mode - 'binned' for discrete categories, 'continuous' for smooth gradients

    Returns:
        List[Tuple[float, str]]: Colorscale definition as list of (position, color) tuples
    """
    if mode == "continuous":
        return _create_continuous_colorscale()
    else:
        return _create_custom_colorscale()


def _create_custom_colorscale() -> List[Tuple[float, str]]:
    """Create custom colorscale matching alert categories"""
    # Create a colorscale that maps percentile ranges to alert colors
    return [
        [0.0, "#FF6B35"],  # 0 - 6.5% (SB) - Dark orange
        [0.065, "#FF6B35"],  #
        [0.065, "#FFB366"],  # 6.5 - 28% (B) - Light orange
        [0.28, "#FFB366"],  #
        [0.28, "#E8E8E8"],  # 28 - 72% (N) - Light grey
        [0.72, "#E8E8E8"],  #
        [0.72, "#4A90E2"],  # 72 - 93.5% (G) - Light blue
        [0.935, "#4A90E2"],  #
        [0.935, "#2E5A87"],  # 93.5 - 100% (SG) - Dark blue
        [1.0, "#2E5A87"],
    ]


def _create_continuous_colorscale() -> List[Tuple[float, str]]:
    """Create smooth continuous colorscale with gradual transitions"""
    # Create a smooth gradient from red (low) through grey (normal) to blue (high)
    return [
        [0.0, "#FF4444"],  # 0% - Bright red (worst performance)
        [0.065, "#FF6B35"],  # 6.5% - Orange - red transition
        [0.15, "#FFA366"],  # 15% - Light orange
        [0.28, "#FFD699"],  # 28% - Very light orange
        [0.40, "#F0F0F0"],  # 40% - Light grey (approaching normal)
        [0.50, "#E8E8E8"],  # 50% - Normal grey (median)
        [0.60, "#E0E8F0"],  # 60% - Very light blue
        [0.72, "#B8D4F0"],  # 72% - Light blue
        [0.85, "#7BB8E8"],  # 85% - Medium blue
        [0.935, "#4A90E2"],  # 93.5% - Good blue
        [1.0, "#1E5A96"],  # 100% - Deep blue (best performance)
    ]


def format_feature_display_names(features_config: Dict[str, bool]) -> Dict[str, str]:
    """
    Convert feature keys to human - readable display names

    Creates a mapping from technical feature names to properly formatted display names
    for visualization components.

    Parameters:
        features_config: Dict[str, bool]
            Configuration dict mapping feature names to whether they should be included

    Returns:
        Dict[str, str]: Mapping from feature keys to display names
    """
    display_names = {}

    for feature in features_config.keys():
        # Transform technical names to display format
        display_name = (
            feature.replace("_", " ").replace("abs(", "|").replace(")", "|").title()
        )
        display_names[feature] = display_name

    return display_names
