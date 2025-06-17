from typing import Any, Dict

import pandas as pd


class ThresholdAnalyzer:
    """
    Analyzer for calculating threshold-based alerts for subject performance
    """

    def __init__(self, threshold_config: Dict[str, Any] = None):
        """
        Initialize the ThresholdAnalyzer with threshold configuration

        Parameters:
            threshold_config: Dict[str, Any]
                Dictionary of feature thresholds and conditions
                Format: {
                    'feature_name': {
                        'condition': 'gt' or 'lt' or 'eq',
                        'value': threshold_value,
                        'context': Optional context like stage filter
                    }
                }
        """
        self.threshold_config = threshold_config or {}

    def evaluate_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """
        Evaluate if a value meets a threshold condition

        Parameters:
            value: Any
                The value to evaluate
            condition: str
                The condition type ('gt', 'lt', 'eq', 'gte', 'lte')
            threshold: Any
                The threshold value to compare against

        Returns:
            bool: True if condition is met, False otherwise
        """
        if pd.isna(value):
            return False

        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def analyze_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze dataframe for threshold violations and generate alerts

        Parameters:
            df: pd.DataFrame
                Input dataframe with session data

        Returns:
            pd.DataFrame
                DataFrame with threshold_alert column added
        """
        if not self.threshold_config:
            return self._create_default_result(df)

        # Create a copy of the dataframe to avoid modifying the original
        df_result = df.copy()
        df_result["threshold_alert"] = "N"  # Default to Normal

        # Process each threshold configuration
        for feature, config in self.threshold_config.items():
            self._process_threshold_feature(df_result, feature, config)

        return df_result

    def _create_default_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create default result when no thresholds are configured"""
        df_result = df.copy()
        df_result["threshold_alert"] = "N"
        return df_result

    def _process_threshold_feature(
        self, df_result: pd.DataFrame, feature: str, config: Dict[str, Any]
    ):
        """Process threshold checking for a single feature"""
        # Skip if feature not in dataframe
        if feature not in df_result.columns:
            return

        # Get condition and threshold value
        condition = config.get("condition", "gt")
        threshold = config.get("value")

        if threshold is None:
            return

        # Context filter if provided (e.g., specific stage)
        context = config.get("context")

        if context:
            self._apply_contextual_threshold(
                df_result, feature, condition, threshold, context
            )
        else:
            self._apply_global_threshold(df_result, feature, condition, threshold)

    def _apply_contextual_threshold(
        self,
        df_result: pd.DataFrame,
        feature: str,
        condition: str,
        threshold: Any,
        context: Dict[str, Any],
    ):
        """Apply threshold check with context filtering"""
        context_col = context.get("column")
        context_values = context.get("values", [])

        if not (context_col and context_col in df_result.columns and context_values):
            return

        # Only check rows matching the context
        context_mask = df_result[context_col].isin(context_values)
        matching_rows = df_result[context_mask]

        for idx in matching_rows.index:
            row_value = df_result.at[idx, feature]
            if self.evaluate_condition(row_value, condition, threshold):
                df_result.at[idx, "threshold_alert"] = "T"

    def _apply_global_threshold(
        self, df_result: pd.DataFrame, feature: str, condition: str, threshold: Any
    ):
        """Apply threshold check to all rows without context filtering"""
        for idx, row in df_result.iterrows():
            if self.evaluate_condition(row[feature], condition, threshold):
                df_result.at[idx, "threshold_alert"] = "T"

    def generate_alert(self, condition_met, alert_type, value=None, stage=None):
        """
        Generate detailed alert format with contextual information

        Parameters:
            condition_met (bool): Whether the alert condition is met
            alert_type (str): Type of alert (total_sessions, stage_sessions, water_day_total)
            value (float/int): The value that triggered the alert
            stage (str): Stage name for stage-specific alerts

        Returns:
            dict: Alert information with detailed format
        """
        if not condition_met:
            return {"alert": "N", "value": value, "stage": stage, "display_format": "N"}

        # Alert condition is met
        if alert_type == "total_sessions":
            # Format: "T | 45"
            display_format = f"T | {value}"
        elif alert_type == "stage_sessions":
            # Format: "T | STAGE_FINAL | 30"
            display_format = f"T | {stage} | {value}"
        elif alert_type == "water_day_total":
            # Format: "T | 3.7"
            display_format = f"T | {value:.1f}"
        else:
            display_format = "T"

        return {
            "alert": "T",
            "value": value,
            "stage": stage,
            "display_format": display_format,
        }

    def check_total_sessions(self, sessions):
        """Check if total sessions exceed threshold"""
        threshold = self.threshold_config.get("session", {}).get("value", 40)
        condition = self.threshold_config.get("session", {}).get("condition", "gt")

        session_count = len(sessions)
        alert_condition = (condition == "gt" and session_count > threshold) or (
            condition == "lt" and session_count < threshold
        )

        return self.generate_alert(
            alert_condition, "total_sessions", value=session_count
        )

    def check_stage_sessions(self, sessions, current_stage):
        """Check if sessions in current stage exceed threshold"""
        # Get stage-specific threshold
        stage_threshold = self.threshold_config.get(
            f"stage_{current_stage}_sessions", {}
        ).get("value", 10)

        # Count sessions in this stage
        stage_sessions = sessions[sessions["current_stage_actual"] == current_stage]
        stage_count = len(stage_sessions)

        # Check if count exceeds threshold
        alert_condition = stage_count > stage_threshold

        return self.generate_alert(
            alert_condition, "stage_sessions", value=stage_count, stage=current_stage
        )

    def check_water_day_total(self, water_day_total):
        """Check if water day total exceeds threshold"""
        threshold = self.threshold_config.get("water_day_total", {}).get("value", 3.5)
        condition = self.threshold_config.get("water_day_total", {}).get(
            "condition", "gt"
        )

        if pd.isna(water_day_total):
            return self.generate_alert(False, "water_day_total")

        alert_condition = (condition == "gt" and water_day_total > threshold) or (
            condition == "lt" and water_day_total < threshold
        )

        return self.generate_alert(
            alert_condition, "water_day_total", value=water_day_total
        )
