import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from aind_analysis_arch_result_access.han_pipeline import get_session_table


class EnhancedDataLoader:
    """
    Enhanced data loader with unified session management capabilities

    Combines the existing AppLoadData functionality with extracted data loading
    methods from AppUtils for improved separation of concerns.
    """

    def __init__(self):
        """Initialize the enhanced data loader"""
        self.session_table = None
        self.last_load_time = None
        self.load_parameters = None

        # Load initial data
        self.load()

    def load(self, load_bpod: bool = False) -> pd.DataFrame:
        """
        Load session dataframe with optional bpod data

        Parameters:
            load_bpod (bool): Whether to load bpod data

        Returns:
            pd.DataFrame: Loaded session table
        """
        try:
            print(
                f"Loading session data with bpod={'enabled' if load_bpod else 'disabled'}"
            )
            self.session_table = get_session_table(if_load_bpod=load_bpod)
            self.last_load_time = datetime.now()
            self.load_parameters = {"load_bpod": load_bpod}

            print(f"Successfully loaded {len(self.session_table)} sessions")
            return self.session_table

        except Exception as e:
            error_msg = f"Failed to load session table: {str(e)}"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)

    def get_data(self) -> pd.DataFrame:
        """
        Get current session table, loading if necessary

        Returns:
            pd.DataFrame: Session table
        """
        if self.session_table is None:
            print("No session data available, loading...")
            self.load()
        return self.session_table

    def reload_data(self, load_bpod: bool = False) -> pd.DataFrame:
        """
        Force reload session data regardless of current state

        Parameters:
            load_bpod (bool): Whether to load bpod data

        Returns:
            pd.DataFrame: Reloaded session data
        """
        print(
            f"Force reloading session data with bpod={'enabled' if load_bpod else 'disabled'}"
        )
        return self.load(load_bpod=load_bpod)

    def get_subject_sessions(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Get all sessions for a specific subject

        Parameters:
            subject_id (str): The subject ID to retrieve sessions for

        Returns:
            pd.DataFrame or None: DataFrame containing all sessions for the subject,
                                 sorted by session date (most recent first), or None if not found
        """
        try:
            # Get all data
            all_data = self.get_data()

            # Filter for specific subject
            subject_data = all_data[all_data["subject_id"] == subject_id].copy()

            if subject_data.empty:
                print(f"No sessions found for subject {subject_id}")
                return None

            # Sort by session date (most recent first)
            subject_data = subject_data.sort_values("session_date", ascending=False)

            print(f"Found {len(subject_data)} sessions for subject {subject_id}")
            return subject_data

        except Exception as e:
            print(f"Error getting sessions for subject {subject_id}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def get_most_recent_subject_sessions(self) -> pd.DataFrame:
        """
        Get the most recent session for each subject (raw data only)

        Note: This returns only the raw session data. For processed data with
        percentiles and metrics, use AppUtils.get_most_recent_subject_sessions()

        Returns:
            pd.DataFrame: DataFrame with most recent raw session for each subject
        """
        try:
            # Get all session data
            all_data = self.get_data()

            if all_data.empty:
                print("No session data available")
                return pd.DataFrame()

            # Sort by subject ID and session date (descending)
            sorted_data = all_data.sort_values(
                ["subject_id", "session_date"], ascending=[True, False]
            )

            # Get most recent session for each subject
            most_recent = sorted_data.groupby("subject_id").first().reset_index()

            print(f"Retrieved most recent sessions for {len(most_recent)} subjects")
            return most_recent

        except Exception as e:
            print(f"Error getting most recent subject sessions: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_subjects_list(self) -> list:
        """
        Get list of all unique subject IDs

        Returns:
            list: List of unique subject IDs
        """
        try:
            data = self.get_data()
            if data.empty:
                return []

            subjects = data["subject_id"].unique().tolist()
            subjects.sort()  # Sort alphabetically for consistency

            print(f"Found {len(subjects)} unique subjects")
            return subjects

        except Exception as e:
            print(f"Error getting subjects list: {str(e)}")
            return []

    def get_sessions_count(self) -> Dict[str, int]:
        """
        Get session counts by subject

        Returns:
            Dict[str, int]: Dictionary mapping subject_id to session count
        """
        try:
            data = self.get_data()
            if data.empty:
                return {}

            session_counts = data["subject_id"].value_counts().to_dict()

            total_sessions = sum(session_counts.values())
            print(
                f"Session counts calculated: {len(session_counts)} subjects, {total_sessions} total sessions"
            )

            return session_counts

        except Exception as e:
            print(f"Error getting session counts: {str(e)}")
            return {}

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the loaded data

        Returns:
            Dict[str, Any]: Summary statistics including subjects, sessions, date range
        """
        try:
            data = self.get_data()

            if data.empty:
                return {
                    "total_sessions": 0,
                    "total_subjects": 0,
                    "date_range": None,
                    "last_load_time": self.last_load_time,
                    "load_parameters": self.load_parameters,
                }

            summary = {
                "total_sessions": len(data),
                "total_subjects": data["subject_id"].nunique(),
                "date_range": {
                    "earliest": data["session_date"].min(),
                    "latest": data["session_date"].max(),
                },
                "last_load_time": self.last_load_time,
                "load_parameters": self.load_parameters,
                "subjects_list": sorted(data["subject_id"].unique().tolist()),
            }

            print(
                f"Data summary: {summary['total_sessions']} sessions, {summary['total_subjects']} subjects"
            )
            return summary

        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
            return {
                "error": str(e),
                "last_load_time": self.last_load_time,
                "load_parameters": self.load_parameters,
            }

    def _check_required_columns(self, data: pd.DataFrame, required_columns: list) -> list:
        """Check for missing required columns"""
        missing_columns = [
            col for col in required_columns if col not in data.columns
        ]
        issues = []
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        return issues

    def _check_null_values(self, data: pd.DataFrame, required_columns: list) -> list:
        """Check for null values in critical columns"""
        issues = []
        for col in required_columns:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"Found {null_count} null values in {col}")
        return issues

    def _check_duplicate_sessions(self, data: pd.DataFrame) -> list:
        """Check for duplicate sessions"""
        issues = []
        if "subject_id" in data.columns and "session" in data.columns:
            duplicates = data.duplicated(subset=["subject_id", "session"]).sum()
            if duplicates > 0:
                issues.append(
                    f"Found {duplicates} duplicate subject-session combinations"
                )
        return issues

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded session data for common issues

        Returns:
            Dict[str, Any]: Validation results with any issues found
        """
        try:
            data = self.get_data()

            if data.empty:
                return {"valid": False, "issues": ["No data loaded"]}

            required_columns = ["subject_id", "session_date", "session"]
            issues = []

            # Run validation checks using helper methods
            issues.extend(self._check_required_columns(data, required_columns))
            issues.extend(self._check_null_values(data, required_columns))
            issues.extend(self._check_duplicate_sessions(data))

            validation_result = {
                "valid": len(issues) == 0,
                "issues": issues,
                "total_sessions": len(data),
                "total_subjects": (
                    data["subject_id"].nunique() if "subject_id" in data.columns else 0
                ),
            }

            if validation_result["valid"]:
                print("Data validation passed")
            else:
                print(f"Data validation found {len(issues)} issues: {issues}")

            return validation_result

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "error": str(e),
            }

    def is_data_loaded(self) -> bool:
        """
        Check if data is currently loaded

        Returns:
            bool: True if data is loaded, False otherwise
        """
        return self.session_table is not None and not self.session_table.empty

    def clear_data(self) -> None:
        """
        Clear all loaded data and reset state
        """
        print("Clearing all loaded data")
        self.session_table = None
        self.last_load_time = None
        self.load_parameters = None


# Backward compatibility: Create alias for the original class name
class AppLoadData(EnhancedDataLoader):
    """
    Backward compatibility alias for EnhancedDataLoader

    This maintains compatibility with existing code that imports AppLoadData
    """

    pass
