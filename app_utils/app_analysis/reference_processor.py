import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy import stats


class ReferenceProcessor:
    """
    Processor for subject performance data that provides:
    - Subject eligibility filtering
    - Data preprocessing
    - Data stratification
    - Feature averaging for quantile analysis
    """

    def __init__(
            self,
            features_config: Dict[str, bool],
            min_sessions: int = 1,
            min_days: int = 1
    ):
        self.features_config = features_config
        self.min_sessions = min_sessions
        self.min_days = min_days

        # Stage order for reference
        self.stage_order = [
            "STAGE_1_WARMUP",
            "STAGE_1",
            "STAGE_2",
            "STAGE_3",
            "STAGE_4",
            "STAGE_FINAL",
            "GRADUATED",
        ]

    def get_eligible_subjects(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of subjects that meet the eligibility criteria

        Parameters:
            df: pd.DataFrame
                Input dataframe with raw performance data

        Returns:
            List[str]
                List of eligible subject IDs
        """
        # Calculate subject eligibility
        subject_stats = df.groupby('subject_id').agg(
            session_count = ('session', 'count'),
            first_date = ('session_date', 'min'),
            last_date = ('session_date', 'max')
        )

        # Calculate training days
        subject_stats['training_days'] = (subject_stats['last_date'] - subject_stats['first_date']).dt.days + 1

        # Filter for eligible subjects
        eligible_subjects = subject_stats[
            (subject_stats['session_count'] >= self.min_sessions) &
            (subject_stats['training_days'] >= self.min_days)
        ].index.tolist()

        return eligible_subjects

    def preprocess_data(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
        """
        Preprocess the input data with standardization/feature transformations

        Parameters:
            df: pd.DataFrame
                Input dataframe with raw performance data
            remove_outliers: bool
                Whether to remove outliers before calculating percentiles

        Returns:
            pd.DataFrame
                Processed dataframe with standardized features
        """
        df = df.copy()

        # Convert to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(df['session_date']):
            df['session_date'] = pd.to_datetime(df['session_date'])

        # Enhanced filtering of off-curriculum sessions (check for both None value and string "None")
        off_curriculum_mask = (
            df['curriculum_name'].isna() | 
            (df['curriculum_name'] == "None") |
            df['current_stage_actual'].isna() | 
            (df['current_stage_actual'] == "None") |
            df['curriculum_version'].isna() |
            (df['curriculum_version'] == "None")
        )
        
        # Report and remove off-curriculum sessions
        off_count = off_curriculum_mask.sum()
        if off_count > 0:
            print(f"Reference processor preprocess: Removing {off_count} off-curriculum sessions")
            df = df[~off_curriculum_mask].copy()

        # Clean data - additional filtering
        df_clean = df.query('curriculum_name != "None" and curriculum_version != "0.1"').copy()

        def _map_curriculum_ver(ver):
            if "2.3" in ver:
                return "v3"
            elif "1.0" in ver:
                return "v1"
            else:
                return "v2"

        df_clean["curriculum_version_group"] = df_clean["curriculum_version"].map(_map_curriculum_ver)

        scaler = StandardScaler()

        for feature, lower_is_better in self.features_config.items():
            if feature not in df_clean.columns:
                continue

            # Standardize the feature
            scaled_values = scaler.fit_transform(df_clean[[feature]])

            # Invert if lower is better (higher values always mean better performance)
            if lower_is_better:
                scaled_values = -scaled_values

            # Add processed feature to dataframe
            df_clean[f'{feature}_processed'] = scaled_values

        # Make outlier removal optional
        if remove_outliers:
            processed_features = [col for col in df_clean.columns if col.endswith('_processed')]
            for feature in processed_features:
                mean = df_clean[feature].mean()
                std = df_clean[feature].std()
                df_clean = df_clean[(df_clean[feature] >= mean - 3*std) & (df_clean[feature] <= mean + 3*std)]

        return df_clean

    def _simplify_strata(self, strat_id: str) -> str:
        """
        Combine strata to simplify distributions
        
        Parameters:
            strat_id: String in format like "Uncoupled Baiting_STAGE_FINAL_v3"
            
        Returns:
            Simplified group identifier
        """
        # First, separate curriculum name from the rest
        parts = strat_id.split('_')
        
        # Handle curriculum name
        if 'Without' in strat_id:
            # Find the index where the stage info starts
            stage_start = next(i for i, part in enumerate(parts) if 'STAGE' in part or 'GRADUATED' in part)
            curriculum = '_'.join(parts[:stage_start])
            stage_parts = parts[stage_start:]  # Get all parts after curriculum name
        else:
            # Find the index where the stage info starts
            stage_start = next(i for i, part in enumerate(parts) if 'STAGE' in part or 'GRADUATED' in part)
            curriculum = '_'.join(parts[:stage_start])
            stage_parts = parts[stage_start:]  # Get all parts after curriculum name
        
        # Get the full stage name and version
        stage = '_'.join(stage_parts[:-1])  # Join all parts except the last (version)
        version = stage_parts[-1]  # Keep the original version (v1, v2, v3)

        # Simplify stage
        if 'STAGE_FINAL' in stage or 'GRADUATED' in stage:
            simplified_stage = 'ADVANCED'
        elif any(s in stage for s in ['STAGE_4', 'STAGE_3']):
            simplified_stage = 'INTERMEDIATE'
        elif any(s in stage for s in ['STAGE_2', 'STAGE_1', 'STAGE_1_WARMUP']):
            simplified_stage = 'BEGINNER'
        else:
            print(f"Warning: Unknown stage format: {stage}")
            simplified_stage = 'UNKNOWN'

        return f"{curriculum}_{simplified_stage}_{version}"

    def assign_subject_strata(self, df: pd.DataFrame, use_simplified: bool = True) -> pd.DataFrame:
        """
        Assign stratification group to each subject based on their most recent session
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed performance data
            use_simplified: bool
                Whether to use simplified strata groups (default: True)
    
        Returns:
            pd.DataFrame
                DataFrame with subject stratification information
        """
        df = df.copy()
        
        # Create strata ID for each session
        df['strata_id'] = df.apply(
            lambda row: f"{row['curriculum_name']}_{row['current_stage_actual']}_{row['curriculum_version_group']}",
            axis=1
        )

        # Add simplified strata if requested
        if use_simplified:
            df['strata'] = df['strata_id'].apply(self._simplify_strata)
        else:
            df['strata'] = df['strata_id']

        return df
    
    def _calculate_weighted_average(self, df: pd.DataFrame, features: List[str],
                                    use_weighted_avg: bool = True, decay_factor: float = 0.9) -> Dict[str, float]:
        """
        Calculate weighted or normal average for a set of features in input dataframe

        Parameters:
            df: pd.DataFrame
                DataFrame containing sessions to average
            features: List[str]
                List of features to average
            use_weighted_avg: bool
                Whether to use weighted averaging
            decay_factor: float
                Factor for exponential decacy weighting (0-1) (higher == more weight on recent sessions)
            
        Returns:
            Dict[str, float]
                Dictionary mapping features to average values
        """
        if df.empty:
            return {}
        
        # If only one row return values
        if len(df) ==1:
            return {feature: df[feature].iloc[0] for feature in features}
        
        # Sort by date for weighting average
        df_sorted = df.sort_values('session_date')

        if use_weighted_avg:
            # Calculate weights based on session recency
            session_count = len(df_sorted)
            # Create exponentially increasing weights (early sessions -> lower weight, later session -> higher weight)
            weights = np.array([decay_factor ** (session_count - i -1) for i in range(session_count)])
            # Normalize weights to sum to 1
            weights = weights / weights.sum()

            # Calculate weighted average for each feature
            averages = {}
            for feature in features:
                feature_values = df_sorted[feature].values
                weighted_avg = np.sum(feature_values * weights)
                averages[feature] = weighted_avg

            return averages
        else:
            # Use normal average
            return df[features].mean().to_dict()
        
    def calculate_subject_averages(self, df: pd.DataFrame, include_history: bool = True,
                                   use_weighted_avg: bool = True, decay_factor: float = 0.9) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Calculate average feature values for each subject within their stratified group,
        using only sessions that match the subject's final strata.
        Optionally calculates historical averages across all strata a subject has been in.
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed and stratified data
            include_history: bool
                Whether to calculate historical averages for all strata a subject has been in
            use_weighted_avg: bool
                Whether to use weighted averaging based on session recency
            decay_factor: float
                 Factor for exponential decacy weighting (0-1) (higher == more weight on recent sessions)
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]
                - DataFrame with average feature values per subject per final strata
                - DataFrame with historical averages across all strata (if include_history=True)
        """
        # Get list of processed features
        processed_features = [col for col in df.columns if col.endswith('_processed')]
        
        # Get the most recent session for each subject to determine final strata
        latest_sessions = df.sort_values('session_date').groupby('subject_id').last().reset_index()
        subject_final_strata = dict(zip(latest_sessions['subject_id'], latest_sessions['strata']))

        # Create a list to store subject-strata combinations with their average feature values
        subject_strata_averages = []

        # Process each subject
        for subject_id, final_strata in subject_final_strata.items():
            # Get only sessions for this subject that match their final strata
            subject_sessions = df[(df['subject_id'] == subject_id) & (df['strata'] == final_strata)]

            # Continue if there are matching sessions
            if not subject_sessions.empty:
                # Get average for processed features
                averages = self._calculate_weighted_average(
                    subject_sessions,
                    processed_features,
                    use_weighted_avg,
                    decay_factor
                )

                # Create a record for this subject-strata combination
                record = {
                    'subject_id': subject_id,
                    'strata': final_strata,
                    'session_count': len(subject_sessions), # Session count for reference
                    'is_current': True,  # Flag to indicate this is the subject's current strata
                    'first_date': subject_sessions['session_date'].min(),
                    'last_date': subject_sessions['session_date'].max()
                }
                record.update(averages)

                subject_strata_averages.append(record)
        
        # Calculate historical data if requested
        subject_history = None
        if include_history:
            # Create a list to store historical subject-strata combinations
            historical_averages = []
            
            # Get all unique subject-strata combinations
            subject_strata_combinations = df[['subject_id', 'strata']].drop_duplicates()
            
            # Process each subject-strata combination
            for _, row in subject_strata_combinations.iterrows():
                subject_id = row['subject_id']
                strata = row['strata']
                
                # Skip if this is the subject's final strata (already included above)
                if subject_final_strata.get(subject_id) == strata:
                    continue
                    
                # Get only sessions for this subject in this strata
                subject_strata_sessions = df[(df['subject_id'] == subject_id) & (df['strata'] == strata)]
                
                if not subject_strata_sessions.empty:
                    # Calculate averages for processed features
                    averages = self._calculate_weighted_average(
                        subject_strata_sessions,
                        processed_features,
                        use_weighted_avg,
                        decay_factor
                    )
                    
                    # Get first and last date in this strata for chronological ordering
                    first_date = subject_strata_sessions['session_date'].min()
                    last_date = subject_strata_sessions['session_date'].max()
                    
                    # Create a record for this historical subject-strata combination
                    record = {
                        'subject_id': subject_id,
                        'strata': strata,
                        'session_count': len(subject_strata_sessions),
                        'first_date': first_date,
                        'last_date': last_date,
                        'is_current': False  # Flag to indicate this is a historical strata
                    }
                    record.update(averages)
                    
                    historical_averages.append(record)
            
            # Combine current and historical data
            all_averages = subject_strata_averages + historical_averages
            
            # Convert to DataFrame
            if all_averages:
                subject_history = pd.DataFrame(all_averages)
            else:
                # Return empty DataFrame with expected columns
                columns = ['subject_id', 'strata', 'session_count', 'first_date', 'last_date', 'is_current'] + processed_features
                subject_history = pd.DataFrame(columns=columns)

        # Convert current strata averages to DataFrame
        if subject_strata_averages:
            current_averages_df = pd.DataFrame(subject_strata_averages)
        else:
            # Return empty DataFrame with expected columns
            columns = ['subject_id', 'strata', 'session_count', 'is_current'] + processed_features
            current_averages_df = pd.DataFrame(columns=columns)
        
        return current_averages_df, subject_history
    
    def calculate_session_level_rolling_averages(self, df: pd.DataFrame, decay_factor: float = 0.9) -> pd.DataFrame:
        """
        Calculate rolling averages for features for each session

        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed and stratified data
            decay_factor: float
                Factor for exponential decacy weighting (0-1) (higher == more weight on recent sessions)

        Returns:
            pd.DataFrame
                Dataframe with rolling averages for each session
        """
        # Get list of processed features
        processed_features = [col for col in df.columns if col.endswith('_processed')]

        if not processed_features:
            print("No processed features found in data")
            return df.copy()
        
        result_df = df.copy()

        # Debug information
        print(f"Calculating rolling averages for {len(processed_features)} features")
        print(f"Feature examples: {processed_features[:3]}")

        for subject_id, subject_data in df.groupby('subject_id'):
            for strata, strata_sessions in subject_data.groupby('strata'):
                # Sort sessions by date (earliest first)
                strata_sessions = strata_sessions.sort_values('session_date')
                
                # Get indices of these sessions in original dataframe
                indices = strata_sessions.index
                
                # Calculate rolling weighted average for each session
                for i, session_idx in enumerate(indices):
                    # Get all sessions up to and including current one
                    included_sessions = strata_sessions.iloc[:i+1]
                    
                    # Skip if only one session (no need for rolling average)
                    if len(included_sessions) == 1:
                        # Use the raw values for the first session
                        for feature in processed_features:
                            result_df.loc[session_idx, f"{feature}_rolling_avg"] = included_sessions[feature].iloc[0]
                        continue
                    
                    # Calculate weighted averages
                    averages = self._calculate_weighted_average(
                        included_sessions,
                        processed_features,
                        use_weighted_avg=True,
                        decay_factor=decay_factor
                    )
                    
                    # Add rolling averages to result
                    for feature, avg_value in averages.items():
                        result_df.loc[session_idx, f"{feature}_rolling_avg"] = avg_value
        
        # Verify column creation
        rolling_avg_cols = [col for col in result_df.columns if col.endswith('_rolling_avg')]
        print(f"Created {len(rolling_avg_cols)} rolling average columns")
        
        return result_df

    def prepare_for_quantile_analysis(self, df: pd.DataFrame, include_history: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for quantile analysis by stratifying and calculating subject averages
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed performance data
            include_history: bool
                Whether to include historical strata data for subjects
                
        Returns:
            Dict[str, pd.DataFrame]
                Dictionary of dataframes with subject averages, keyed by strata
        """
        # Assign strata to each session
        stratified_df = self.assign_subject_strata(df)
        
        # Calculate subject averages and optionally get historical data
        subject_averages, subject_history = self.calculate_subject_averages(
            stratified_df, include_history=include_history
        )
        
        # Store the historical data as an attribute that can be accessed separately
        self.subject_history = subject_history
        
        # Create enhanced strata dataframes that include historical data
        strata_dfs = {}
        if include_history and subject_history is not None:
            # Get unique strata across both current and historical data
            all_strata = set(subject_averages['strata'].unique())
            if subject_history is not None:
                all_strata.update(subject_history['strata'].unique())
            
            # For each strata, combine current and historical data
            for strata in all_strata:
                # Get current subjects in this strata
                current_strata_df = subject_averages[subject_averages['strata'] == strata].copy()
                
                # Get historical subjects in this strata (exclude subjects already in current)
                if subject_history is not None:
                    historical_strata_df = subject_history[
                        (subject_history['strata'] == strata) & 
                        (~subject_history['subject_id'].isin(current_strata_df['subject_id']))
                    ].copy()
                    
                    # Combine current and historical data for this strata
                    if not historical_strata_df.empty:
                        strata_dfs[strata] = pd.concat([current_strata_df, historical_strata_df])
                    else:
                        strata_dfs[strata] = current_strata_df
                else:
                    strata_dfs[strata] = current_strata_df
        else:
            # Just use current data if not including history
            for strata in subject_averages['strata'].unique():
                strata_dfs[strata] = subject_averages[subject_averages['strata'] == strata].copy()
        
        return strata_dfs
    
    def prepare_session_level_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare session-level data for percentile calculation:
        1. Assign strata to each session
        2. Calculate rolling weighted averages for each session
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed data
                
        Returns:
            pd.DataFrame
                DataFrame with session-level rolling averages, ready for percentile calculation
        """
        # Assign strata to each session
        stratified_df = self.assign_subject_strata(df)
        
        # Calculate rolling weighted averages for each session
        session_level_data = self.calculate_session_level_rolling_averages(stratified_df)
        
        # Add session_index (sequential number for each subject-strata combination)
        session_level_data['session_index'] = session_level_data.groupby(['subject_id', 'strata']).cumcount() + 1
        
        return session_level_data