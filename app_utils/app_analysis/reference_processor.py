import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy import stats
from .statistical_utils import StatisticalUtils


class ReferenceProcessor:
    """
    Processor for subject performance data that provides:
    - Subject eligibility filtering
    - Data preprocessing with enhanced outlier detection
    - Data stratification
    - Feature averaging for quantile analysis
    """

    def __init__(
            self,
            features_config: Dict[str, bool],
            min_sessions: int = 1,
            min_days: int = 1,
            outlier_config: Optional[Dict[str, Any]] = None
    ):
        self.features_config = features_config
        self.min_sessions = min_sessions
        self.min_days = min_days
        
        # Enhanced outlier detection configuration
        if outlier_config is None:
            outlier_config = {
                'method': 'iqr',  # 'iqr', 'modified_zscore', or 'none'
                'factor': 1.5,    # IQR multiplier factor
                'handling': 'weighted',  # 'weighted', 'remove', or 'none'
                'outlier_weight': 0.5,   # Weight for detected outliers (0.5 = half weight)
                'min_data_points': 4     # Minimum data points needed for outlier detection
            }
        self.outlier_config = outlier_config
        
        # Initialize statistical utilities
        self.statistical_utils = StatisticalUtils()
        
        print(f"ReferenceProcessor initialized with outlier config: {self.outlier_config}")

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

    def preprocess_data(self, df: pd.DataFrame, remove_outliers: bool = None) -> pd.DataFrame:
        """
        Preprocess the input data with strata-specific standardization and enhanced outlier detection
        
        ENHANCED IN PHASE 2: Now uses IQR-based outlier detection with weighted averaging
        instead of complete outlier removal for better data retention and robustness.

        Parameters:
            df: pd.DataFrame
                Input dataframe with raw performance data
            remove_outliers: bool, optional
                Whether to apply outlier detection (overrides config if specified)

        Returns:
            pd.DataFrame
                Processed dataframe with strata-specific standardized features and outlier weights
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

        # CRITICAL CHANGE: Assign strata BEFORE standardization
        print("Assigning strata before standardization for strata-specific processing...")
        df_with_strata = self.assign_subject_strata(df_clean, use_simplified=True)
        
        # Determine outlier detection approach
        apply_outlier_detection = remove_outliers
        if apply_outlier_detection is None:
            apply_outlier_detection = self.outlier_config['handling'] != 'none'
        
        # PHASE 2 ENHANCEMENT: Enhanced outlier detection and weighted standardization
        print("Applying strata-specific standardization with enhanced outlier detection...")
        processed_df_list = []
        
        # Track outlier detection statistics
        outlier_stats = {
            'total_sessions': 0,
            'total_outliers_detected': 0,
            'outliers_by_feature': {},
            'outliers_by_strata': {}
        }
        
        # Process each strata separately
        for strata, strata_df in df_with_strata.groupby('strata'):
            print(f"  Processing strata '{strata}' with {len(strata_df)} sessions")
            strata_processed = strata_df.copy()
            
            # Initialize outlier weights for this strata (default to 1.0 = no outlier detected)
            strata_processed['outlier_weight'] = 1.0
            strata_outlier_count = 0
            
            # Check if we have enough subjects for meaningful standardization
            unique_subjects = strata_df['subject_id'].nunique()
            min_subjects_for_standardization = 5  # Minimum subjects needed for standardization
            
            if unique_subjects >= min_subjects_for_standardization:
                # Apply enhanced outlier detection and standardization within this strata
                for feature, lower_is_better in self.features_config.items():
                    if feature not in strata_df.columns:
                        continue
                    
                    # Get non-null feature values for this strata
                    feature_values = strata_df[feature].dropna()
                    
                    if len(feature_values) < 3:  # Need minimum data points for standardization
                        print(f"    Skipping {feature} - insufficient data points ({len(feature_values)})")
                        continue
                    
                    # PHASE 2: Apply outlier detection before standardization
                    if apply_outlier_detection and len(feature_values) >= self.outlier_config['min_data_points']:
                        print(f"    Applying {self.outlier_config['method']} outlier detection to {feature}")
                        
                        # Get outlier detection results
                        feature_values_array = strata_df[feature].values
                        outlier_mask, feature_weights = self._detect_outliers(feature_values_array)
                        
                        # Update outlier weights for detected outliers
                        outlier_indices = strata_df.index[outlier_mask]
                        strata_processed.loc[outlier_indices, 'outlier_weight'] = feature_weights[outlier_mask]
                        
                        # Track outlier statistics
                        feature_outlier_count = np.sum(outlier_mask)
                        strata_outlier_count += feature_outlier_count
                        
                        if feature not in outlier_stats['outliers_by_feature']:
                            outlier_stats['outliers_by_feature'][feature] = 0
                        outlier_stats['outliers_by_feature'][feature] += feature_outlier_count
                        
                        print(f"      Detected {feature_outlier_count} outliers ({feature_outlier_count/len(feature_values)*100:.1f}%)")
                    
                    # Create and fit scaler for this strata and feature
                    scaler = StandardScaler()
                    
                    try:
                        # Fit on all non-null values in this strata
                        scaler.fit(feature_values.values.reshape(-1, 1))
                        
                        # Transform all values (including nulls which will remain null)
                        scaled_values = np.full(len(strata_df), np.nan)
                        non_null_mask = strata_df[feature].notna()
                        
                        if non_null_mask.any():
                            scaled_values[non_null_mask] = scaler.transform(
                                strata_df.loc[non_null_mask, feature].values.reshape(-1, 1)
                            ).flatten()
                        
                        # Invert if lower is better (higher values always mean better performance)
                        if lower_is_better:
                            scaled_values = -scaled_values
                        
                        # Add processed feature to strata dataframe
                        strata_processed[f'{feature}_processed'] = scaled_values
                        
                        print(f"    Standardized {feature}: mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}, inverted={lower_is_better}")
                        
                    except Exception as e:
                        print(f"    Error standardizing {feature} in strata {strata}: {e}")
                        # If standardization fails, copy original values
                        strata_processed[f'{feature}_processed'] = strata_df[feature]
            
            else:
                print(f"    Insufficient subjects ({unique_subjects} < {min_subjects_for_standardization}) - using raw values")
                # If insufficient subjects, just copy raw values as "processed"
                for feature, lower_is_better in self.features_config.items():
                    if feature in strata_df.columns:
                        # Still apply inversion if needed
                        if lower_is_better:
                            strata_processed[f'{feature}_processed'] = -strata_df[feature]
                        else:
                            strata_processed[f'{feature}_processed'] = strata_df[feature]
            
            # Track strata outlier statistics
            outlier_stats['outliers_by_strata'][strata] = strata_outlier_count
            outlier_stats['total_sessions'] += len(strata_df)
            outlier_stats['total_outliers_detected'] += strata_outlier_count
            
            processed_df_list.append(strata_processed)
        
        # Combine all processed strata back together
        if processed_df_list:
            df_final = pd.concat(processed_df_list, ignore_index=True)
            print(f"Combined {len(processed_df_list)} strata into final dataset with {len(df_final)} sessions")
        else:
            print("No valid strata found - returning empty dataframe")
            return pd.DataFrame()

        # Report outlier detection statistics
        if apply_outlier_detection and outlier_stats['total_sessions'] > 0:
            outlier_rate = (outlier_stats['total_outliers_detected'] / outlier_stats['total_sessions']) * 100
            print(f"\n📊 PHASE 2 Outlier Detection Results:")
            print(f"  Method: {self.outlier_config['method']}")
            print(f"  Total sessions processed: {outlier_stats['total_sessions']}")
            print(f"  Total outliers detected: {outlier_stats['total_outliers_detected']} ({outlier_rate:.1f}%)")
            print(f"  Handling: {self.outlier_config['handling']} (weight={self.outlier_config['outlier_weight']})")
            
            # Feature-specific outlier rates
            print(f"  Outliers by feature:")
            for feature, count in outlier_stats['outliers_by_feature'].items():
                feature_rate = (count / outlier_stats['total_sessions']) * 100
                print(f"    {feature}: {count} ({feature_rate:.1f}%)")
            
            # Strata-specific outlier counts
            print(f"  Outliers by strata:")
            for strata, count in outlier_stats['outliers_by_strata'].items():
                print(f"    {strata}: {count}")
            print()

        # PHASE 2: Remove old 3-sigma outlier removal (replaced with weighted approach above)
        # The old remove_outliers logic has been replaced with the enhanced outlier detection

        return df_final

    def _detect_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using the configured method
        
        Parameters:
            data: np.ndarray
                Feature data for outlier detection
                
        Returns:
            Tuple[np.ndarray, np.ndarray]
                (outlier_mask, weights) where outlier_mask is boolean array
                and weights are suggested weights for each data point
        """
        method = self.outlier_config['method']
        
        if method == 'iqr':
            return self.statistical_utils.detect_outliers_iqr(
                data, 
                factor=self.outlier_config['factor']
            )
        elif method == 'modified_zscore':
            return self.statistical_utils.detect_outliers_modified_zscore(
                data, 
                threshold=self.outlier_config.get('threshold', 3.5)
            )
        elif method == 'none':
            # No outlier detection - return all normal weights
            return np.zeros(len(data), dtype=bool), np.ones(len(data))
        else:
            print(f"Warning: Unknown outlier detection method '{method}', using IQR")
            return self.statistical_utils.detect_outliers_iqr(data, factor=1.5)

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
        
        # Handle curriculum name - find the index where the stage info starts
        try:
            stage_start = next(i for i, part in enumerate(parts) if 'STAGE' in part or 'GRADUATED' in part)
            curriculum = '_'.join(parts[:stage_start])
            stage_parts = parts[stage_start:]  # Get all parts after curriculum name
        except StopIteration:
            # No STAGE or GRADUATED found, return simplified format with UNKNOWN stage
            print(f"Warning: No STAGE or GRADUATED found in strata_id: {strat_id}")
            if len(parts) >= 2:
                curriculum = '_'.join(parts[:-1])  # All but last part as curriculum
                version = parts[-1]  # Last part as version
                return f"{curriculum}_UNKNOWN_{version}"
            else:
                return f"{strat_id}_UNKNOWN_v1"  # Default fallback
        
        # Get the full stage name and version
        if len(stage_parts) >= 2:
            stage = '_'.join(stage_parts[:-1])  # Join all parts except the last (version)
            version = stage_parts[-1]  # Keep the original version (v1, v2, v3)
        else:
            stage = stage_parts[0] if stage_parts else 'UNKNOWN'
            version = 'v1'  # Default version

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
    
    def prepare_session_level_data(self, df: pd.DataFrame, decay_factor: float = 0.9) -> pd.DataFrame:
        """
        Enhanced session-level data preparation for unified percentile calculation:
        1. Assign strata to each session
        2. Calculate rolling weighted averages for each session
        3. Add metadata for strata continuity tracking
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with preprocessed data
            decay_factor: float
                Factor for exponential decay weighting (0-1) (higher == more weight on recent sessions)
                
        Returns:
            pd.DataFrame
                Enhanced DataFrame with session-level rolling averages and metadata
        """
        # Assign strata to each session
        stratified_df = self.assign_subject_strata(df)
        
        # Calculate rolling weighted averages for each session
        session_level_data = self.calculate_session_level_rolling_averages(stratified_df, decay_factor)
        
        # Add session_index (sequential number for each subject-strata combination)
        session_level_data['session_index'] = session_level_data.groupby(['subject_id', 'strata']).cumcount() + 1
        
        # Add metadata for strata transitions
        session_level_data = self._add_strata_transition_metadata(session_level_data)
        
        # Add is_current_strata flag by identifying most recent strata for each subject
        latest_strata = stratified_df.sort_values('session_date').groupby('subject_id')['strata'].last()
        session_level_data['is_current_strata'] = session_level_data.apply(
            lambda row: row['strata'] == latest_strata.get(row['subject_id'], ''), 
            axis=1
        )
        
        # Add is_last_session flag for last session in each subject-strata combination
        # This will be useful for validation against strata-level calculations
        max_indices = session_level_data.groupby(['subject_id', 'strata'])['session_index'].transform('max')
        session_level_data['is_last_session'] = session_level_data['session_index'] == max_indices
        
        return session_level_data

    def _add_strata_transition_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata about strata transitions to track when subjects move between strata
        
        Parameters:
            df: pd.DataFrame
                DataFrame with session-level data
                
        Returns:
            pd.DataFrame
                DataFrame with strata transition metadata
        """
        result_df = df.copy()
        
        # Sort by subject and date
        result_df = result_df.sort_values(['subject_id', 'session_date'])
        
        # Initialize strata transition columns
        result_df['strata_transition'] = False
        result_df['previous_strata'] = None
        
        # Process each subject
        for subject_id, subject_data in result_df.groupby('subject_id'):
            # Skip if only one session
            if len(subject_data) <= 1:
                continue
                
            # Get indices in original dataframe
            indices = subject_data.index
            
            # Track strata changes
            previous_strata = None
            
            for i, idx in enumerate(indices):
                current_strata = result_df.loc[idx, 'strata']
                
                if i > 0 and current_strata != previous_strata:
                    # Mark strata transition
                    result_df.loc[idx, 'strata_transition'] = True
                    result_df.loc[idx, 'previous_strata'] = previous_strata
                
                previous_strata = current_strata
        
        return result_df