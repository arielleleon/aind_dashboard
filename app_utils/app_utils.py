from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer
from .app_analysis.overall_percentile_calculator import OverallPercentileCalculator
from .app_alerts import AlertService
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class AppUtils:
    """
    Central utility class for app
    Access to data loading and computation functions
    """

    def __init__(self):
        """
        Initialize app utils
        """
        self.data_loader = AppLoadData()
        self.reference_processor = None
        self.quantile_analyzer = None
        self.alert_service = None
        self.threshold_analyzer = None
        self.percentile_calculator = OverallPercentileCalculator()
        
        # Simplified cache for processed data
        self._cache = {
            'raw_data': None,
            'session_level_data': None,
            'optimized_storage': None,
            'ui_structures': None,
            'unified_alerts': None,
            'last_process_time': None,
            'data_hash': None
        }

    def get_session_data(self, load_bpod = False, use_cache = True):
        """
        Get session data from the data loader
        
        Parameters:
            load_bpod (bool): Whether to load bpod data
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Session data
        """
        if use_cache and self._cache['raw_data'] is not None:
            return self._cache['raw_data']
            
        if load_bpod:
            data = self.data_loader.load(load_bpod = True)
        else:
            data = self.data_loader.get_data()
            
        # Cache the raw data
        self._cache['raw_data'] = data
        
        # Reset other caches since raw data has changed
        self._invalidate_derived_caches()
        
        return data
    
    def _invalidate_derived_caches(self):
        """Reset all derived data caches when raw data changes"""
        self._cache['session_level_data'] = None
        self._cache['optimized_storage'] = None  # Optimized storage cache
        self._cache['ui_structures'] = None      # UI-optimized structures cache
        self._cache['unified_alerts'] = None
        self._cache['last_process_time'] = None
        self._cache['data_hash'] = None
        
        # Also clear percentile calculator cache
        if hasattr(self, 'percentile_calculator'):
            self.percentile_calculator.clear_cache()
    
    def reload_data(self, load_bpod = False):
        """
        Force reload session data
        """
        data = self.data_loader.load(load_bpod = load_bpod)
        
        # Update cache and invalidate derived caches
        self._cache['raw_data'] = data
        self._invalidate_derived_caches()
        
        return data
    
    def initialize_reference_processor(self, features_config, min_sessions = 5, min_days = 7, outlier_config = None):
        """
        Initialize reference processor with enhanced outlier detection support

        Parameters:
            features_config (Dict[str, bool]): Configuration of features (feature_name: higher or lower better)
            min_sessions (int): Minimum number of sessions required for eligibility
            min_days (int): Minimum number of days required for eligibility
            outlier_config (Dict[str, Any], optional): Outlier detection configuration

        Returns: 
            ReferenceProcessor: Initialized reference processor
        """
        # PHASE 2: Default outlier configuration for enhanced robustness
        if outlier_config is None:
            outlier_config = {
                'method': 'iqr',  # Use IQR method for better outlier detection
                'factor': 1.5,    # Standard IQR multiplier
                'handling': 'weighted',  # Use weighted approach instead of removal
                'outlier_weight': 0.5,   # Outliers get half weight
                'min_data_points': 4     # Minimum points needed for outlier detection
            }
        
        self.reference_processor = ReferenceProcessor(
            features_config = features_config,
            min_sessions = min_sessions,
            min_days = min_days,
            outlier_config = outlier_config
        )
        
        print(f"📊 PHASE 2: Reference processor initialized with enhanced outlier detection")
        print(f"   Method: {outlier_config['method']}")
        print(f"   Handling: {outlier_config['handling']} (weight: {outlier_config['outlier_weight']})")
        
        return self.reference_processor
    
    def initialize_quantile_analyzer(self, stratified_data):
        """
        Initialize quantile analyzer

        Parameters:
            stratified_data (Dict[str, pd.DataFrame]): Dictionary of stratified data

        Returns:
            QuantileAnalyzer: Initialized quantile analyzer
        """
        self.quantile_analyzer = QuantileAnalyzer(
            stratified_data = stratified_data,
            historical_data = getattr(self, 'historical_data', None)
        )
        return self.quantile_analyzer
    
    def get_subject_percentiles(self, subject_id):
        """
        Get percentile data for specific subject

        Parameters:
            subject_id (str): subject ID to get percentile data for

        Returns:
            pd.DataFrame: Dataframe with percentile data for the subject
        """
        if self.quantile_analyzer is None:
            raise ValueError("Quantile analyzer not initialized. Process data first.")
        
        return self.quantile_analyzer.get_subject_history(subject_id)
    
    def get_session_overall_percentiles(self, subject_ids=None, use_cache=True, feature_weights=None):
        """
        Get overall percentile scores for subjects using the session-level pipeline
        
        Parameters:
            subject_ids (List[str], optional): List of specific subjects to calculate for
            use_cache (bool): Whether to use cached results if available
            feature_weights (Dict[str, float], optional): Optional weights for features

        Returns:
            pd.DataFrame: DataFrame with session-level overall percentile scores
        """
        # Get session-level data
        if use_cache and self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Filter for specific subjects if requested
        if subject_ids is not None:
            session_data = session_data[session_data['subject_id'].isin(subject_ids)]
        
        # Get the most recent session for each subject (for summary views)
        most_recent = session_data.sort_values(['subject_id', 'session_date']).groupby('subject_id').last().reset_index()
        
        return most_recent
    
    def initialize_alert_service(self, config: Optional[Dict[str, Any]] = None) -> AlertService:
        """
        Initialize alert service for monitoring and reporting issues
        
        Parameters:
            config (Optional[Dict[str, Any]]): Configuration for alert service
        
        Returns:
            AlertService: Initialized alert service
        """
        # Create alert service with access to this AppUtils instance
        self.alert_service = AlertService(app_utils=self, config=config)
        
        # Force reset caches for a clean start
        if hasattr(self.alert_service, 'force_reset'):
            self.alert_service.force_reset()
        
        return self.alert_service
    

    def get_quantile_alerts(self, subject_ids=None):
        """
        Get quantile alerts for given subjects
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        return self.alert_service.get_quantile_alerts(subject_ids)

    def get_alerts(self, subject_ids=None):
        """
        Alias for get_quantile_alerts for backward compatibility
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        return self.get_quantile_alerts(subject_ids)

    def initialize_threshold_analyzer(self, threshold_config: Optional[Dict[str, Any]] = None) -> ThresholdAnalyzer:
        """
        Initialize threshold analyzer
        
        Parameters:
            threshold_config: Optional[Dict[str, Any]]
                Configuration for threshold alerts
                
        Returns:
            ThresholdAnalyzer: Initialized threshold analyzer
        """
        
        self.threshold_analyzer = ThresholdAnalyzer(threshold_config)
        return self.threshold_analyzer

    def get_unified_alerts(self, subject_ids = None, use_cache = True):
        """ 
        Get unified alerts
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
            use_cache (bool): Whether to use cached alerts if available
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their unified alerts
        """
        # Return cached alerts if available and not requesting specific subjects
        if use_cache and subject_ids is None and self._cache['unified_alerts'] is not None:
            print("Using cached unified alerts")
            return self._cache['unified_alerts']
            
        if self.alert_service is None:
            self.initialize_alert_service()

        alerts = self.alert_service.get_unified_alerts(subject_ids)
        
        # Cache results if getting alerts for all subjects
        if subject_ids is None:
            self._cache['unified_alerts'] = alerts
            
        return alerts
        
    def get_formatted_data(self, window_days=30, reference_date=None, use_cache=True):
        """
        Get formatted data for display using the unified session-level pipeline
        
        Parameters:
            window_days (int): Number of days to include in sliding window
            reference_date (datetime, optional): Reference date for sliding window
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Formatted data for display
        """
        # Get all session data from the unified pipeline
        if use_cache and self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Apply time window filter
        if reference_date is None:
            reference_date = session_data['session_date'].max()
        
        start_date = reference_date - timedelta(days=window_days)
        time_filtered_df = session_data[session_data['session_date'] >= start_date]
        
        # Get most recent session for each subject in the window
        time_filtered_df = time_filtered_df.sort_values('session_date', ascending=False)
        result_df = time_filtered_df.drop_duplicates(subset=['subject_id'], keep='first')
        
        print(f"Applied {window_days} day window filter: {len(result_df)} subjects")
        return result_df
    
    def get_subject_sessions(self, subject_id):
        """
        Get all sessions for a specific subject
        
        Parameters:
        -----------
        subject_id : str
            The subject ID to retrieve sessions for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing all sessions for the subject
        """
        try:
            # Get all data
            all_data = self.data_loader.get_data()
            
            # Filter for specific subject
            subject_data = all_data[all_data['subject_id'] == subject_id].copy()
            
            if subject_data.empty:
                print(f"No sessions found for subject {subject_id}")
                return None
                
            # Sort by session date (most recent first)
            subject_data = subject_data.sort_values('session_date', ascending=False)
            
            return subject_data
        except Exception as e:
            print(f"Error getting sessions for subject {subject_id}: {str(e)}")
            return None
    
    def process_data_pipeline(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        NEW UNIFIED PIPELINE: Process raw session data through the complete session-level pipeline
        
        This replaces the previous dual approach (strata-level + session-level) with a unified
        session-level approach that provides all necessary metrics for the application.
        
        Parameters:
            df: pd.DataFrame
                Raw session data from the data loader
            use_cache: bool
                Whether to use cached results if available
                
        Returns:
            pd.DataFrame
                Complete session-level data with percentiles, rolling averages, and metadata
        """
        print(f"Starting unified data processing pipeline with {len(df)} sessions")
        
        # Check cache first
        if use_cache and self._cache['session_level_data'] is not None:
            print("Using cached session-level data")
            return self._cache['session_level_data']
        
        # Step 1: Initialize processors if needed
        if self.reference_processor is None:
            features_config = {
                'finished_trials': False,  # Higher is better
                'ignore_rate': True,     # Lower is better
                'total_trials': False,   # Higher is better
                'foraging_performance': False,   # Higher is better
                'abs(bias_naive)': True  # Lower is better 
            }
            # PHASE 2: Initialize with enhanced outlier detection
            outlier_config = {
                'method': 'iqr',
                'factor': 1.5,
                'handling': 'weighted',
                'outlier_weight': 0.5,
                'min_data_points': 4
            }
            self.initialize_reference_processor(features_config, min_sessions=1, min_days=1, outlier_config=outlier_config)
        
        # Step 2: Get eligible subjects and preprocess data with enhanced outlier detection
        eligible_subjects = self.reference_processor.get_eligible_subjects(df)
        eligible_df = df[df['subject_id'].isin(eligible_subjects)]
        print(f"Got {len(eligible_subjects)} eligible subjects")
        
        # PHASE 2: Enhanced preprocessing with outlier detection
        processed_df = self.reference_processor.preprocess_data(eligible_df, remove_outliers=True)
        print(f"Preprocessed data with Phase 2 enhancements: {len(processed_df)} sessions")
        
        # Report outlier weights if present
        if 'outlier_weight' in processed_df.columns:
            outlier_sessions = (processed_df['outlier_weight'] < 1.0).sum()
            total_sessions = len(processed_df)
            outlier_rate = (outlier_sessions / total_sessions) * 100
            print(f"📊 PHASE 2 Results: {outlier_sessions}/{total_sessions} sessions ({outlier_rate:.1f}%) have outlier weights")
        
        # Step 3: Prepare session-level data with rolling averages and strata assignments
        session_level_data = self.reference_processor.prepare_session_level_data(processed_df)
        print(f"Prepared session-level data: {len(session_level_data)} sessions")
        
        # Step 4: Calculate reference distributions for percentile calculation
        # We still need reference distributions, but now they're used for session-level percentiles
        if self.quantile_analyzer is None:
            # Create reference distributions using the current approach but for session percentiles
            stratified_data = self.reference_processor.prepare_for_quantile_analysis(
                processed_df, include_history=True
            )
            self.initialize_quantile_analyzer(stratified_data)
            print(f"Initialized quantile analyzer with {len(stratified_data)} strata")
        
        # Step 5: Calculate session-level percentiles using reference distributions
        session_with_percentiles = self.quantile_analyzer.calculate_session_level_percentiles(session_level_data)
        print(f"Calculated session-level percentiles")
        
        # Step 6: Calculate overall percentiles for each session
        comprehensive_data = self.percentile_calculator.calculate_session_overall_percentile(
            session_with_percentiles
        )
        print(f"Calculated overall session percentiles")
        
        # Step 6.5: Calculate overall rolling averages for hover information
        comprehensive_data = self.percentile_calculator.calculate_session_overall_rolling_average(
            comprehensive_data
        )
        print(f"Calculated overall session rolling averages")
        
        # Step 7: Add alerts and metadata
        comprehensive_data = self._add_session_metadata(comprehensive_data)
        
        # Cache the results
        self._cache['session_level_data'] = comprehensive_data
        self._cache['last_process_time'] = datetime.now()
        
        # Create optimized storage structure for fast lookups
        optimized_storage = self.optimize_session_data_storage(comprehensive_data)
        self._cache['optimized_storage'] = optimized_storage
        
        # Create UI-optimized structures for fast component rendering
        ui_structures = self.create_ui_optimized_structures(comprehensive_data)
        self._cache['ui_structures'] = ui_structures
        
        print(f"Unified pipeline complete: {len(comprehensive_data)} sessions processed")
        print(f"Optimized storage created with {len(optimized_storage['subjects'])} subjects")
        print(f"UI structures created for {len(ui_structures['feature_rank_data'])} subjects")
        return comprehensive_data
    
    def _add_session_metadata(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata to session-level data for easier consumption by UI components
        
        Parameters:
            session_data: pd.DataFrame
                Session data with percentiles
                
        Returns:
            pd.DataFrame
                Enhanced session data with metadata
        """
        result_df = session_data.copy()
        
        # Add percentile categories for each feature
        feature_list = list(self.reference_processor.features_config.keys())
        
        for feature in feature_list:
            percentile_col = f"{feature}_session_percentile"
            category_col = f"{feature}_category"
            
            if percentile_col in result_df.columns:
                # Map percentiles to categories using the same logic as alerts
                result_df[category_col] = result_df[percentile_col].apply(
                    lambda x: self._map_percentile_to_category(x) if not pd.isna(x) else 'NS'
                )
        
        # Add overall percentile category
        if 'session_overall_percentile' in result_df.columns:
            result_df['overall_percentile_category'] = result_df['session_overall_percentile'].apply(
                lambda x: self._map_percentile_to_category(x) if not pd.isna(x) else 'NS'
            )
        
        # PHASE 2: Add simple boolean outlier flag based on outlier_weight
        if 'outlier_weight' in result_df.columns:
            result_df['is_outlier'] = result_df['outlier_weight'] < 1.0
        else:
            result_df['is_outlier'] = False  # Default if no outlier detection applied
        
        return result_df
    
    def _map_percentile_to_category(self, percentile: float) -> str:
        """
        Map percentile value to alert category
        
        Parameters:
            percentile: float
                Percentile value (0-100)
                
        Returns:
            str: Alert category (SB, B, N, G, SG)
        """
        if pd.isna(percentile):
            return 'NS'
        
        # Use the correct thresholds from the alert service
        if percentile < 6.5:
            return 'SB'  # Severely Below: < 6.5%
        elif percentile < 28:
            return 'B'   # Below: < 28%
        elif percentile <= 72:
            return 'N'   # Normal: 28% - 72%
        elif percentile <= 93.5:
            return 'G'   # Good: 72% - 93.5%
        else:
            return 'SG'  # Severely Good: > 93.5%

    def get_subject_session_level_percentiles(self, subject_id: str) -> pd.DataFrame:
        """
        Get session-level percentiles for a specific subject using the unified pipeline
        
        Parameters:
            subject_id: str
                The subject ID to get session-level percentiles for
                
        Returns:
            pd.DataFrame
                DataFrame containing session-level percentiles for the subject
        """
        # Get all session data from the unified pipeline
        if self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Filter for the requested subject
        subject_data = session_data[session_data['subject_id'] == subject_id]
        
        if subject_data.empty:
            print(f"No session-level data found for subject {subject_id}")
            return pd.DataFrame()
        
        # Sort by session date or index
        if 'session_date' in subject_data.columns:
            subject_data = subject_data.sort_values('session_date')
        elif 'session_index' in subject_data.columns:
            subject_data = subject_data.sort_values('session_index')
        
        return subject_data

    def get_most_recent_subject_sessions(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get the most recent session for each subject with all session-level metrics
        
        Parameters:
            use_cache: bool
                Whether to use cached data if available
                
        Returns:
            pd.DataFrame
                DataFrame with most recent session for each subject
        """
        # Get all session data from the unified pipeline
        if use_cache and self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Sort by subject ID and session date (descending)
        session_data = session_data.sort_values(['subject_id', 'session_date'], ascending=[True, False])
        
        # Get most recent session for each subject
        most_recent = session_data.groupby('subject_id').first().reset_index()
        
        return most_recent

    def optimize_session_data_storage(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize session-level data storage for efficient lookup and memory usage
        
        This creates optimized data structures for:
        1. Subject-indexed session data
        2. Strata-indexed reference distributions  
        3. Compressed historical data
        
        Parameters:
            session_data: pd.DataFrame
                Complete session-level data from unified pipeline
                
        Returns:
            Dict[str, Any]: Optimized storage structure
        """
        print("Optimizing session data storage...")
        
        # Create subject-indexed storage for fast subject lookups
        subject_data = {}
        strata_reference = {}
        
        # Group by subject for efficient subject-based operations
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            # Sort sessions by date
            subject_sessions = subject_sessions.sort_values('session_date')
            
            # Store only essential columns to save memory
            essential_columns = [
                'subject_id', 'session_date', 'session', 'strata', 'session_index',
                'session_overall_percentile', 'overall_percentile_category',
                'session_overall_rolling_avg',  # Add overall rolling average for hover info
                'is_current_strata', 'is_last_session',
                # PHASE 2: Add outlier detection information
                'outlier_weight',  # Phase 2 outlier weight (0.5 for outliers, 1.0 for normal)
                'is_outlier',      # Simple boolean flag for outlier status
                # CRITICAL FIX: Add essential metadata columns
                'PI', 'trainer', 'rig', 'current_stage_actual', 'curriculum_name',
                'water_day_total', 'base_weight', 'target_weight', 'weight_after',
                'total_trials', 'finished_trials', 'ignore_rate', 'foraging_performance',
                'abs(bias_naive)', 'finished_rate'
            ]
            
            # Add feature-specific columns
            feature_columns = [col for col in subject_sessions.columns 
                             if col.endswith(('_session_percentile', '_category', '_processed_rolling_avg'))]
            essential_columns.extend(feature_columns)
            
            # Filter to available columns and ensure uniqueness
            available_columns = [col for col in essential_columns if col in subject_sessions.columns]
            # Remove duplicates while preserving order
            unique_columns = []
            seen = set()
            for col in available_columns:
                if col not in seen:
                    unique_columns.append(col)
                    seen.add(col)
            
            # Store compressed subject data
            subject_data[subject_id] = {
                'sessions': subject_sessions[unique_columns].to_dict('records'),
                'current_strata': subject_sessions['strata'].iloc[-1],
                'total_sessions': len(subject_sessions),
                'first_session_date': subject_sessions['session_date'].min(),
                'last_session_date': subject_sessions['session_date'].max(),
                'strata_history': subject_sessions[['strata', 'session_date']].drop_duplicates('strata').to_dict('records')
            }
        
        # Create strata-indexed reference distributions for percentile calculations
        for strata, strata_sessions in session_data.groupby('strata'):
            # Store only the reference distribution data needed for percentile calculations
            processed_features = [col for col in strata_sessions.columns if col.endswith('_processed_rolling_avg')]
            
            if processed_features:
                reference_data = strata_sessions[processed_features + ['subject_id']].dropna()
                
                strata_reference[strata] = {
                    'subject_count': len(reference_data),
                    'session_count': len(strata_sessions),
                    'reference_distributions': {
                        feature: reference_data[feature].values.tolist() 
                        for feature in processed_features
                        if not reference_data[feature].isna().all()
                    }
                }
        
        # Create optimized storage structure
        optimized_storage = {
            'subjects': subject_data,
            'strata_reference': strata_reference,
            'metadata': {
                'total_subjects': len(subject_data),
                'total_sessions': len(session_data),
                'total_strata': len(strata_reference),
                'storage_timestamp': pd.Timestamp.now(),
                'data_hash': self._calculate_data_hash(session_data)
            }
        }
        
        print(f"Optimized storage created:")
        print(f"  - {len(subject_data)} subjects")
        print(f"  - {len(strata_reference)} strata references")
        print(f"  - {len(session_data)} total sessions")
        
        return optimized_storage
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash for data validation"""
        import hashlib
        data_str = f"{len(df)}_{df['subject_id'].nunique()}_{df['session_date'].max()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def get_subject_optimized_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized subject data using the new storage format
        
        Parameters:
            subject_id: str
                Subject ID to retrieve data for
            use_cache: bool
                Whether to use cached optimized storage
                
        Returns:
            Dict[str, Any]: Optimized subject data
        """
        # Check if we have optimized storage cached
        if use_cache and 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            return optimized_storage['subjects'].get(subject_id, {})
        
        # If not cached, create optimized storage
        if self._cache['session_level_data'] is not None:
            optimized_storage = self.optimize_session_data_storage(self._cache['session_level_data'])
            self._cache['optimized_storage'] = optimized_storage
            return optimized_storage['subjects'].get(subject_id, {})
        
        # Fallback to processing data
        raw_data = self.get_session_data(use_cache=True)
        session_data = self.process_data_pipeline(raw_data, use_cache=True)
        optimized_storage = self.optimize_session_data_storage(session_data)
        self._cache['optimized_storage'] = optimized_storage
        
        return optimized_storage['subjects'].get(subject_id, {})
    
    def get_strata_reference_data(self, strata: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get reference distribution data for a specific strata
        
        Parameters:
            strata: str
                Strata to get reference data for
            use_cache: bool
                Whether to use cached data
                
        Returns:
            Dict[str, Any]: Reference distribution data
        """
        # Check for optimized storage
        if use_cache and 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            return optimized_storage['strata_reference'].get(strata, {})
        
        # Create optimized storage if needed
        if self._cache['session_level_data'] is not None:
            optimized_storage = self.optimize_session_data_storage(self._cache['session_level_data'])
            self._cache['optimized_storage'] = optimized_storage
            return optimized_storage['strata_reference'].get(strata, {})
        
        return {}

    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current storage optimization
        
        Returns:
            Dict[str, Any]: Storage summary statistics
        """
        summary = {
            'raw_data_cached': self._cache['raw_data'] is not None,
            'session_level_cached': self._cache['session_level_data'] is not None,
            'optimized_storage_cached': 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None,
            'formatted_data_cached': self._cache['formatted_data'] is not None
        }
        
        # Add size information if available
        if summary['optimized_storage_cached']:
            storage = self._cache['optimized_storage']
            summary.update({
                'subjects_count': len(storage['subjects']),
                'strata_count': len(storage['strata_reference']),
                'total_sessions': storage['metadata']['total_sessions'],
                'storage_timestamp': storage['metadata']['storage_timestamp']
            })
        
        if summary['session_level_cached']:
            summary['session_level_rows'] = len(self._cache['session_level_data'])
        
        if summary['formatted_data_cached']:
            summary['formatted_data_rows'] = len(self._cache['formatted_data'])
        
        return summary

    def create_ui_optimized_structures(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create UI-optimized data structures for fast component rendering
        
        Specialized structures for:
        1. Feature rank plot data
        2. Subject detail views
        3. Table display optimization
        4. Time series visualization
        
        Parameters:
            session_data: pd.DataFrame
                Complete session-level data from unified pipeline
                
        Returns:
            Dict[str, Any]: UI-optimized data structures
        """
        print("Creating UI-optimized data structures...")
        
        # Import pandas for data checking
        import pandas as pd
        
        # DEBUG: Check what columns are available in session_data
        print(f"📊 Session data columns ({len(session_data.columns)} total):")
        session_percentile_cols = [col for col in session_data.columns if col.endswith('_session_percentile')]
        rolling_avg_cols = [col for col in session_data.columns if col.endswith('_rolling_avg')]
        category_cols = [col for col in session_data.columns if col.endswith('_category')]
        overall_cols = [col for col in session_data.columns if 'overall_percentile' in col]
        
        print(f"  Session percentile columns ({len(session_percentile_cols)}): {session_percentile_cols}")
        print(f"  Rolling average columns ({len(rolling_avg_cols)}): {rolling_avg_cols}")
        print(f"  Category columns ({len(category_cols)}): {category_cols}")
        print(f"  Overall percentile columns ({len(overall_cols)}): {overall_cols}")
        
        # Check if we have sample data with percentiles
        if len(session_data) > 0:
            print(f"  Sample data for first row:")
            sample_cols = session_percentile_cols[:2] + ['session_overall_percentile']
            for col in sample_cols:
                if col in session_data.columns:
                    value = session_data.iloc[0][col]
                    print(f"    {col}: {value}")
        print("")  # Empty line for readability
        
        ui_structures = {
            'feature_rank_data': {},
            'subject_lookup': {},
            'strata_lookup': {},
            'time_series_data': {},
            'table_display_cache': {}
        }
        
        # 1. Feature Rank Plot Optimization
        # Pre-compute feature ranking data for each subject
        features = ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)']
        
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            # Get most recent session data
            latest_session = subject_sessions.sort_values('session_date').iloc[-1]
            
            feature_ranks = {}
            for feature in features:
                session_percentile_col = f"{feature}_session_percentile"
                category_col = f"{feature}_category"
                
                if session_percentile_col in latest_session:
                    feature_ranks[feature] = {
                        'percentile': latest_session[session_percentile_col],
                        'category': latest_session.get(category_col, 'NS'),
                        'value': latest_session.get(feature, None)
                    }
            
            ui_structures['feature_rank_data'][subject_id] = {
                'features': feature_ranks,
                'overall_percentile': latest_session.get('session_overall_percentile'),
                'overall_category': latest_session.get('overall_percentile_category', 'NS'),
                'strata': latest_session.get('strata', 'Unknown'),
                'session_date': latest_session.get('session_date'),
                'session_count': latest_session.get('session', 0)
            }
        
        # 2. Subject Detail Lookup Optimization
        # Create fast subject lookup with essential display data
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            subject_sessions = subject_sessions.sort_values('session_date')
            latest_session = subject_sessions.iloc[-1]
            
            ui_structures['subject_lookup'][subject_id] = {
                'latest': {
                    'session_date': latest_session['session_date'],
                    'session': latest_session['session'],
                    'strata': latest_session['strata'],
                    'overall_percentile': latest_session.get('session_overall_percentile'),
                    'overall_category': latest_session.get('overall_percentile_category', 'NS'),
                    'PI': latest_session.get('PI', 'N/A'),
                    'trainer': latest_session.get('trainer', 'N/A'),
                    'rig': latest_session.get('rig', 'N/A')
                },
                'summary': {
                    'total_sessions': len(subject_sessions),
                    'first_session_date': subject_sessions['session_date'].min(),
                    'last_session_date': subject_sessions['session_date'].max(),
                    'unique_strata': subject_sessions['strata'].nunique(),
                    'current_strata': latest_session['strata']
                }
            }
        
        # 3. Strata Lookup Optimization
        # Pre-compute strata summaries for filtering
        for strata, strata_sessions in session_data.groupby('strata'):
            unique_subjects = strata_sessions['subject_id'].nunique()
            total_sessions = len(strata_sessions)
            
            # Calculate strata performance metrics
            overall_percentiles = strata_sessions['session_overall_percentile'].dropna()
            
            ui_structures['strata_lookup'][strata] = {
                'subject_count': unique_subjects,
                'session_count': total_sessions,
                'avg_performance': overall_percentiles.mean() if len(overall_percentiles) > 0 else None,
                'performance_std': overall_percentiles.std() if len(overall_percentiles) > 0 else None,
                'subjects': strata_sessions['subject_id'].unique().tolist()
            }
        
        # 4. Time Series Data Optimization
        # Pre-compute time series data for subjects with compressed format
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            subject_sessions = subject_sessions.sort_values('session_date')
            
            # Extract time series data in compressed format
            time_series = {
                'sessions': subject_sessions['session'].tolist(),
                'dates': subject_sessions['session_date'].dt.strftime('%Y-%m-%d').tolist(),
                'overall_percentiles': subject_sessions['session_overall_percentile'].fillna(-1).tolist(),
                'overall_rolling_avg': subject_sessions['session_overall_rolling_avg'].fillna(-1).tolist(),
                'strata': subject_sessions['strata'].tolist()
            }
            
            # Add confidence intervals for overall percentiles
            if 'session_overall_percentile_ci_lower' in subject_sessions.columns:
                time_series['overall_percentiles_ci_lower'] = subject_sessions['session_overall_percentile_ci_lower'].fillna(-1).tolist()
                time_series['overall_percentiles_ci_upper'] = subject_sessions['session_overall_percentile_ci_upper'].fillna(-1).tolist()
                print(f"Added overall percentile CI data for {subject_id}: {len(subject_sessions['session_overall_percentile_ci_lower'].dropna())} valid CI bounds")
            
            # PHASE 2: Add outlier detection information for visualization
            if 'is_outlier' in subject_sessions.columns:
                time_series['is_outlier'] = subject_sessions['is_outlier'].fillna(False).tolist()
                outlier_count = subject_sessions['is_outlier'].sum()
                print(f"Added outlier data for {subject_id}: {outlier_count} outlier sessions out of {len(subject_sessions)}")
            
            # Add RAW feature values for timeseries plotting (not the processed rolling averages)
            for feature in features:
                # Store raw feature values for timeseries component to apply its own rolling average
                if feature in subject_sessions.columns:
                    time_series[f"{feature}_raw"] = subject_sessions[feature].fillna(-1).tolist()
                    print(f"Added raw data for {feature}: {len(subject_sessions[feature].dropna())} valid values")
                
                # Keep percentiles for fallback compatibility
                percentile_col = f"{feature}_session_percentile"
                if percentile_col in subject_sessions.columns:
                    time_series[f"{feature}_percentiles"] = subject_sessions[percentile_col].fillna(-1).tolist()
                
                # Add confidence intervals for feature percentiles
                ci_lower_col = f"{feature}_session_percentile_ci_lower"
                ci_upper_col = f"{feature}_session_percentile_ci_upper"
                
                if ci_lower_col in subject_sessions.columns and ci_upper_col in subject_sessions.columns:
                    time_series[f"{feature}_percentile_ci_lower"] = subject_sessions[ci_lower_col].fillna(-1).tolist()
                    time_series[f"{feature}_percentile_ci_upper"] = subject_sessions[ci_upper_col].fillna(-1).tolist()
                    print(f"Added CI data for {feature}: {len(subject_sessions[ci_lower_col].dropna())} valid CI bounds")
            
            ui_structures['time_series_data'][subject_id] = time_series
        
        # 5. Table Display Cache WITH THRESHOLD ANALYSIS
        # Pre-compute table display data for fast rendering
        most_recent = session_data.sort_values('session_date').groupby('subject_id').last().reset_index()
        
        # CRITICAL FIX: Initialize threshold analyzer for UI cache creation
        print("Computing threshold alerts for UI cache...")
        
        # Define threshold configurations
        threshold_config = {
            'session': {
                'condition': 'gt',
                'value': 40  # Total sessions threshold
            },
            'water_day_total': {
                'condition': 'gt',
                'value': 3.5  # Water day total threshold (ml)
            }
        }
        
        # Stage-specific session thresholds
        stage_thresholds = {
            'STAGE_1': 5,
            'STAGE_2': 5,
            'STAGE_3': 6,
            'STAGE_4': 10,
            'STAGE_FINAL': 10,
            'GRADUATED': 20
        }
        
        # Combine general thresholds with stage-specific thresholds
        combined_config = threshold_config.copy()
        for stage, threshold in stage_thresholds.items():
            combined_config[f"stage_{stage}_sessions"] = {
                'condition': 'gt',
                'value': threshold
            }
        
        # Initialize threshold analyzer
        from app_utils.app_analysis.threshold_analyzer import ThresholdAnalyzer
        threshold_analyzer = ThresholdAnalyzer(combined_config)
        
        table_data = []
        for _, row in most_recent.iterrows():
            subject_id = row['subject_id']
            
            # Calculate threshold alerts for this subject
            total_sessions_alert = 'N'
            stage_sessions_alert = 'N'
            water_day_total_alert = 'N'
            overall_threshold_alert = 'N'
            
            # Get all sessions for this subject (needed for threshold calculations)
            subject_sessions = session_data[session_data['subject_id'] == subject_id]
            if not subject_sessions.empty:
                
                # 1. Check total sessions alert
                total_sessions_result = threshold_analyzer.check_total_sessions(subject_sessions)
                total_sessions_alert = total_sessions_result['display_format']
                if total_sessions_result['alert'] == 'T':
                    overall_threshold_alert = 'T'
                
                # 2. Check stage-specific sessions alert
                current_stage = row.get('current_stage_actual')
                if current_stage and current_stage in stage_thresholds:
                    stage_sessions_result = threshold_analyzer.check_stage_sessions(subject_sessions, current_stage)
                    stage_sessions_alert = stage_sessions_result['display_format']
                    if stage_sessions_result['alert'] == 'T':
                        overall_threshold_alert = 'T'
                
                # 3. Check water day total alert
                water_day_total = row.get('water_day_total')
                if not pd.isna(water_day_total):
                    water_alert_result = threshold_analyzer.check_water_day_total(water_day_total)
                    water_day_total_alert = water_alert_result['display_format']
                    if water_alert_result['alert'] == 'T':
                        overall_threshold_alert = 'T'
            
            display_row = {
                'subject_id': row['subject_id'],
                'session_date': row['session_date'],
                'session': row['session'],
                'strata': row['strata'],
                'strata_abbr': self._get_strata_abbreviation(row['strata']),
                'overall_percentile': row.get('session_overall_percentile'),
                'overall_category': row.get('overall_percentile_category', 'NS'),
                'percentile_category': row.get('overall_percentile_category', 'NS'),  # Alias for compatibility
                'combined_alert': row.get('overall_percentile_category', 'NS'),  # Will be updated with alerts
                'session_overall_rolling_avg': row.get('session_overall_rolling_avg'),  # For percentile plot hover
                'PI': row.get('PI', 'N/A'),
                'trainer': row.get('trainer', 'N/A'),
                'rig': row.get('rig', 'N/A'),
                'current_stage_actual': row.get('current_stage_actual', 'N/A'),
                'curriculum_name': row.get('curriculum_name', 'N/A'),
                # Add essential metadata columns for filtering
                'water_day_total': row.get('water_day_total'),
                'base_weight': row.get('base_weight'),
                'target_weight': row.get('target_weight'),
                'weight_after': row.get('weight_after'),
                'total_trials': row.get('total_trials'),
                'finished_trials': row.get('finished_trials'),
                'ignore_rate': row.get('ignore_rate'),
                'foraging_performance': row.get('foraging_performance'),
                'abs(bias_naive)': row.get('abs(bias_naive)'),
                'finished_rate': row.get('finished_rate'),
                # Add additional raw data columns requested by user
                'water_in_session_foraging': row.get('water_in_session_foraging'),
                'water_in_session_manual': row.get('water_in_session_manual'),
                'water_in_session_total': row.get('water_in_session_total'),
                'water_after_session': row.get('water_after_session'),
                'target_weight_ratio': row.get('target_weight_ratio'),
                'weight_after_ratio': row.get('weight_after_ratio'),
                'reward_volume_left_mean': row.get('reward_volume_left_mean'),
                'reward_volume_right_mean': row.get('reward_volume_right_mean'),
                'reaction_time_median': row.get('reaction_time_median'),
                'reaction_time_mean': row.get('reaction_time_mean'),
                'early_lick_rate': row.get('early_lick_rate'),
                'invalid_lick_ratio': row.get('invalid_lick_ratio'),
                'double_dipping_rate_finished_trials': row.get('double_dipping_rate_finished_trials'),
                'double_dipping_rate_finished_reward_trials': row.get('double_dipping_rate_finished_reward_trials'),
                'double_dipping_rate_finished_noreward_trials': row.get('double_dipping_rate_finished_noreward_trials'),
                'lick_consistency_mean_finished_trials': row.get('lick_consistency_mean_finished_trials'),
                'lick_consistency_mean_finished_reward_trials': row.get('lick_consistency_mean_finished_reward_trials'),
                'lick_consistency_mean_finished_noreward_trials': row.get('lick_consistency_mean_finished_noreward_trials'),
                'avg_trial_length_in_seconds': row.get('avg_trial_length_in_seconds'),
                # FIXED: Set computed threshold alert values instead of defaults
                'threshold_alert': overall_threshold_alert,
                'total_sessions_alert': total_sessions_alert,
                'stage_sessions_alert': stage_sessions_alert,
                'water_day_total_alert': water_day_total_alert,
                'ns_reason': '',
                # PHASE 2: Add outlier detection information
                'outlier_weight': row.get('outlier_weight', 1.0),  # Default to normal weight
                'is_outlier': row.get('is_outlier', False)         # Default to not outlier
            }
            
            # Add feature-specific data (both percentiles and rolling averages)
            for feature in features:
                percentile_col = f"{feature}_session_percentile"
                category_col = f"{feature}_category"
                rolling_avg_col = f"{feature}_processed_rolling_avg"
                # NEW: Add CI columns
                ci_lower_col = f"{feature}_session_percentile_ci_lower"
                ci_upper_col = f"{feature}_session_percentile_ci_upper"
                
                display_row[f"{feature}_session_percentile"] = row.get(percentile_col)
                display_row[f"{feature}_category"] = row.get(category_col, 'NS')
                
                # CRITICAL FIX: Add rolling average columns to table display cache
                display_row[f"{feature}_processed_rolling_avg"] = row.get(rolling_avg_col)
                
                # CRITICAL FIX: Add CI columns to table display cache
                display_row[f"{feature}_session_percentile_ci_lower"] = row.get(ci_lower_col)
                display_row[f"{feature}_session_percentile_ci_upper"] = row.get(ci_upper_col)
            
            # CRITICAL FIX: Add overall percentile CI columns
            overall_ci_lower_col = "session_overall_percentile_ci_lower"
            overall_ci_upper_col = "session_overall_percentile_ci_upper"
            display_row[overall_ci_lower_col] = row.get(overall_ci_lower_col)
            display_row[overall_ci_upper_col] = row.get(overall_ci_upper_col)
            
            table_data.append(display_row)
        
        ui_structures['table_display_cache'] = table_data
        
        print(f"UI structures created:")
        print(f"  - Feature rank data: {len(ui_structures['feature_rank_data'])} subjects")
        print(f"  - Subject lookups: {len(ui_structures['subject_lookup'])} subjects")
        print(f"  - Strata lookups: {len(ui_structures['strata_lookup'])} strata")
        print(f"  - Time series data: {len(ui_structures['time_series_data'])} subjects")
        print(f"  - Table display cache: {len(ui_structures['table_display_cache'])} rows")
        
        # Count threshold alerts in UI cache
        threshold_count = sum(1 for row in table_data if row['threshold_alert'] == 'T')
        print(f"  - Threshold alerts computed: {threshold_count} subjects with alerts")
        
        return ui_structures
    
    def _get_strata_abbreviation(self, strata: str) -> str:
        """Get abbreviated strata name for UI display"""
        if not strata:
            return ''
        
        # Hard coded mappings for common terms
        strata_mappings = {
            'Uncoupled Baiting': 'UB',
            'Coupled Baiting': 'CB', 
            'Uncoupled Without Baiting': 'UWB',
            'Coupled Without Baiting': 'CWB',
            'BEGINNER': 'B',
            'INTERMEDIATE': 'I',
            'ADVANCED': 'A',
            'v1': '1',
            'v2': '2',
            'v3': '3'
        }
        
        # Split the strata name
        parts = strata.split('_')
        
        # Handle different strata formats
        if len(parts) >= 3:
            # Format: curriculum_Stage_Version
            curriculum = '_'.join(parts[:-2])
            stage = parts[-2]
            version = parts[-1]
            
            # Get abbreviations
            curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
            stage_abbr = strata_mappings.get(stage, stage[0])
            version_abbr = strata_mappings.get(version, version[-1])
            
            return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
        
        return strata.replace(" ", "")

    def get_feature_rank_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized feature rank data for the feature rank plot
        
        Parameters:
            subject_id: str
                Subject ID to get feature rank data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Feature rank data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['feature_rank_data'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['feature_rank_data'].get(subject_id, {})
        
        return {}
    
    def get_subject_display_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized subject data for UI display components
        
        Parameters:
            subject_id: str
                Subject ID to get display data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Subject display data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['subject_lookup'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['subject_lookup'].get(subject_id, {})
        
        return {}
    
    def get_table_display_data(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get optimized table display data for fast rendering
        
        Parameters:
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            List[Dict[str, Any]]: Table data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['table_display_cache']
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['table_display_cache']
        
        return []
    
    def get_time_series_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized time series data for visualization components
        
        Parameters:
            subject_id: str
                Subject ID to get time series data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Time series data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['time_series_data'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['time_series_data'].get(subject_id, {})
        
        return {}

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory usage summary for optimization monitoring
        
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        try:
            import psutil
            import sys
            
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            summary = {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'cache_sizes': {},
                'optimization_status': {},
                'total_data_objects': 0
            }
            
            # Calculate cache sizes
            for cache_key, cache_data in self._cache.items():
                if cache_data is not None:
                    try:
                        cache_size = sys.getsizeof(cache_data)
                        summary['cache_sizes'][cache_key] = {
                            'size_mb': cache_size / 1024 / 1024,
                            'type': type(cache_data).__name__
                        }
                        
                        # Add detailed size for specific caches
                        if cache_key == 'session_level_data' and hasattr(cache_data, '__len__'):
                            summary['cache_sizes'][cache_key]['rows'] = len(cache_data)
                        elif cache_key == 'ui_structures' and isinstance(cache_data, dict):
                            for struct_key, struct_data in cache_data.items():
                                if isinstance(struct_data, dict):
                                    summary['cache_sizes'][f'{cache_key}.{struct_key}'] = {
                                        'size_mb': sys.getsizeof(struct_data) / 1024 / 1024,
                                        'count': len(struct_data)
                                    }
                        elif cache_key == 'optimized_storage' and isinstance(cache_data, dict):
                            metadata = cache_data.get('metadata', {})
                            summary['cache_sizes'][cache_key].update({
                                'subjects': metadata.get('total_subjects', 0),
                                'sessions': metadata.get('total_sessions', 0),
                                'strata': metadata.get('total_strata', 0)
                            })
                            
                    except Exception as e:
                        summary['cache_sizes'][cache_key] = {'error': str(e)}
            
            # Optimization status
            summary['optimization_status'] = {
                'unified_pipeline_active': self._cache.get('session_level_data') is not None,
                'optimized_storage_active': self._cache.get('optimized_storage') is not None,
                'ui_structures_active': self._cache.get('ui_structures') is not None,
                'memory_efficient_caching': True  # We always use efficient caching now
            }
            
            # Calculate total cached objects
            for cache_data in self._cache.values():
                if cache_data is not None:
                    summary['total_data_objects'] += 1
            
            return summary
            
        except Exception as e:
            return {'error': f"Memory monitoring failed: {str(e)}"}
    
    def compress_cache_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Compress cached data to reduce memory usage
        
        Parameters:
            force: bool
                Whether to force compression even if memory usage is acceptable
                
        Returns:
            Dict[str, Any]: Compression results and memory savings
        """
        import pickle
        import gzip
        import sys
        
        results = {
            'compressed_caches': [],
            'memory_saved_mb': 0,
            'compression_ratios': {}
        }
        
        # Check if compression is needed
        memory_summary = self.get_memory_usage_summary()
        process_memory = memory_summary.get('process_memory_mb', 0)
        
        # Compress if memory usage is high or force is requested
        if process_memory > 500 or force:  # 500MB threshold
            print(f"Compressing cache data (current memory: {process_memory:.1f}MB)...")
            
            # Compress large cache objects
            compressible_caches = ['session_level_data', 'optimized_storage', 'ui_structures']
            
            for cache_key in compressible_caches:
                if cache_key in self._cache and self._cache[cache_key] is not None:
                    try:
                        # Get original size
                        original_data = self._cache[cache_key]
                        original_size = sys.getsizeof(original_data)
                        
                        # Compress using pickle + gzip
                        compressed_data = gzip.compress(pickle.dumps(original_data))
                        compressed_size = sys.getsizeof(compressed_data)
                        
                        # Calculate compression ratio
                        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
                        
                        # Only compress if we get significant savings (>30% reduction)
                        if compression_ratio > 1.3:
                            # Store compressed data with metadata
                            self._cache[f'{cache_key}_compressed'] = {
                                'data': compressed_data,
                                'original_size': original_size,
                                'compressed_size': compressed_size,
                                'compression_ratio': compression_ratio
                            }
                            
                            # Remove original data to save memory
                            self._cache[cache_key] = None
                            
                            results['compressed_caches'].append(cache_key)
                            results['memory_saved_mb'] += (original_size - compressed_size) / 1024 / 1024
                            results['compression_ratios'][cache_key] = compression_ratio
                            
                            print(f"  Compressed {cache_key}: {compression_ratio:.1f}x reduction")
                        
                    except Exception as e:
                        print(f"  Failed to compress {cache_key}: {str(e)}")
        
        return results
    
    def decompress_cache_data(self, cache_key: str) -> Any:
        """
        Decompress cached data when needed
        
        Parameters:
            cache_key: str
                Cache key to decompress
                
        Returns:
            Any: Decompressed data
        """
        compressed_key = f'{cache_key}_compressed'
        
        if compressed_key in self._cache and self._cache[compressed_key] is not None:
            try:
                import pickle
                import gzip
                
                compressed_info = self._cache[compressed_key]
                compressed_data = compressed_info['data']
                
                # Decompress
                decompressed_data = pickle.loads(gzip.decompress(compressed_data))
                
                # Cache the decompressed data temporarily
                self._cache[cache_key] = decompressed_data
                
                print(f"Decompressed {cache_key} ({compressed_info['compression_ratio']:.1f}x)")
                
                return decompressed_data
                
            except Exception as e:
                print(f"Failed to decompress {cache_key}: {str(e)}")
                return None
        
        # Return regular cached data if not compressed
        return self._cache.get(cache_key)
    
    def optimize_memory_usage(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Optimize memory usage by compressing caches and cleaning up unused data
        
        Parameters:
            aggressive: bool
                Whether to use aggressive optimization (may impact performance)
                
        Returns:
            Dict[str, Any]: Optimization results
        """
        print("Optimizing memory usage...")
        
        # Get initial memory state
        initial_memory = self.get_memory_usage_summary()
        initial_memory_mb = initial_memory.get('process_memory_mb', 0)
        
        optimization_results = {
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': 0,
            'memory_saved_mb': 0,
            'optimizations_applied': []
        }
        
        # 1. Compress large cache objects
        compression_results = self.compress_cache_data(force=aggressive)
        if compression_results['compressed_caches']:
            optimization_results['optimizations_applied'].append('cache_compression')
            optimization_results['compression_details'] = compression_results
        
        # 2. Clean up redundant caches (aggressive mode)
        if aggressive:
            # Remove raw data cache if processed data exists
            if (self._cache.get('session_level_data') is not None and 
                self._cache.get('raw_data') is not None):
                self._cache['raw_data'] = None
                optimization_results['optimizations_applied'].append('raw_data_cleanup')
            
            # Remove stratified data if UI structures exist
            if (self._cache.get('ui_structures') is not None and 
                self._cache.get('stratified_data') is not None):
                self._cache['stratified_data'] = None
                optimization_results['optimizations_applied'].append('stratified_data_cleanup')
        
        # 3. Force garbage collection
        import gc
        gc.collect()
        optimization_results['optimizations_applied'].append('garbage_collection')
        
        # Get final memory state
        final_memory = self.get_memory_usage_summary()
        final_memory_mb = final_memory.get('process_memory_mb', 0)
        
        optimization_results['final_memory_mb'] = final_memory_mb
        optimization_results['memory_saved_mb'] = initial_memory_mb - final_memory_mb
        
        print(f"Memory optimization complete:")
        print(f"  Initial memory: {initial_memory_mb:.1f}MB")
        print(f"  Final memory: {final_memory_mb:.1f}MB")
        print(f"  Memory saved: {optimization_results['memory_saved_mb']:.1f}MB")
        print(f"  Optimizations: {', '.join(optimization_results['optimizations_applied'])}")
        
        return optimization_results

    def clear_ui_cache(self):
        """
        Clear UI-optimized caches to force regeneration with updated column structure
        Call this after modifying the columns in create_ui_optimized_structures
        """
        print("Clearing UI caches to force regeneration with new columns...")
        self._cache['ui_structures'] = None
        self._cache['optimized_storage'] = None
        print("✅ UI caches cleared - new columns will be included on next data access")
        
    def force_reload_with_new_columns(self):
        """
        Force complete reload of data with new column structure
        This clears all caches and reprocesses data to include new columns
        """
        print("🔄 Force reloading data with new column structure...")
        
        # Clear all caches
        self._invalidate_derived_caches()
        
        # Reload raw data
        raw_data = self.reload_data()
        print(f"Reloaded {len(raw_data)} sessions")
        
        # Reprocess pipeline with new column structure  
        session_data = self.process_data_pipeline(raw_data, use_cache=False)
        print(f"Reprocessed {len(session_data)} sessions with new columns")
        
        # Get sample of new table data to verify columns
        table_data = self.get_table_display_data(use_cache=False)
        if table_data:
            print(f"✅ Table display cache regenerated with {len(table_data[0])} columns")
        
        return session_data

    def force_regenerate_ui_cache_with_threshold_alerts(self):
        """
        Force regeneration of UI cache with threshold alerts included
        This clears the UI cache and ensures threshold alerts are computed
        """
        print("🔄 Force regenerating UI cache with threshold alerts...")
        
        # Clear UI-related caches
        self._cache['ui_structures'] = None
        self._cache['optimized_storage'] = None
        
        # Get session-level data (this should be available)
        if self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process data if needed
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Force create new UI structures with threshold analysis
        ui_structures = self.create_ui_optimized_structures(session_data)
        self._cache['ui_structures'] = ui_structures
        
        # Verify threshold alerts were computed
        table_data = ui_structures.get('table_display_cache', [])
        threshold_count = sum(1 for row in table_data if row.get('threshold_alert') == 'T')
        
        print(f"✅ UI cache regenerated with {threshold_count} threshold alerts computed")
        
        return threshold_count

    def test_phase2_outlier_detection(self, comparison_methods: List[str] = None) -> Dict[str, Any]:
        """
        Test and validate Phase 2 outlier detection improvements
        
        Parameters:
            comparison_methods: List[str], optional
                List of outlier detection methods to compare ['iqr', 'modified_zscore', 'none']
                
        Returns:
            Dict[str, Any]: Comprehensive comparison results
        """
        if comparison_methods is None:
            comparison_methods = ['iqr', 'modified_zscore', 'none']
        
        print("🧪 PHASE 2 Testing: Outlier Detection Method Comparison")
        print("=" * 60)
        
        # Get raw data for testing
        raw_data = self.get_session_data(use_cache=True)
        eligible_subjects = self.reference_processor.get_eligible_subjects(raw_data) if self.reference_processor else []
        eligible_df = raw_data[raw_data['subject_id'].isin(eligible_subjects)]
        
        if eligible_df.empty:
            return {'error': 'No eligible data for testing'}
        
        comparison_results = {}
        
        for method in comparison_methods:
            print(f"\n🔍 Testing method: {method.upper()}")
            
            # Create test configuration
            test_config = {
                'method': method,
                'factor': 1.5,
                'handling': 'weighted',
                'outlier_weight': 0.5,
                'min_data_points': 4
            }
            
            # Create temporary reference processor for testing
            features_config = {
                'finished_trials': False,
                'ignore_rate': True,
                'total_trials': False,
                'foraging_performance': False,
                'abs(bias_naive)': True
            }
            
            test_processor = ReferenceProcessor(
                features_config=features_config,
                min_sessions=1,
                min_days=1,
                outlier_config=test_config
            )
            
            # Process data with this method
            try:
                processed_df = test_processor.preprocess_data(eligible_df, remove_outliers=True)
                
                # Collect statistics
                total_sessions = len(processed_df)
                outlier_sessions = 0
                outlier_rate = 0.0
                
                if 'outlier_weight' in processed_df.columns:
                    outlier_sessions = (processed_df['outlier_weight'] < 1.0).sum()
                    outlier_rate = (outlier_sessions / total_sessions) * 100 if total_sessions > 0 else 0
                
                # Calculate feature-specific outlier rates
                feature_outlier_rates = {}
                processed_features = [col for col in processed_df.columns if col.endswith('_processed')]
                
                for feature_col in processed_features:
                    feature_name = feature_col.replace('_processed', '')
                    if feature_name in features_config:
                        # Check how many outliers this feature would detect
                        feature_values = processed_df[feature_col].dropna().values
                        if len(feature_values) > 0 and method != 'none':
                            outlier_mask, _ = test_processor._detect_outliers(feature_values)
                            feature_outlier_count = np.sum(outlier_mask)
                            feature_outlier_rate = (feature_outlier_count / len(feature_values)) * 100
                            feature_outlier_rates[feature_name] = {
                                'count': feature_outlier_count,
                                'rate': feature_outlier_rate,
                                'total_values': len(feature_values)
                            }
                
                # Store results
                comparison_results[method] = {
                    'total_sessions': total_sessions,
                    'outlier_sessions': outlier_sessions,
                    'outlier_rate': outlier_rate,
                    'feature_outlier_rates': feature_outlier_rates,
                    'data_retention': 100.0,  # We use weighting, so 100% retention
                    'method_config': test_config
                }
                
                print(f"   Sessions processed: {total_sessions}")
                print(f"   Outliers detected: {outlier_sessions} ({outlier_rate:.1f}%)")
                print(f"   Data retention: 100% (weighted approach)")
                
                # Feature-specific results
                if feature_outlier_rates:
                    print(f"   Feature-specific outlier rates:")
                    for feature, stats in feature_outlier_rates.items():
                        print(f"     {feature}: {stats['count']}/{stats['total_values']} ({stats['rate']:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Error testing {method}: {str(e)}")
                comparison_results[method] = {'error': str(e)}
        
        # Generate summary comparison
        print(f"\n📊 PHASE 2 Comparison Summary:")
        print("-" * 40)
        
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        
        if valid_results:
            print(f"{'Method':<15} {'Outlier Rate':<12} {'Sessions':<10} {'Retention'}")
            print("-" * 50)
            
            for method, results in valid_results.items():
                outlier_rate = results['outlier_rate']
                total_sessions = results['total_sessions']
                retention = results['data_retention']
                
                print(f"{method:<15} {outlier_rate:>8.1f}%     {total_sessions:>7}    {retention:>6.0f}%")
        
        # Recommendations
        print(f"\n💡 PHASE 2 Recommendations:")
        if 'iqr' in valid_results and 'none' in valid_results:
            iqr_rate = valid_results['iqr']['outlier_rate']
            print(f"   • IQR method detected {iqr_rate:.1f}% outliers vs. 0% with no detection")
            if iqr_rate >= 1.0 and iqr_rate <= 5.0:
                print(f"   • ✅ IQR rate ({iqr_rate:.1f}%) is within expected range (1-5%)")
            elif iqr_rate > 5.0:
                print(f"   • ⚠️  IQR rate ({iqr_rate:.1f}%) is higher than expected - consider adjusting factor")
            else:
                print(f"   • ⚠️  IQR rate ({iqr_rate:.1f}%) is lower than expected - data may be very clean")
        
        print(f"   • Weighted approach maintains 100% data retention")
        print(f"   • Enhanced robustness vs. previous 3-sigma method")
        
        return comparison_results