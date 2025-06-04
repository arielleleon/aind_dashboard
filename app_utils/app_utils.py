from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer, BootstrapManager
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
        # PHASE 3: Bootstrap Manager for enhanced statistical robustness
        self.bootstrap_manager = None
        
        # Simplified cache for processed data
        self._cache = {
            'raw_data': None,
            'session_level_data': None,
            'optimized_storage': None,
            'ui_structures': None,
            'unified_alerts': None,
            'last_process_time': None,
            'data_hash': None,
            # PHASE 3: Bootstrap cache for enhanced reference distributions
            'bootstrap_coverage_stats': None,
            'bootstrap_enabled_strata': None
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
        # PHASE 3: Clear bootstrap caches when data changes
        self._cache['bootstrap_coverage_stats'] = None
        self._cache['bootstrap_enabled_strata'] = None
        
        # Also clear percentile calculator cache
        if hasattr(self, 'percentile_calculator'):
            self.percentile_calculator.clear_cache()
        
        # PHASE 3: Clear bootstrap manager cache when data changes
        if hasattr(self, 'bootstrap_manager') and self.bootstrap_manager is not None:
            self.bootstrap_manager.clear_cache()
    
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
        Initialize quantile analyzer with Phase 3 bootstrap manager integration

        Parameters:
            stratified_data (Dict[str, pd.DataFrame]): Dictionary of stratified data

        Returns:
            QuantileAnalyzer: Initialized quantile analyzer with bootstrap support
        """
        self.quantile_analyzer = QuantileAnalyzer(
            stratified_data = stratified_data,
            historical_data = getattr(self, 'historical_data', None),
            # PHASE 3: Pass bootstrap manager for enhanced confidence intervals
            bootstrap_manager = self.bootstrap_manager
        )
        
        # PHASE 3: Report bootstrap integration status
        if self.bootstrap_manager is not None:
            print(" QuantileAnalyzer initialized with Phase 3 bootstrap enhancement")
        else:
            print(" QuantileAnalyzer initialized with standard confidence intervals")
        
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
            # PHASE 3: Initialize bootstrap manager before quantile analyzer if not already done
            if self.bootstrap_manager is None:
                print("Initializing Bootstrap Manager for Phase 3 enhanced statistical robustness...")
                self.initialize_bootstrap_manager()
            
            # Create reference distributions using the current approach but for session percentiles
            stratified_data = self.reference_processor.prepare_for_quantile_analysis(
                processed_df, include_history=True
            )
            # PHASE 3: Pass bootstrap manager to quantile analyzer
            self.initialize_quantile_analyzer(stratified_data)
            print(f"Initialized quantile analyzer with {len(stratified_data)} strata")
            
            # PHASE 3: Generate bootstrap distributions for enhanced confidence intervals
            # This ensures bootstrap distributions are available for the quantile analyzer
            print("Generating bootstrap distributions for enhanced statistical robustness...")
            bootstrap_result = self.generate_bootstrap_distributions(force_regenerate=False)
            
            # Report bootstrap generation results
            if bootstrap_result.get('bootstrap_enabled_count', 0) > 0:
                enabled_count = bootstrap_result['bootstrap_enabled_count']
                total_strata = bootstrap_result['total_strata']
                print(f"✅ Bootstrap enhancement enabled for {enabled_count}/{total_strata} strata")
                print(f"   - Enhancement will be applied to confidence intervals in percentile calculations")
            else:
                print("⚠️  No bootstrap distributions generated - using standard confidence intervals")
                if bootstrap_result.get('warnings'):
                    print(f"   Warnings: {len(bootstrap_result['warnings'])} issues detected")
        
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
        
        # Step 6.75: PHASE 1 OPTIMIZATION - Pre-compute bootstrap CIs during pipeline
        comprehensive_data = self.calculate_session_bootstrap_cis(comprehensive_data)
        
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
        
        # PHASE 3: Add bootstrap enhancement indicators
        if self.bootstrap_manager is not None:
            print("Adding bootstrap enhancement indicators to session metadata...")
            
            # Process each session to determine which percentiles used bootstrap
            for idx, row in result_df.iterrows():
                strata = row.get('strata', '')
                
                # Check bootstrap availability for each feature
                for feature in feature_list:
                    percentile_col = f"{feature}_session_percentile"
                    ci_lower_col = f"{feature}_session_percentile_ci_lower"
                    ci_upper_col = f"{feature}_session_percentile_ci_upper"
                    bootstrap_indicator_col = f"{feature}_bootstrap_enhanced"
                    
                    # Check if this feature has bootstrap enhancement for this strata
                    if (percentile_col in result_df.columns and 
                        ci_lower_col in result_df.columns and 
                        ci_upper_col in result_df.columns):
                        
                        # Check if bootstrap is available and CI values are not NaN
                        bootstrap_available = self.bootstrap_manager.is_bootstrap_available(strata, feature)
                        has_valid_ci = (not pd.isna(row[ci_lower_col]) and not pd.isna(row[ci_upper_col]))
                        
                        # Bootstrap enhanced if bootstrap is available AND we have valid CIs
                        # (CIs could still be Wilson score if bootstrap failed)
                        bootstrap_enhanced = bootstrap_available and has_valid_ci
                        
                        result_df.loc[idx, bootstrap_indicator_col] = bootstrap_enhanced
                    else:
                        result_df.loc[idx, bootstrap_indicator_col] = False

                # Check bootstrap enhancement for overall percentile
                overall_ci_lower_col = "session_overall_percentile_ci_lower"
                overall_ci_upper_col = "session_overall_percentile_ci_upper"
                overall_bootstrap_indicator_col = "session_overall_bootstrap_enhanced"
                
                if (overall_ci_lower_col in result_df.columns and 
                    overall_ci_upper_col in result_df.columns):
                    
                    # Overall percentile is bootstrap enhanced if ANY feature percentile is bootstrap enhanced
                    feature_bootstrap_indicators = [f"{feature}_bootstrap_enhanced" for feature in feature_list]
                    any_feature_bootstrap = any(
                        result_df.loc[idx, col] for col in feature_bootstrap_indicators 
                        if col in result_df.columns
                    )
                    
                    has_overall_ci = (not pd.isna(row[overall_ci_lower_col]) and not pd.isna(row[overall_ci_upper_col]))
                    
                    result_df.loc[idx, overall_bootstrap_indicator_col] = any_feature_bootstrap and has_overall_ci
                else:
                    result_df.loc[idx, overall_bootstrap_indicator_col] = False
            
            # Count how many sessions have bootstrap enhancement
            bootstrap_enhanced_sessions = 0
            for feature in feature_list:
                bootstrap_col = f"{feature}_bootstrap_enhanced"
                if bootstrap_col in result_df.columns:
                    enhanced_count = result_df[bootstrap_col].sum()
                    if enhanced_count > 0:
                        print(f"  {feature}: {enhanced_count} sessions with bootstrap-enhanced CIs")
                        bootstrap_enhanced_sessions = max(bootstrap_enhanced_sessions, enhanced_count)
            
            overall_bootstrap_col = "session_overall_bootstrap_enhanced"
            if overall_bootstrap_col in result_df.columns:
                overall_enhanced_count = result_df[overall_bootstrap_col].sum()
                print(f"  Overall percentile: {overall_enhanced_count} sessions with bootstrap-enhanced CIs")
            
            print(f"Total sessions with any bootstrap enhancement: {bootstrap_enhanced_sessions}")
        else:
            # If no bootstrap manager, set all bootstrap indicators to False
            for feature in feature_list:
                result_df[f"{feature}_bootstrap_enhanced"] = False
            result_df["session_overall_bootstrap_enhanced"] = False
        
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
        4. PHASE 3: Bootstrap indicators and coverage statistics
        
        Parameters:
            session_data: pd.DataFrame
                Complete session-level data from unified pipeline
                
        Returns:
            Dict[str, Any]: Optimized storage structure with bootstrap support
        """
        print("Optimizing session data storage...")
        
        # Create subject-indexed storage for fast subject lookups
        subject_data = {}
        strata_reference = {}
        
        # PHASE 3: Initialize bootstrap coverage tracking
        bootstrap_coverage = {}
        bootstrap_enabled_strata_set = set()
        
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
                'abs(bias_naive)', 'finished_rate',
                # AUTOWATER COLUMNS: Add all autowater metrics to table display cache
                'total_trials_with_autowater', 'finished_trials_with_autowater', 'finished_rate_with_autowater', 'ignore_rate_with_autowater', 'autowater_collected', 'autowater_ignored', 'water_day_total_last_session', 'water_after_session_last_session',
                # PHASE 3: Add bootstrap enhancement indicators
                'session_overall_bootstrap_enhanced'
            ]
            
            # Add feature-specific columns
            feature_columns = [col for col in subject_sessions.columns 
                             if col.endswith(('_session_percentile', '_category', '_processed_rolling_avg'))]
            essential_columns.extend(feature_columns)
            
            # PHASE 3: Add confidence interval columns for bootstrap support
            ci_columns = [col for col in subject_sessions.columns 
                         if col.endswith(('_ci_lower', '_ci_upper'))]
            essential_columns.extend(ci_columns)
            
            # PHASE 3: Add bootstrap indicator columns
            bootstrap_indicator_columns = [col for col in subject_sessions.columns 
                                         if col.endswith('_bootstrap_enhanced')]
            essential_columns.extend(bootstrap_indicator_columns)
            
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
                
                # PHASE 3: Check for bootstrap availability and calculate coverage statistics
                bootstrap_enabled = False
                feature_bootstrap_coverage = {}
                
                if self.bootstrap_manager is not None:
                    # Check if bootstrap is available for this strata and any features
                    for feature_col in processed_features:
                        feature_name = feature_col.replace('_processed_rolling_avg', '')
                        if self.bootstrap_manager.is_bootstrap_available(strata, feature_name):
                            bootstrap_enabled = True
                            bootstrap_enabled_strata_set.add(strata)
                            
                            # Calculate coverage statistics for this feature
                            # Look for CI columns in the session data
                            ci_lower_col = f"{feature_name}_session_percentile_ci_lower"
                            ci_upper_col = f"{feature_name}_session_percentile_ci_upper"
                            
                            if ci_lower_col in strata_sessions.columns and ci_upper_col in strata_sessions.columns:
                                valid_ci_count = strata_sessions[[ci_lower_col, ci_upper_col]].dropna().shape[0]
                                total_sessions = len(strata_sessions)
                                coverage_rate = valid_ci_count / total_sessions if total_sessions > 0 else 0
                                
                                feature_bootstrap_coverage[feature_name] = {
                                    'bootstrap_available': True,
                                    'ci_coverage_rate': coverage_rate,
                                    'valid_ci_sessions': valid_ci_count,
                                    'total_sessions': total_sessions
                                }
                            else:
                                feature_bootstrap_coverage[feature_name] = {
                                    'bootstrap_available': True,
                                    'ci_coverage_rate': 0.0,
                                    'valid_ci_sessions': 0,
                                    'total_sessions': len(strata_sessions),
                                    'warning': 'Bootstrap available but CI columns missing'
                                }
                        else:
                            feature_bootstrap_coverage[feature_name] = {
                                'bootstrap_available': False,
                                'ci_coverage_rate': 0.0,
                                'reason': 'Bootstrap not available for this strata/feature combination'
                            }
                
                # Store bootstrap coverage statistics for this strata
                bootstrap_coverage[strata] = {
                    'bootstrap_enabled': bootstrap_enabled,
                    'feature_coverage': feature_bootstrap_coverage,
                    'subject_count': len(reference_data),
                    'session_count': len(strata_sessions)
                }
                
                strata_reference[strata] = {
                    'subject_count': len(reference_data),
                    'session_count': len(strata_sessions),
                    'reference_distributions': {
                        feature: reference_data[feature].values.tolist() 
                        for feature in processed_features
                        if not reference_data[feature].isna().all()
                    },
                    # PHASE 3: Add bootstrap indicator to strata reference
                    'bootstrap_enabled': bootstrap_enabled
                }
        
        # Create optimized storage structure with Phase 3 enhancements
        optimized_storage = {
            'subjects': subject_data,
            'strata_reference': strata_reference,
            'metadata': {
                'total_subjects': len(subject_data),
                'total_sessions': len(session_data),
                'total_strata': len(strata_reference),
                'storage_timestamp': pd.Timestamp.now(),
                'data_hash': self._calculate_data_hash(session_data),
                # PHASE 3: Bootstrap metadata
                'bootstrap_enabled_strata_count': len(bootstrap_enabled_strata_set),
                'bootstrap_enabled_strata_list': list(bootstrap_enabled_strata_set),
                'phase3_enhanced': True
            },
            # PHASE 3: Bootstrap coverage statistics as separate cache structure
            'bootstrap_coverage': bootstrap_coverage
        }
        
        # PHASE 3: Cache bootstrap coverage statistics separately for fast access
        self._cache['bootstrap_coverage_stats'] = bootstrap_coverage
        self._cache['bootstrap_enabled_strata'] = bootstrap_enabled_strata_set
        
        print(f"Optimized storage created:")
        print(f"  - {len(subject_data)} subjects")
        print(f"  - {len(strata_reference)} strata references")
        print(f"  - {len(session_data)} total sessions")
        # PHASE 3: Report bootstrap enhancement status
        print(f"  - PHASE 3: {len(bootstrap_enabled_strata_set)} strata with bootstrap enhancement")
        if bootstrap_coverage:
            bootstrap_features = sum(len(coverage['feature_coverage']) for coverage in bootstrap_coverage.values())
            print(f"  - PHASE 3: {bootstrap_features} feature-strata combinations analyzed for bootstrap coverage")
        
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
            Dict[str, Any]: Storage summary statistics with Phase 3 bootstrap information
        """
        summary = {
            'raw_data_cached': self._cache['raw_data'] is not None,
            'session_level_cached': self._cache['session_level_data'] is not None,
            'optimized_storage_cached': 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None,
            'formatted_data_cached': self._cache.get('formatted_data') is not None,
            # PHASE 3: Bootstrap cache status
            'bootstrap_coverage_cached': self._cache['bootstrap_coverage_stats'] is not None,
            'bootstrap_enabled_strata_cached': self._cache['bootstrap_enabled_strata'] is not None
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
            
            # PHASE 3: Add bootstrap storage information
            metadata = storage.get('metadata', {})
            if metadata.get('phase3_enhanced', False):
                summary.update({
                    'bootstrap_enabled_strata_count': metadata.get('bootstrap_enabled_strata_count', 0),
                    'bootstrap_enabled_strata_list': metadata.get('bootstrap_enabled_strata_list', []),
                    'phase3_enhanced': True
                })
        
        if summary['session_level_cached']:
            summary['session_level_rows'] = len(self._cache['session_level_data'])
        
        if summary['formatted_data_cached']:
            summary['formatted_data_rows'] = len(self._cache.get('formatted_data', []))
        
        return summary
    
    def get_bootstrap_coverage_stats(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get bootstrap coverage statistics for all strata
        
        Parameters:
            use_cache: bool
                Whether to use cached coverage statistics
                
        Returns:
            Dict[str, Any]: Bootstrap coverage statistics by strata
        """
        # Check cache first
        if use_cache and self._cache['bootstrap_coverage_stats'] is not None:
            return self._cache['bootstrap_coverage_stats']
        
        # Check optimized storage
        if 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            bootstrap_coverage = optimized_storage.get('bootstrap_coverage', {})
            if bootstrap_coverage:
                self._cache['bootstrap_coverage_stats'] = bootstrap_coverage
                return bootstrap_coverage
        
        # No coverage statistics available
        return {}
    
    def get_bootstrap_enabled_strata(self, use_cache: bool = True) -> set:
        """
        Get set of strata names that have bootstrap enhancement enabled
        
        Parameters:
            use_cache: bool
                Whether to use cached strata set
                
        Returns:
            set: Set of strata names with bootstrap enhancement
        """
        # Check cache first  
        if use_cache and self._cache['bootstrap_enabled_strata'] is not None:
            return self._cache['bootstrap_enabled_strata']
        
        # Check optimized storage metadata
        if 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            metadata = optimized_storage.get('metadata', {})
            strata_list = metadata.get('bootstrap_enabled_strata_list', [])
            if strata_list:
                strata_set = set(strata_list)
                self._cache['bootstrap_enabled_strata'] = strata_set
                return strata_set
        
        # No enabled strata found
        return set()
    
    def is_bootstrap_enabled_for_strata(self, strata: str, use_cache: bool = True) -> bool:
        """
        Check if a specific strata has bootstrap enhancement enabled
        
        Parameters:
            strata: str
                Strata name to check
            use_cache: bool
                Whether to use cached data
                
        Returns:
            bool: True if bootstrap is enabled for this strata
        """
        enabled_strata = self.get_bootstrap_enabled_strata(use_cache)
        return strata in enabled_strata
    
    def get_strata_bootstrap_coverage(self, strata: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get detailed bootstrap coverage information for a specific strata
        
        Parameters:
            strata: str
                Strata name to get coverage for
            use_cache: bool
                Whether to use cached data
                
        Returns:
            Dict[str, Any]: Coverage statistics for the strata
        """
        coverage_stats = self.get_bootstrap_coverage_stats(use_cache)
        return coverage_stats.get(strata, {})
    
    def generate_bootstrap_distributions(self, 
                                       force_regenerate: bool = False,
                                       strata_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate bootstrap distributions for eligible strata using the current data
        
        Parameters:
            force_regenerate: bool
                Force regeneration of all bootstrap distributions
            strata_filter: Optional[List[str]]
                List of specific strata to process (None = all strata)
                
        Returns:
            Dict[str, Any]: Bootstrap generation results
        """
        if self.bootstrap_manager is None:
            # Initialize with default configuration
            self.initialize_bootstrap_manager()
        
        if self.reference_processor is None:
            print("Warning: Reference processor not initialized. Cannot generate bootstrap distributions.")
            return {'error': 'Reference processor not available'}
        
        # Get current session data for session date checking
        session_data = self._cache.get('session_level_data')
        session_dates = None
        if session_data is not None:
            session_dates = session_data['session_date']
        
        # CRITICAL FIX: Remove recursive call to process_data_pipeline
        # Get stratified data directly from quantile analyzer instead of triggering recursive processing
        if hasattr(self, 'quantile_analyzer') and self.quantile_analyzer is not None:
            strata_data = getattr(self.quantile_analyzer, 'stratified_data', None)
            if strata_data is None:
                print("Warning: No stratified data found in quantile analyzer - bootstrap generation skipped")
                return {'error': 'No stratified data available for bootstrap generation', 'bootstrap_enabled_count': 0}
        else:
            print("Warning: Quantile analyzer not available - bootstrap generation skipped")
            return {'error': 'Quantile analyzer not available', 'bootstrap_enabled_count': 0}
        
        # Filter strata if requested
        if strata_filter is not None:
            strata_data = {k: v for k, v in strata_data.items() if k in strata_filter}
        
        # Generate bootstrap distributions
        print(f"Generating bootstrap distributions for {len(strata_data)} strata...")
        result = self.bootstrap_manager.generate_bootstrap_for_all_strata(
            strata_data=strata_data,
            session_dates=session_dates,
            force_regenerate=force_regenerate
        )
        
        # Update optimized cache with new bootstrap information
        if result.get('bootstrap_enabled_count', 0) > 0:
            print("Updating optimized cache with new bootstrap information...")
            if self._cache['session_level_data'] is not None:
                # Regenerate optimized storage to include bootstrap coverage
                optimized_storage = self.optimize_session_data_storage(self._cache['session_level_data'])
                self._cache['optimized_storage'] = optimized_storage
        
        return result

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
                
                # Add confidence intervals for feature percentiles (Wilson CIs)
                ci_lower_col = f"{feature}_session_percentile_ci_lower"
                ci_upper_col = f"{feature}_session_percentile_ci_upper"
                
                if ci_lower_col in subject_sessions.columns and ci_upper_col in subject_sessions.columns:
                    time_series[f"{feature}_percentile_ci_lower"] = subject_sessions[ci_lower_col].fillna(-1).tolist()
                    time_series[f"{feature}_percentile_ci_upper"] = subject_sessions[ci_upper_col].fillna(-1).tolist()
                    print(f"Added CI data for {feature}: {len(subject_sessions[ci_lower_col].dropna())} valid CI bounds")
                
                # PHASE 3: Add bootstrap indicators for feature percentiles
                bootstrap_indicator_col = f"{feature}_bootstrap_enhanced"
                if bootstrap_indicator_col in subject_sessions.columns:
                    time_series[f"{feature}_bootstrap_enhanced"] = subject_sessions[bootstrap_indicator_col].fillna(False).tolist()
                    print(f"Added bootstrap indicators for {feature}: {subject_sessions[bootstrap_indicator_col].sum()} bootstrap-enhanced sessions")
                
                # NEW: Add bootstrap CIs for raw rolling averages (separate from percentile CIs)
                # Use pre-computed bootstrap CIs from session data instead of calculating in real-time
                if self.bootstrap_manager is not None:
                    # Get pre-computed bootstrap CI columns from session data
                    ci_lower_col = f"{feature}_bootstrap_ci_lower"
                    ci_upper_col = f"{feature}_bootstrap_ci_upper"
                    
                    if ci_lower_col in subject_sessions.columns and ci_upper_col in subject_sessions.columns:
                        # Use pre-computed values (replace NaN with -1 for UI compatibility)
                        bootstrap_ci_lower_values = subject_sessions[ci_lower_col].fillna(-1).tolist()
                        bootstrap_ci_upper_values = subject_sessions[ci_upper_col].fillna(-1).tolist()
                        
                        # Count valid CIs for reporting
                        valid_bootstrap_cis = sum(1 for lower, upper in zip(bootstrap_ci_lower_values, bootstrap_ci_upper_values) 
                                                if lower != -1 and upper != -1)
                        print(f"Using pre-computed bootstrap CIs for {feature}: {valid_bootstrap_cis} valid CIs")
                    else:
                        # Fallback: create empty arrays if pre-computed CIs not available
                        bootstrap_ci_lower_values = [-1] * len(subject_sessions)
                        bootstrap_ci_upper_values = [-1] * len(subject_sessions)
                        print(f"No pre-computed bootstrap CIs for {feature} - using empty arrays")
                    
                    # Add bootstrap CI arrays to time series
                    time_series[f"{feature}_bootstrap_ci_lower"] = bootstrap_ci_lower_values
                    time_series[f"{feature}_bootstrap_ci_upper"] = bootstrap_ci_upper_values
                    
                    # Calculate CI width for time series
                    bootstrap_ci_width_values = []
                    for lower, upper in zip(bootstrap_ci_lower_values, bootstrap_ci_upper_values):
                        if lower != -1 and upper != -1:
                            bootstrap_ci_width_values.append(upper - lower)
                        else:
                            bootstrap_ci_width_values.append(-1)
                    time_series[f"{feature}_bootstrap_ci_width"] = bootstrap_ci_width_values
                
            # PHASE 3: Add overall percentile bootstrap indicator
            overall_bootstrap_col = "session_overall_bootstrap_enhanced"
            if overall_bootstrap_col in subject_sessions.columns:
                time_series["overall_bootstrap_enhanced"] = subject_sessions[overall_bootstrap_col].fillna(False).tolist()
                overall_bootstrap_count = subject_sessions[overall_bootstrap_col].sum()
                print(f"Added overall bootstrap indicators for {subject_id}: {overall_bootstrap_count} bootstrap-enhanced sessions")
            
            # NEW: Add bootstrap CIs for overall rolling average
            if self.bootstrap_manager is not None and 'session_overall_rolling_avg' in subject_sessions.columns:
                overall_bootstrap_ci_lower_values = []
                overall_bootstrap_ci_upper_values = []
                
                for idx, row in subject_sessions.iterrows():
                    overall_rolling_avg = row.get('session_overall_rolling_avg')
                    
                    if pd.isna(overall_rolling_avg):
                        overall_bootstrap_ci_lower_values.append(-1)
                        overall_bootstrap_ci_upper_values.append(-1)
                        continue
                    
                    strata = row['strata']
                    
                    # For overall bootstrap CI, use combined reference data from all features
                    if hasattr(self, 'quantile_analyzer') and self.quantile_analyzer is not None:
                        strata_data = self.quantile_analyzer.percentile_data.get(strata)
                        if strata_data is not None:
                            # FIXED: Collect all processed feature columns (not rolling_avg columns)
                            all_processed_values = []
                            for feature in features:
                                reference_col = f"{feature}_processed"
                                if reference_col in strata_data.columns:
                                    feature_values = strata_data[reference_col].dropna().values
                                    all_processed_values.extend(feature_values)
                            
                            if len(all_processed_values) >= 10:  # Need sufficient data
                                # Use bootstrap CI for the overall rolling average
                                ci_lower, ci_upper = self.bootstrap_manager.statistical_utils.calculate_bootstrap_raw_value_ci(
                                    reference_data=np.array(all_processed_values),
                                    target_value=overall_rolling_avg,
                                    confidence_level=0.95,
                                    n_bootstrap=500,
                                    random_state=42
                                )
                                
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    overall_bootstrap_ci_lower_values.append(ci_lower)
                                    overall_bootstrap_ci_upper_values.append(ci_upper)
                                else:
                                    overall_bootstrap_ci_lower_values.append(-1)
                                    overall_bootstrap_ci_upper_values.append(-1)
                            else:
                                overall_bootstrap_ci_lower_values.append(-1)
                                overall_bootstrap_ci_upper_values.append(-1)
                        else:
                            overall_bootstrap_ci_lower_values.append(-1)
                            overall_bootstrap_ci_upper_values.append(-1)
                    else:
                        overall_bootstrap_ci_lower_values.append(-1)
                        overall_bootstrap_ci_upper_values.append(-1)
                
                time_series["overall_bootstrap_ci_lower"] = overall_bootstrap_ci_lower_values
                time_series["overall_bootstrap_ci_upper"] = overall_bootstrap_ci_upper_values
                
                # Calculate overall CI width for time series
                overall_bootstrap_ci_width_values = []
                for lower, upper in zip(overall_bootstrap_ci_lower_values, overall_bootstrap_ci_upper_values):
                    if lower != -1 and upper != -1:
                        overall_bootstrap_ci_width_values.append(upper - lower)
                    else:
                        overall_bootstrap_ci_width_values.append(-1)
                time_series["overall_bootstrap_ci_width"] = overall_bootstrap_ci_width_values
                
                valid_overall_bootstrap_cis = sum(1 for lower, upper in zip(overall_bootstrap_ci_lower_values, overall_bootstrap_ci_upper_values) 
                                                if lower != -1 and upper != -1)
                print(f"Added overall bootstrap CIs: {valid_overall_bootstrap_cis} valid CIs")
            
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
                # AUTOWATER COLUMNS: Add all autowater metrics to table display cache
                'total_trials_with_autowater': row.get('total_trials_with_autowater'),
                'finished_trials_with_autowater': row.get('finished_trials_with_autowater'),
                'finished_rate_with_autowater': row.get('finished_rate_with_autowater'),
                'ignore_rate_with_autowater': row.get('ignore_rate_with_autowater'),
                'autowater_collected': row.get('autowater_collected'),
                'autowater_ignored': row.get('autowater_ignored'),
                'water_day_total_last_session': row.get('water_day_total_last_session'),
                'water_after_session_last_session': row.get('water_after_session_last_session'),
                # FIXED: Set computed threshold alert values instead of defaults
                'threshold_alert': overall_threshold_alert,
                'total_sessions_alert': total_sessions_alert,
                'stage_sessions_alert': stage_sessions_alert,
                'water_day_total_alert': water_day_total_alert,
                'ns_reason': '',
                # PHASE 2: Add outlier detection information
                'outlier_weight': row.get('outlier_weight', 1.0),  # Default to normal weight
                'is_outlier': row.get('is_outlier', False),         # Default to not outlier
                # PHASE 3: Add bootstrap enhancement indicators
                'session_overall_bootstrap_enhanced': row.get('session_overall_bootstrap_enhanced', False)
            }
            
            # Add feature-specific data (both percentiles and rolling averages)
            for feature in features:
                percentile_col = f"{feature}_session_percentile"
                category_col = f"{feature}_category"
                rolling_avg_col = f"{feature}_processed_rolling_avg"
                # Wilson CI columns for percentiles
                ci_lower_col = f"{feature}_session_percentile_ci_lower"
                ci_upper_col = f"{feature}_session_percentile_ci_upper"
                # PHASE 3: Add bootstrap indicator columns
                bootstrap_indicator_col = f"{feature}_bootstrap_enhanced"
                
                display_row[f"{feature}_session_percentile"] = row.get(percentile_col)
                display_row[f"{feature}_category"] = row.get(category_col, 'NS')
                
                # CRITICAL FIX: Add rolling average columns to table display cache
                display_row[f"{feature}_processed_rolling_avg"] = row.get(rolling_avg_col)
                
                # Wilson CI columns (for percentile CIs)
                display_row[f"{feature}_session_percentile_ci_lower"] = row.get(ci_lower_col)
                display_row[f"{feature}_session_percentile_ci_upper"] = row.get(ci_upper_col)
                
                # PHASE 3: Add bootstrap indicator columns to table display cache
                display_row[f"{feature}_bootstrap_enhanced"] = row.get(bootstrap_indicator_col, False)
                
                # NEW: Add bootstrap CIs for raw rolling averages
                if self.bootstrap_manager is not None:
                    strata = row['strata']
                    rolling_avg_value = row.get(rolling_avg_col)
                    
                    if pd.isna(rolling_avg_value):
                        display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                        display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                        display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                    else:
                        # Use the new bootstrap raw value CI method for table display
                        if hasattr(self, 'quantile_analyzer') and self.quantile_analyzer is not None:
                            # Get reference data from quantile analyzer
                            strata_data = self.quantile_analyzer.percentile_data.get(strata)
                            
                            # FIXED: Use the correct column name that exists in reference data
                            reference_col = f"{feature}_processed"
                            
                            if strata_data is not None and reference_col in strata_data.columns:
                                reference_values = strata_data[reference_col].dropna().values
                                
                                if len(reference_values) >= 5:
                                    # Use the new bootstrap raw value CI method
                                    ci_lower, ci_upper = self.bootstrap_manager.statistical_utils.calculate_bootstrap_raw_value_ci(
                                        reference_data=reference_values,
                                        target_value=rolling_avg_value,
                                        confidence_level=0.95,
                                        n_bootstrap=500,
                                        random_state=42
                                    )
                                    
                                    if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                        display_row[f"{feature}_bootstrap_ci_lower"] = ci_lower
                                        display_row[f"{feature}_bootstrap_ci_upper"] = ci_upper
                                        # Calculate CI width
                                        display_row[f"{feature}_bootstrap_ci_width"] = ci_upper - ci_lower
                                    else:
                                        display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                                        display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                                        display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                                else:
                                    display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                                    display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                                    display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                            else:
                                display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                                display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                                display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                        else:
                            display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                            display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                            display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                else:
                    display_row[f"{feature}_bootstrap_ci_lower"] = np.nan
                    display_row[f"{feature}_bootstrap_ci_upper"] = np.nan
                    display_row[f"{feature}_bootstrap_ci_width"] = np.nan
            
            # CRITICAL FIX: Add overall percentile CI columns (Wilson CIs)
            overall_ci_lower_col = "session_overall_percentile_ci_lower"
            overall_ci_upper_col = "session_overall_percentile_ci_upper"
            display_row[overall_ci_lower_col] = row.get(overall_ci_lower_col)
            display_row[overall_ci_upper_col] = row.get(overall_ci_upper_col)
            
            # NEW: Add overall bootstrap CIs for raw rolling averages
            if self.bootstrap_manager is not None:
                overall_rolling_avg = row.get('session_overall_rolling_avg')
                
                if pd.isna(overall_rolling_avg):
                    display_row["session_overall_bootstrap_ci_lower"] = np.nan
                    display_row["session_overall_bootstrap_ci_upper"] = np.nan
                    display_row["session_overall_bootstrap_ci_width"] = np.nan
                else:
                    # For overall CI, use the combined reference data from all features
                    strata = row['strata']
                    
                    if hasattr(self, 'quantile_analyzer') and self.quantile_analyzer is not None:
                        strata_data = self.quantile_analyzer.percentile_data.get(strata)
                        if strata_data is not None:
                            # FIXED: Collect all processed feature columns for overall calculation
                            all_processed_values = []
                            for feature in features:
                                reference_col = f"{feature}_processed"
                                if reference_col in strata_data.columns:
                                    feature_values = strata_data[reference_col].dropna().values
                                    all_processed_values.extend(feature_values)
                            
                            if len(all_processed_values) >= 10:  # Need more data for overall calculation
                                # Use bootstrap CI for the overall rolling average
                                ci_lower, ci_upper = self.bootstrap_manager.statistical_utils.calculate_bootstrap_raw_value_ci(
                                    reference_data=np.array(all_processed_values),
                                    target_value=overall_rolling_avg,
                                    confidence_level=0.95,
                                    n_bootstrap=500,
                                    random_state=42
                                )
                                
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    display_row["session_overall_bootstrap_ci_lower"] = ci_lower
                                    display_row["session_overall_bootstrap_ci_upper"] = ci_upper
                                    # Calculate CI width
                                    display_row["session_overall_bootstrap_ci_width"] = ci_upper - ci_lower
                                else:
                                    display_row["session_overall_bootstrap_ci_lower"] = np.nan
                                    display_row["session_overall_bootstrap_ci_upper"] = np.nan
                                    display_row["session_overall_bootstrap_ci_width"] = np.nan
                            else:
                                display_row["session_overall_bootstrap_ci_lower"] = np.nan
                                display_row["session_overall_bootstrap_ci_upper"] = np.nan
                                display_row["session_overall_bootstrap_ci_width"] = np.nan
                        else:
                            display_row["session_overall_bootstrap_ci_lower"] = np.nan
                            display_row["session_overall_bootstrap_ci_upper"] = np.nan
                            display_row["session_overall_bootstrap_ci_width"] = np.nan
                    else:
                        display_row["session_overall_bootstrap_ci_lower"] = np.nan
                        display_row["session_overall_bootstrap_ci_upper"] = np.nan
                        display_row["session_overall_bootstrap_ci_width"] = np.nan
            else:
                display_row["session_overall_bootstrap_ci_lower"] = np.nan
                display_row["session_overall_bootstrap_ci_upper"] = np.nan
                display_row["session_overall_bootstrap_ci_width"] = np.nan
            
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
        
        # NEW: Report bootstrap CI coverage
        bootstrap_ci_counts = {}
        bootstrap_ci_width_stats = {}
        bootstrap_ci_width_thresholds = {}
        for feature in features:
            bootstrap_ci_count = sum(1 for row in table_data 
                                   if not pd.isna(row.get(f"{feature}_bootstrap_ci_lower")) and 
                                      not pd.isna(row.get(f"{feature}_bootstrap_ci_upper")))
            bootstrap_ci_counts[feature] = bootstrap_ci_count
            
            # Calculate CI width statistics and thresholds
            ci_widths = [row.get(f"{feature}_bootstrap_ci_width") for row in table_data 
                        if not pd.isna(row.get(f"{feature}_bootstrap_ci_width"))]
            if ci_widths and len(ci_widths) >= 4:  # Need at least 4 values for meaningful percentiles
                bootstrap_ci_width_stats[feature] = {
                    'mean_width': np.mean(ci_widths),
                    'median_width': np.median(ci_widths),
                    'min_width': np.min(ci_widths),
                    'max_width': np.max(ci_widths)
                }
                # Calculate percentile-based thresholds: bottom 25% = certain, top 25% = uncertain
                p25 = np.percentile(ci_widths, 25)
                p75 = np.percentile(ci_widths, 75)
                bootstrap_ci_width_thresholds[feature] = {'certain_threshold': p25, 'uncertain_threshold': p75}
        
        overall_bootstrap_ci_count = sum(1 for row in table_data 
                                       if not pd.isna(row.get("session_overall_bootstrap_ci_lower")) and 
                                          not pd.isna(row.get("session_overall_bootstrap_ci_upper")))
        bootstrap_ci_counts['overall'] = overall_bootstrap_ci_count
        
        # Calculate overall CI width statistics and thresholds
        overall_ci_widths = [row.get("session_overall_bootstrap_ci_width") for row in table_data 
                           if not pd.isna(row.get("session_overall_bootstrap_ci_width"))]
        if overall_ci_widths and len(overall_ci_widths) >= 4:
            bootstrap_ci_width_stats['overall'] = {
                'mean_width': np.mean(overall_ci_widths),
                'median_width': np.median(overall_ci_widths),
                'min_width': np.min(overall_ci_widths),
                'max_width': np.max(overall_ci_widths)
            }
            p25 = np.percentile(overall_ci_widths, 25)
            p75 = np.percentile(overall_ci_widths, 75)
            bootstrap_ci_width_thresholds['overall'] = {'certain_threshold': p25, 'uncertain_threshold': p75}
        
        # Apply certainty categories to table data
        for row in table_data:
            # Feature-specific certainty categories
            for feature in features:
                ci_width = row.get(f"{feature}_bootstrap_ci_width")
                if pd.isna(ci_width) or feature not in bootstrap_ci_width_thresholds:
                    certainty = 'intermediate'  # Default for missing data
                else:
                    thresholds = bootstrap_ci_width_thresholds[feature]
                    if ci_width <= thresholds['certain_threshold']:
                        certainty = 'certain'
                    elif ci_width >= thresholds['uncertain_threshold']:
                        certainty = 'uncertain'
                    else:
                        certainty = 'intermediate'
                row[f"{feature}_bootstrap_ci_certainty"] = certainty
            
            # Overall certainty category
            overall_ci_width = row.get("session_overall_bootstrap_ci_width")
            if pd.isna(overall_ci_width) or 'overall' not in bootstrap_ci_width_thresholds:
                overall_certainty = 'intermediate'
            else:
                thresholds = bootstrap_ci_width_thresholds['overall']
                if overall_ci_width <= thresholds['certain_threshold']:
                    overall_certainty = 'certain'
                elif overall_ci_width >= thresholds['uncertain_threshold']:
                    overall_certainty = 'uncertain'
                else:
                    overall_certainty = 'intermediate'
            row["session_overall_bootstrap_ci_certainty"] = overall_certainty
        
        print(f"  - Bootstrap CIs computed:")
        for feature, count in bootstrap_ci_counts.items():
            width_info = ""
            if feature in bootstrap_ci_width_stats:
                stats = bootstrap_ci_width_stats[feature]
                width_info = f" (avg width: {stats['mean_width']:.3f})"
            print(f"    {feature}: {count} subjects with bootstrap CIs{width_info}")
        
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
        print(" UI caches cleared - new columns will be included on next data access")
        
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
        
        print(f" UI cache regenerated with {threshold_count} threshold alerts computed")
        
        return threshold_count

    def initialize_bootstrap_manager(self, bootstrap_config: Optional[Dict[str, Any]] = None) -> BootstrapManager:
        """
        Initialize Bootstrap Manager for Phase 3 enhanced statistical robustness
        
        Parameters:
            bootstrap_config: Optional[Dict[str, Any]]
                Bootstrap configuration (uses defaults if None)
                
        Returns:
            BootstrapManager: Initialized bootstrap manager
        """
        self.bootstrap_manager = BootstrapManager(bootstrap_config)
        
        print("Bootstrap Manager initialized for Phase 3 enhanced statistical robustness")
        print("  - Tiered strategy: Large/Medium/Small strata processing")  
        print("  - 30-day update detection with session date triggers")
        print("  - Quality validation with graceful degradation")
        
        return self.bootstrap_manager

    def get_bootstrap_enhancement_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of bootstrap enhancement coverage
        
        Parameters:
            use_cache: bool
                Whether to use cached data if available
                
        Returns:
            Dict[str, Any]
                Bootstrap enhancement summary with statistics and subject details
        """
        # Get session-level data
        if use_cache and self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        if session_data.empty:
            return {'error': 'No session data available'}
        
        summary = {
            'total_sessions': len(session_data),
            'total_subjects': session_data['subject_id'].nunique(),
            'total_strata': session_data['strata'].nunique(),
            'bootstrap_manager_available': self.bootstrap_manager is not None,
            'feature_enhancement': {},
            'overall_enhancement': {},
            'strata_breakdown': {},
            'subject_breakdown': {},
            'enhancement_statistics': {}
        }
        
        if not summary['bootstrap_manager_available']:
            summary['error'] = 'Bootstrap manager not available'
            return summary
        
        # Get feature list
        feature_list = list(self.reference_processor.features_config.keys())
        
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

    def force_regenerate_bootstrap_indicators(self):
        """
        Force regeneration of bootstrap indicators in cached data
        
        This method clears caches and reprocesses data to ensure bootstrap
        indicators are calculated and included in all data structures.
        
        Returns:
            Dict[str, Any]: Summary of regeneration results
        """
        print("🔄 Force regenerating data with bootstrap indicators...")
        
        # Clear all caches to ensure fresh calculation
        self._invalidate_derived_caches()
        
        # Reload and reprocess data to include bootstrap indicators
        raw_data = self.reload_data()
        print(f"Reloaded {len(raw_data)} sessions")
        
        # Process pipeline with bootstrap indicators
        session_data = self.process_data_pipeline(raw_data, use_cache=False)
        print(f"Reprocessed {len(session_data)} sessions")
        
        # Verify bootstrap indicator columns are present
        bootstrap_cols = [col for col in session_data.columns if col.endswith('_bootstrap_enhanced')]
        print(f"✅ Generated {len(bootstrap_cols)} bootstrap indicator columns")
        
        # Get table display data to verify UI structures include bootstrap indicators
        table_data = self.get_table_display_data(use_cache=False)
        if table_data:
            table_df = pd.DataFrame(table_data)
            table_bootstrap_cols = [col for col in table_df.columns if 'bootstrap_enhanced' in col]
            print(f"✅ Table display includes {len(table_bootstrap_cols)} bootstrap indicator columns")
        
        # Generate bootstrap enhancement summary
        summary = self.get_bootstrap_enhancement_summary(use_cache=False)
        
        if 'error' not in summary:
            enhancement_count = summary['enhancement_statistics']['total_actual_feature_enhancements']
            total_possible = summary['enhancement_statistics']['total_possible_feature_enhancements']
            enhancement_rate = summary['enhancement_statistics']['overall_enhancement_rate']
            
            print(f"📊 Bootstrap enhancement statistics:")
            print(f"   - Enhanced sessions: {enhancement_count}/{total_possible} ({enhancement_rate}%)")
            print(f"   - Subjects with enhancement: {summary['enhancement_statistics']['subjects_with_any_enhancement']}")
            print(f"   - Strata with enhancement: {summary['enhancement_statistics']['strata_with_any_enhancement']}")
        
        print("✅ Bootstrap indicators regeneration complete!")
        
        return {
            'bootstrap_columns_generated': len(bootstrap_cols),
            'table_bootstrap_columns': len(table_bootstrap_cols) if table_data else 0,
            'enhancement_summary': summary
        }

    def calculate_session_bootstrap_cis(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bootstrap CIs for all session rolling averages during pipeline processing
        
        This pre-computes bootstrap CIs once and stores them in session data to avoid
        expensive real-time calculations during UI creation.
        
        Parameters:
            session_data: pd.DataFrame
                Session-level data with rolling averages and percentiles
                
        Returns:
            pd.DataFrame
                Session data with bootstrap CI columns added
        """
        print("🚀 PHASE 1 OPTIMIZATION: Pre-computing bootstrap CIs during pipeline...")
        
        if self.bootstrap_manager is None:
            print("⚠️  Bootstrap manager not available - skipping bootstrap CI calculation")
            return session_data
        
        result_df = session_data.copy()
        features = list(self.reference_processor.features_config.keys())
        
        # Counters for reporting
        total_ci_calculations = 0
        successful_ci_calculations = 0
        strata_processed = set()
        
        # Process by strata for efficiency (shared reference data)
        for strata, strata_sessions in result_df.groupby('strata'):
            strata_processed.add(strata)
            
            # Get reference data for this strata once
            if hasattr(self, 'quantile_analyzer') and self.quantile_analyzer is not None:
                strata_data = self.quantile_analyzer.percentile_data.get(strata)
                
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
                        ci_lower, ci_upper = self.bootstrap_manager.statistical_utils.calculate_bootstrap_raw_value_ci(
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
                        
                        ci_lower, ci_upper = self.bootstrap_manager.statistical_utils.calculate_bootstrap_raw_value_ci(
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
        print(f"✅ Bootstrap CI pre-computation complete:")
        print(f"   - {successful_ci_calculations}/{total_ci_calculations} successful calculations ({(successful_ci_calculations/total_ci_calculations*100):.1f}%)")
        print(f"   - {len(strata_processed)} strata processed")
        print(f"   - Reduced bootstrap samples from 500 to 150 for better performance")
        
        return result_df

