from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class UIDataManager:
    """
    UI Data Manager for creating optimized data structures for fast component rendering
    
    This class handles all UI-specific data transformations and optimizations that were
    previously mixed with business logic in AppUtils. It focuses solely on preparing
    data for UI components with optimal performance.
    """

    def __init__(self):
        """Initialize UI Data Manager"""
        self.features = ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)']
        
    def map_percentile_to_category(self, percentile: float) -> str:
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
    
    def get_strata_abbreviation(self, strata: str) -> str:
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
    
    def optimize_session_data_storage(self, session_data: pd.DataFrame, bootstrap_manager=None, cache_manager=None) -> Dict[str, Any]:
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
            bootstrap_manager: Optional bootstrap manager for coverage stats
            cache_manager: Optional cache manager for data hashing
                
        Returns:
            Dict[str, Any]: Optimized storage structure with bootstrap support
        """
        print("Optimizing session data storage...")
        
        # Handle empty DataFrame case
        if session_data.empty or 'subject_id' not in session_data.columns:
            return {
                'subjects': {},
                'strata_reference': {},
                'bootstrap_coverage': {},
                'metadata': {
                    'total_subjects': 0,
                    'total_sessions': 0,
                    'total_strata': 0,
                    'data_hash': '',
                    'phase3_enhanced': True,
                    'bootstrap_enabled_strata': 0,
                    'optimization_timestamp': datetime.now().isoformat()
                }
            }
        
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
            
            # PHASE 3: Check for bootstrap availability and calculate coverage statistics
            bootstrap_enabled = False
            feature_bootstrap_coverage = {}
            
            if bootstrap_manager is not None:
                # Check if bootstrap is available for this strata and any features
                for feature_col in processed_features:
                    feature_name = feature_col.replace('_processed_rolling_avg', '')
                    if bootstrap_manager.is_bootstrap_available(strata, feature_name):
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
                'subject_count': len(strata_sessions['subject_id'].unique()),
                'session_count': len(strata_sessions)
            }
            
            # Create strata reference even if no processed features exist
            reference_distributions = {}
            if processed_features:
                reference_data = strata_sessions[processed_features + ['subject_id']].dropna()
                reference_distributions = {
                    feature: reference_data[feature].values.tolist() 
                    for feature in processed_features
                    if not reference_data[feature].isna().all()
                }
            
            strata_reference[strata] = {
                'subject_count': len(strata_sessions['subject_id'].unique()),
                'session_count': len(strata_sessions),
                'reference_distributions': reference_distributions,
                # PHASE 3: Add bootstrap indicator to strata reference
                'bootstrap_enabled': bootstrap_enabled
            }
        
        # Create optimized storage structure with Phase 3 enhancements
        data_hash = self._calculate_data_hash(session_data, cache_manager)
        optimized_storage = {
            'subjects': subject_data,
            'strata_reference': strata_reference,
            'metadata': {
                'total_subjects': len(subject_data),
                'total_sessions': len(session_data),
                'total_strata': len(strata_reference),
                'storage_timestamp': pd.Timestamp.now(),
                'data_hash': data_hash,
                # PHASE 3: Bootstrap metadata
                'bootstrap_enabled_strata_count': len(bootstrap_enabled_strata_set),
                'bootstrap_enabled_strata_list': list(bootstrap_enabled_strata_set),
                'phase3_enhanced': True
            },
            # PHASE 3: Bootstrap coverage statistics as separate cache structure
            'bootstrap_coverage': bootstrap_coverage
        }
        
        print(f"Optimized storage created:")
        print(f"  - {len(subject_data)} subjects")
        print(f"  - {len(strata_reference)} strata references")
        print(f"  - {len(session_data)} total sessions")
        # PHASE 3: Report bootstrap enhancement status
        print(f"  {len(bootstrap_enabled_strata_set)} strata with bootstrap enhancement")
        if bootstrap_coverage:
            bootstrap_features = sum(len(coverage['feature_coverage']) for coverage in bootstrap_coverage.values())
            print(f" {bootstrap_features} feature-strata combinations analyzed for bootstrap coverage")
        
        return optimized_storage
    
    def _calculate_data_hash(self, df: pd.DataFrame, cache_manager=None) -> str:
        """Calculate a hash for data validation"""
        if cache_manager is not None and hasattr(cache_manager, 'calculate_data_hash'):
            return cache_manager.calculate_data_hash(df)
        
        # Fallback hash calculation
        import hashlib
        data_str = f"{len(df)}_{df['subject_id'].nunique()}_{df['session_date'].max()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def create_ui_optimized_structures(self, session_data: pd.DataFrame, bootstrap_manager=None) -> Dict[str, Any]:
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
            bootstrap_manager: Optional bootstrap manager for CI calculations
                
        Returns:
            Dict[str, Any]: UI-optimized data structures
        """
        print("Creating UI-optimized data structures...")
        
        # DEBUG: Check what columns are available in session_data
        print(f" Session data columns ({len(session_data.columns)} total):")
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
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            # Get most recent session data
            latest_session = subject_sessions.sort_values('session_date').iloc[-1]
            
            feature_ranks = {}
            for feature in self.features:
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
        ui_structures['time_series_data'] = self._create_time_series_data(session_data, bootstrap_manager)
        
        # 5. Table Display Cache
        ui_structures['table_display_cache'] = self._create_table_display_cache(session_data, bootstrap_manager)
        
        print(f"UI structures created:")
        print(f"  - Feature rank data: {len(ui_structures['feature_rank_data'])} subjects")
        print(f"  - Subject lookups: {len(ui_structures['subject_lookup'])} subjects")
        print(f"  - Strata lookups: {len(ui_structures['strata_lookup'])} strata")
        print(f"  - Time series data: {len(ui_structures['time_series_data'])} subjects")
        print(f"  - Table display cache: {len(ui_structures['table_display_cache'])} rows")
        
        return ui_structures 
    
    def _create_time_series_data(self, session_data: pd.DataFrame, bootstrap_manager=None) -> Dict[str, Any]:
        """Create time series data for visualization components"""
        time_series_data = {}
        
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
            for feature in self.features:
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
                
                # Add bootstrap CIs for raw rolling averages (separate from percentile CIs)
                if bootstrap_manager is not None:
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
            
            time_series_data[subject_id] = time_series
        
        return time_series_data
    
    def _create_table_display_cache(self, session_data: pd.DataFrame, bootstrap_manager=None) -> List[Dict[str, Any]]:
        """Create table display cache for fast rendering"""
        # Get most recent session for each subject
        most_recent = session_data.sort_values('session_date').groupby('subject_id').last().reset_index()
        
        # Initialize threshold analyzer for table display
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
        
        # Import threshold analyzer
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
                'strata_abbr': self.get_strata_abbreviation(row['strata']),
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
                # Set computed threshold alert values
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
            for feature in self.features:
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
                
                # Add rolling average columns to table display cache
                display_row[f"{feature}_processed_rolling_avg"] = row.get(rolling_avg_col)
                
                # Wilson CI columns (for percentile CIs)
                display_row[f"{feature}_session_percentile_ci_lower"] = row.get(ci_lower_col)
                display_row[f"{feature}_session_percentile_ci_upper"] = row.get(ci_upper_col)
                
                # PHASE 3: Add bootstrap indicator columns to table display cache
                display_row[f"{feature}_bootstrap_enhanced"] = row.get(bootstrap_indicator_col, False)
                
                # Add bootstrap CIs for raw rolling averages - use pre-computed values from session data
                bootstrap_ci_lower_col = f"{feature}_bootstrap_ci_lower"
                bootstrap_ci_upper_col = f"{feature}_bootstrap_ci_upper"
                
                # Get pre-computed bootstrap CI values from the session data
                ci_lower_value = row.get(bootstrap_ci_lower_col, np.nan)
                ci_upper_value = row.get(bootstrap_ci_upper_col, np.nan)
                
                display_row[f"{feature}_bootstrap_ci_lower"] = ci_lower_value
                display_row[f"{feature}_bootstrap_ci_upper"] = ci_upper_value
                
                # Calculate CI width if both bounds are available
                if not pd.isna(ci_lower_value) and not pd.isna(ci_upper_value):
                    ci_width = ci_upper_value - ci_lower_value
                    display_row[f"{feature}_bootstrap_ci_width"] = ci_width
                    
                    # MODERATE: Use improved but more lenient certainty classification
                    rolling_avg_value = row.get(rolling_avg_col, 0)
                    display_row[f"{feature}_bootstrap_ci_certainty"] = self._calculate_ci_certainty_moderate(
                        ci_width, rolling_avg_value, feature
                    )
                else:
                    display_row[f"{feature}_bootstrap_ci_width"] = np.nan
                    display_row[f"{feature}_bootstrap_ci_certainty"] = 'intermediate'
            
            # Add overall percentile CI columns (Wilson CIs)
            overall_ci_lower_col = "session_overall_percentile_ci_lower"
            overall_ci_upper_col = "session_overall_percentile_ci_upper"
            display_row[overall_ci_lower_col] = row.get(overall_ci_lower_col)
            display_row[overall_ci_upper_col] = row.get(overall_ci_upper_col)
            
            # Add overall bootstrap CIs for table display - use pre-computed values from session data
            overall_bootstrap_ci_lower_col = "session_overall_bootstrap_ci_lower"
            overall_bootstrap_ci_upper_col = "session_overall_bootstrap_ci_upper"
            
            # Get pre-computed overall bootstrap CI values from the session data
            overall_ci_lower_value = row.get(overall_bootstrap_ci_lower_col, np.nan)
            overall_ci_upper_value = row.get(overall_bootstrap_ci_upper_col, np.nan)
            
            display_row["session_overall_bootstrap_ci_lower"] = overall_ci_lower_value
            display_row["session_overall_bootstrap_ci_upper"] = overall_ci_upper_value
            
            # Calculate overall CI width if both bounds are available
            if not pd.isna(overall_ci_lower_value) and not pd.isna(overall_ci_upper_value):
                overall_ci_width = overall_ci_upper_value - overall_ci_lower_value
                display_row["session_overall_bootstrap_ci_width"] = overall_ci_width
                
                # MODERATE: Use improved but more lenient certainty classification for overall percentile
                overall_rolling_avg = row.get('session_overall_rolling_avg', 0)
                display_row["session_overall_bootstrap_ci_certainty"] = self._calculate_ci_certainty_moderate(
                    overall_ci_width, overall_rolling_avg
                )
            else:
                display_row["session_overall_bootstrap_ci_width"] = np.nan
                display_row["session_overall_bootstrap_ci_certainty"] = 'intermediate'
            
            table_data.append(display_row)
        
        return table_data
    
    def get_subject_display_data(self, subject_id: str, ui_structures: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimized subject data for UI display components
        
        Parameters:
            subject_id: str
                Subject ID to get display data for
            ui_structures: Dict[str, Any]
                Pre-computed UI structures
                
        Returns:
            Dict[str, Any]: Subject display data optimized for UI rendering
        """
        return ui_structures.get('subject_lookup', {}).get(subject_id, {})
    
    def get_table_display_data(self, ui_structures: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get optimized table display data for fast rendering
        
        Parameters:
            ui_structures: Dict[str, Any]
                Pre-computed UI structures
                
        Returns:
            List[Dict[str, Any]]: Table data optimized for UI rendering
        """
        return ui_structures.get('table_display_cache', [])
    
    def get_time_series_data(self, subject_id: str, ui_structures: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimized time series data for visualization components
        
        Parameters:
            subject_id: str
                Subject ID to get time series data for
            ui_structures: Dict[str, Any]
                Pre-computed UI structures
                
        Returns:
            Dict[str, Any]: Time series data optimized for UI rendering
        """
        return ui_structures.get('time_series_data', {}).get(subject_id, {}) 

    def _calculate_ci_certainty_moderate(self, ci_width: float, target_value: float, feature_name: str = None) -> str:
        """
        Calculate CI certainty classification using moderate relative width thresholds
        
        Uses more lenient thresholds than the strict version to avoid too many "uncertain" classifications.
        Aims for roughly: 50% certain, 30% intermediate, 20% uncertain
        
        Parameters:
            ci_width: float
                Absolute CI width (upper - lower)
            target_value: float
                Target value (rolling average) that the CI is for
            feature_name: str, optional
                Feature name for feature-specific thresholds
                
        Returns:
            str: 'certain', 'intermediate', or 'uncertain'
        """
        if pd.isna(ci_width) or pd.isna(target_value) or target_value == 0:
            return 'intermediate'
        
        # Calculate relative CI width as percentage
        relative_width = abs(ci_width / target_value)
        
        # MODERATE feature-specific thresholds (more lenient than strict version)
        if feature_name:
            # More moderate thresholds - easier to be "certain", harder to be "uncertain"
            feature_thresholds = {
                'finished_trials': {'certain': 0.10, 'uncertain': 0.40},      # 10% / 40% of trial count (was 5%/20%)
                'ignore_rate': {'certain': 0.15, 'uncertain': 0.50},          # 15% / 50% of ignore rate (was 10%/30%) 
                'total_trials': {'certain': 0.10, 'uncertain': 0.40},         # 10% / 40% of trial count (was 5%/20%)
                'foraging_performance': {'certain': 0.12, 'uncertain': 0.35}, # 12% / 35% of performance (was 8%/25%)
                'abs(bias_naive)': {'certain': 0.20, 'uncertain': 0.60}       # 20% / 60% of bias value (was 15%/40%)
            }
            
            thresholds = feature_thresholds.get(feature_name, {'certain': 0.12, 'uncertain': 0.35})
        else:
            # Default moderate thresholds for overall percentiles
            thresholds = {'certain': 0.12, 'uncertain': 0.35}  # 12% / 35% relative width (was 8%/25%)
        
        # Classify based on relative width
        if relative_width < thresholds['certain']:
            return 'certain'
        elif relative_width > thresholds['uncertain']:
            return 'uncertain'
        else:
            return 'intermediate' 