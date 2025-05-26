#!/usr/bin/env python3
"""
Test script to verify threshold alerts are working correctly in the pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_utils import AppUtils
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
import pandas as pd

def test_threshold_alerts():
    """Test threshold alerts functionality"""
    print("=" * 60)
    print("TESTING THRESHOLD ALERTS PIPELINE")
    print("=" * 60)
    
    # Initialize app utils
    print("1. Initializing AppUtils...")
    app_utils = AppUtils()
    
    # Load data and run unified pipeline
    print("2. Loading data and running unified pipeline...")
    raw_data = app_utils.get_session_data(use_cache=False)
    print(f"   Loaded {len(raw_data)} sessions")
    
    # Process unified pipeline
    session_data = app_utils.process_data_pipeline(raw_data, use_cache=False)
    print(f"   Processed {len(session_data)} sessions through unified pipeline")
    
    # Force regenerate UI cache with threshold alerts
    print("3. Regenerating UI cache with threshold alerts...")
    threshold_count = app_utils.force_regenerate_ui_cache_with_threshold_alerts()
    print(f"   ✅ {threshold_count} threshold alerts computed in UI cache")
    
    # Test getting table display data with threshold alerts
    print("4. Testing table display data...")
    table_data = app_utils.get_table_display_data(use_cache=True)
    if table_data:
        print(f"   Retrieved {len(table_data)} subjects from table display cache")
        
        # Check threshold alert distribution
        threshold_alerts = [row['threshold_alert'] for row in table_data]
        threshold_count = sum(1 for alert in threshold_alerts if alert == 'T')
        total_sessions_alerts = sum(1 for row in table_data if 'T |' in row.get('total_sessions_alert', ''))
        stage_sessions_alerts = sum(1 for row in table_data if 'T |' in row.get('stage_sessions_alert', ''))
        water_day_alerts = sum(1 for row in table_data if 'T |' in row.get('water_day_total_alert', ''))
        
        print(f"   Threshold alert distribution:")
        print(f"     - Overall threshold alerts: {threshold_count}")
        print(f"     - Total sessions alerts: {total_sessions_alerts}")
        print(f"     - Stage sessions alerts: {stage_sessions_alerts}")
        print(f"     - Water day total alerts: {water_day_alerts}")
        
        # Show sample subjects with threshold alerts
        if threshold_count > 0:
            print(f"   Sample subjects with threshold alerts:")
            alert_subjects = [row for row in table_data if row['threshold_alert'] == 'T']
            for i, subject in enumerate(alert_subjects[:3]):  # Show first 3
                print(f"     {i+1}. {subject['subject_id']}: " +
                      f"Sessions={subject.get('session', 'N/A')}, " +
                      f"Stage={subject.get('current_stage_actual', 'N/A')}, " +
                      f"Water={subject.get('water_day_total', 'N/A')}")
                print(f"        Total Sessions Alert: {subject.get('total_sessions_alert', 'N')}")
                print(f"        Stage Sessions Alert: {subject.get('stage_sessions_alert', 'N')}")
                print(f"        Water Day Total Alert: {subject.get('water_day_total_alert', 'N')}")
    
    # Test the dataframe formatting
    print("5. Testing AppDataFrame formatting...")
    app_dataframe = AppDataFrame(app_utils=app_utils)
    formatted_df = app_dataframe.format_dataframe(raw_data)
    
    if not formatted_df.empty:
        print(f"   Formatted dataframe has {len(formatted_df)} subjects")
        
        # Check threshold alert columns in formatted dataframe
        final_threshold_count = (formatted_df['threshold_alert'] == 'T').sum()
        final_total_sessions_count = formatted_df['total_sessions_alert'].str.contains('T \|', na=False).sum()
        final_stage_sessions_count = formatted_df['stage_sessions_alert'].str.contains('T \|', na=False).sum()
        final_water_day_count = formatted_df['water_day_total_alert'].str.contains('T \|', na=False).sum()
        
        print(f"   Final formatted dataframe threshold alerts:")
        print(f"     - Overall threshold alerts: {final_threshold_count}")
        print(f"     - Total sessions alerts: {final_total_sessions_count}")
        print(f"     - Stage sessions alerts: {final_stage_sessions_count}")
        print(f"     - Water day total alerts: {final_water_day_count}")
        
        # Check combined alerts
        combined_with_threshold = formatted_df['combined_alert'].str.contains('T', na=False).sum()
        print(f"     - Combined alerts with 'T': {combined_with_threshold}")
        
        # Check if threshold alert columns exist
        threshold_columns = ['threshold_alert', 'total_sessions_alert', 'stage_sessions_alert', 'water_day_total_alert']
        existing_columns = [col for col in threshold_columns if col in formatted_df.columns]
        print(f"   Threshold alert columns present: {existing_columns}")
        
        # Show sample rows with threshold alerts
        if final_threshold_count > 0:
            print(f"   Sample formatted rows with threshold alerts:")
            alert_rows = formatted_df[formatted_df['threshold_alert'] == 'T'].head(3)
            for _, row in alert_rows.iterrows():
                print(f"     - {row['subject_id']}: {row['combined_alert']}")
    
    print("\n" + "=" * 60)
    print("THRESHOLD ALERTS TEST COMPLETE")
    print("=" * 60)
    
    return {
        'ui_cache_threshold_count': threshold_count if 'threshold_count' in locals() else 0,
        'formatted_df_threshold_count': final_threshold_count if 'final_threshold_count' in locals() else 0,
        'test_passed': threshold_count > 0 if 'threshold_count' in locals() else False
    }

if __name__ == "__main__":
    results = test_threshold_alerts()
    print(f"\nTest Results: {results}") 