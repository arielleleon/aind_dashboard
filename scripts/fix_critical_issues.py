#!/usr/bin/env python3
"""
Script to fix critical linting issues automatically.
"""

import re
import subprocess
from pathlib import Path


def fix_whitespace_issues():
    """Fix whitespace issues in callback files."""
    callback_files = [
        "callbacks/session_interaction_callbacks.py",
        "callbacks/table_callbacks.py", 
        "callbacks/tooltip_callbacks.py"
    ]
    
    for file_path in callback_files:
        try:
            print(f"Fixing whitespace in {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove trailing whitespace and blank line whitespace
            lines = content.split('\n')
            cleaned_lines = [line.rstrip() for line in lines]
            cleaned_content = '\n'.join(cleaned_lines)
            
            with open(file_path, 'w') as f:
                f.write(cleaned_content)
                
            print(f"✓ Fixed whitespace in {file_path}")
            
        except Exception as e:
            print(f"✗ Error fixing {file_path}: {e}")


def fix_spacing_issues():
    """Fix spacing around operators."""
    files_to_fix = [
        "app_utils/percentile_utils.py",
        "callbacks/shared_callback_utils.py"
    ]
    
    for file_path in files_to_fix:
        try:
            print(f"Fixing spacing in {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix missing whitespace around arithmetic operators
            # E226: missing whitespace around arithmetic operator
            content = re.sub(r'(\w)([+\-*/%])(\w)', r'\1 \2 \3', content)
            
            with open(file_path, 'w') as f:
                f.write(content)
                
            print(f"✓ Fixed spacing in {file_path}")
            
        except Exception as e:
            print(f"✗ Error fixing {file_path}: {e}")


def fix_f_string_issues():
    """Fix f-strings missing placeholders."""
    # These need manual review, but we can identify them
    print("F-string placeholders need manual review in:")
    f_string_files = [
        "app_elements/app_content/app_dataframe/app_dataframe.py:107",
        "app_elements/app_subject_detail/app_session_card.py:208",
        "app_elements/app_subject_detail/app_subject_percentile_timeseries.py:356,374,557,574",
        "app_elements/app_subject_detail/app_subject_timeseries.py:211",
        "app_utils/app_analysis/quantile_analyzer.py:79,361",
        "app_utils/app_analysis/reference_processor.py:317",
        "app_utils/ui_utils.py:909",
        "callbacks/subject_detail_callbacks.py:70,82,88"
    ]
    
    for location in f_string_files:
        print(f"  - {location}")


def check_critical_imports():
    """Check for critical missing imports."""
    print("Critical import issues found:")
    print("  - app_utils/cache_utils.py: Missing 'import psutil' (lines 124, 201)")
    print("  - Consider adding: pip install psutil")


def main():
    """Main execution."""
    print("🔧 Fixing Critical Code Quality Issues")
    print("=" * 50)
    
    print("\n1. Fixing whitespace issues...")
    fix_whitespace_issues()
    
    print("\n2. Fixing spacing around operators...")
    fix_spacing_issues()
    
    print("\n3. Checking f-string issues...")
    fix_f_string_issues()
    
    print("\n4. Checking critical imports...")
    check_critical_imports()
    
    print("\n🎉 Critical fixes completed!")
    print("Next steps:")
    print("  1. Review f-string placeholders manually")
    print("  2. Add missing imports (psutil)")
    print("  3. Run 'make lint' to verify fixes")


if __name__ == "__main__":
    main() 