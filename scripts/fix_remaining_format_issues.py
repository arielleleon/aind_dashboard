#!/usr/bin/env python3
"""
Script to fix remaining code formatting issues.
"""

import re


def fix_psutil_import():
    """Fix the missing psutil import in cache_utils.py."""
    file_path = "app_utils/cache_utils.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if psutil import already exists
        if 'import psutil' not in content:
            # Find the import section and add psutil
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i
            
            # Insert psutil import after other imports
            lines.insert(import_section_end + 1, 'import psutil')
            content = '\n'.join(lines)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✓ Added psutil import to {file_path}")
        else:
            print(f"✓ psutil import already exists in {file_path}")
            
    except Exception as e:
        print(f"✗ Error fixing psutil import: {e}")


def fix_whitespace_before_colon():
    """Fix E203: whitespace before ':' in app_subject_timeseries.py:317."""
    file_path = "app_elements/app_subject_detail/app_subject_timeseries.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix whitespace before colon in slice notation
        content = re.sub(r'\s+:', ':', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed whitespace before colon in {file_path}")
        
    except Exception as e:
        print(f"✗ Error fixing whitespace before colon: {e}")


def fix_arithmetic_spacing():
    """Fix E226: missing whitespace around arithmetic operators."""
    file_path = "callbacks/shared_callback_utils.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # More specific regex for arithmetic operators
        # Look for patterns like "variable+variable" or "number*number"
        content = re.sub(r'(\w|\))([+\-*/])(\w|\()', r'\1 \2 \3', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed arithmetic spacing in {file_path}")
        
    except Exception as e:
        print(f"✗ Error fixing arithmetic spacing: {e}")


def fix_comparison_to_true():
    """Fix E712: comparison to True should use 'is True' or just the condition."""
    file_path = "app_utils/app_alerts/alert_service.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find and fix comparison to True
        # Replace "== True" with "is True" or just the variable
        content = re.sub(r'(\w+)\s*==\s*True\b', r'\1 is True', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed comparison to True in {file_path}")
        
    except Exception as e:
        print(f"✗ Error fixing comparison to True: {e}")


def fix_bare_except():
    """Fix E722: bare 'except:' clause."""
    file_path = "app_elements/app_subject_detail/app_subject_percentile_timeseries.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace bare except with except Exception
        content = re.sub(r'\bexcept\s*:', 'except Exception:', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed bare except clause in {file_path}")
        
    except Exception as e:
        print(f"✗ Error fixing bare except: {e}")


def fix_line_break_after_operator():
    """Fix W504: line break after binary operator."""
    files_to_fix = [
        "app_utils/app_alerts/alert_service.py",
        "app_utils/filter_utils.py"
    ]
    
    for file_path in files_to_fix:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for i, line in enumerate(lines):
                # Look for lines ending with binary operators
                if re.search(r'\s+(and|or|\+|\-|\*|/|%|==|!=|<=|>=|<|>)\s*$', line.strip()):
                    # Move the operator to the next line
                    stripped = line.strip()
                    operator_match = re.search(r'(\s+)(and|or|\+|\-|\*|/|%|==|!=|<=|>=|<|>)\s*$', stripped)
                    if operator_match and i + 1 < len(lines):
                        operator = operator_match.group(2)
                        line_without_op = stripped[:operator_match.start()]
                        next_line = lines[i + 1]
                        
                        # Reconstruct the lines
                        fixed_lines.append(line_without_op + '\n')
                        lines[i + 1] = '        ' + operator + ' ' + next_line.lstrip()
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            with open(file_path, 'w') as f:
                f.writelines(fixed_lines)
            
            print(f"✓ Fixed line break after operator in {file_path}")
            
        except Exception as e:
            print(f"✗ Error fixing line break after operator in {file_path}: {e}")


def main():
    """Main execution."""
    print("🔧 Fixing Remaining Code Formatting Issues")
    print("=" * 50)
    
    print("\n1. Fixing missing psutil import...")
    fix_psutil_import()
    
    print("\n2. Fixing whitespace before colon...")
    fix_whitespace_before_colon()
    
    print("\n3. Fixing arithmetic operator spacing...")
    fix_arithmetic_spacing()
    
    print("\n4. Fixing comparison to True...")
    fix_comparison_to_true()
    
    print("\n5. Fixing bare except clause...")
    fix_bare_except()
    
    print("\n6. Fixing line breaks after operators...")
    fix_line_break_after_operator()
    
    print("\n🎉 Remaining formatting issues fixed!")
    print("Run 'make lint' to verify the fixes.")


if __name__ == "__main__":
    main() 