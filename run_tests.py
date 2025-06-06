#!/usr/bin/env python3
"""
Test runner script for AIND Dashboard

This script provides easy access to different types of tests and testing workflows.
It's designed to work with the refactor-friendly test structure.

Usage:
    python run_tests.py --help
    python run_tests.py --smoke          # Run only smoke tests
    python run_tests.py --unit           # Run only unit tests  
    python run_tests.py --e2e            # Run only E2E tests
    python run_tests.py --all            # Run all tests
    python run_tests.py --fast           # Run fast tests (no E2E)
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def check_environment():
    """Check if the current environment is ready for testing"""
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT CHECK")
    print("="*60)
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda environment: {conda_env}")
    else:
        print("⚠️  Not in a conda environment")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"🐍 Python version: {python_version}")
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"📁 Working directory: {cwd}")
    
    # Check if test directories exist
    test_dirs = ['tests', 'tests/unit', 'tests/e2e']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"✅ {test_dir}/ directory found")
        else:
            print(f"❌ {test_dir}/ directory NOT found")
    
    # Try importing some critical modules to check dependencies
    critical_imports = [
        'dash',
        'pandas', 
        'numpy',
        'plotly'
    ]
    
    print("\n📦 Checking critical dependencies:")
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - NOT FOUND")


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for AIND Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --smoke     # Quick baseline verification
  python run_tests.py --unit      # Detailed unit tests
  python run_tests.py --e2e       # End-to-end browser tests
  python run_tests.py --all       # Complete test suite
  python run_tests.py --fast      # Unit tests + integration (no E2E)
  python run_tests.py --coverage  # Run with detailed coverage report
        """
    )
    
    parser.add_argument(
        '--smoke', action='store_true',
        help='Run smoke tests only (quick baseline verification)'
    )
    parser.add_argument(
        '--unit', action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--e2e', action='store_true',
        help='Run end-to-end tests only'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run integration tests only'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Run fast tests (unit + integration, no E2E)'
    )
    parser.add_argument(
        '--coverage', action='store_true',
        help='Run with detailed coverage reporting'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--install-deps', action='store_true',
        help='Install testing dependencies first'
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("AIND Dashboard Test Runner")
    print(f"Working directory: {project_root}")
    
    # Install dependencies if requested
    if args.install_deps:
        print("\nInstalling testing dependencies...")
        install_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"]
        if not run_command(install_cmd, "Installing testing dependencies"):
            return 1
    
    # Check dependencies
    if not check_environment():
        return 1
    
    # Build pytest command based on arguments
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov", "--cov-report=html", "--cov-report=term"])
    
    success = True
    
    # Determine which tests to run
    if args.smoke:
        cmd = pytest_cmd + ["-m", "smoke", "tests/e2e/test_app_smoke.py"]
        success = run_command(cmd, "Smoke Tests - Baseline Verification")
        
    elif args.unit:
        cmd = pytest_cmd + ["-m", "unit", "tests/unit/"]
        success = run_command(cmd, "Unit Tests - Individual Components")
        
    elif args.e2e:
        cmd = pytest_cmd + ["-m", "e2e", "tests/e2e/"]
        success = run_command(cmd, "End-to-End Tests - Full Application")
        
    elif args.integration:
        cmd = pytest_cmd + ["-m", "integration", "tests/"]
        success = run_command(cmd, "Integration Tests - Component Interactions")
        
    elif args.fast:
        # Run unit and integration tests (skip E2E)
        cmd = pytest_cmd + ["-m", "not e2e", "tests/"]
        success = run_command(cmd, "Fast Tests - Unit + Integration (No E2E)")
        
    elif args.all:
        # Run all tests
        cmd = pytest_cmd + ["tests/"]
        success = run_command(cmd, "Complete Test Suite - All Tests")
        
    else:
        # Default: run smoke tests as a quick verification
        print("No specific test type specified. Running smoke tests as baseline verification.")
        cmd = pytest_cmd + ["tests/e2e/test_app_smoke.py"]
        success = run_command(cmd, "Default Smoke Tests")
    
    # Print summary
    if success:
        print("Tests completed successfully!")
        if args.coverage:
            print("Coverage report generated in htmlcov/index.html")
    else:
        print("Some tests failed!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 