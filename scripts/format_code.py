#!/usr/bin/env python3
"""
Comprehensive code formatting and quality check script.
Applies formatting incrementally to manage context limits and provides detailed reporting.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


class CodeQualityManager:
    """Manages code quality checks and formatting with incremental processing."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {}
        self.timestamp = datetime.now().isoformat()
        
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n {description}")
        print(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            success = result.returncode == 0
            status = " SUCCESS" if success else " FAILED"
            print(f"{status} (exit code: {result.returncode})")
            
            if result.stdout:
                print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            if result.stderr and not success:
                print("STDERR:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
                
            return {
                "command": " ".join(command),
                "description": description,
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f" ERROR: {str(e)}")
            return {
                "command": " ".join(command),
                "description": description,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def format_with_black(self, target_paths: List[str] = None) -> Dict[str, Any]:
        """Apply Black code formatting."""
        if target_paths is None:
            target_paths = ["app.py", "shared_utils.py", "app_elements/", "app_utils/", "callbacks/"]
        
        results = []
        for path in target_paths:
            if Path(self.project_root / path).exists():
                command = ["black", "--diff", path]
                result = self.run_command(command, f"Black formatting check: {path}")
                results.append(result)
                
                if result["success"]:
                    # Actually apply the formatting
                    apply_command = ["black", path]
                    apply_result = self.run_command(apply_command, f"Black formatting apply: {path}")
                    results.append(apply_result)
        
        return {"tool": "black", "results": results}
    
    def sort_imports_with_isort(self, target_paths: List[str] = None) -> Dict[str, Any]:
        """Sort imports with isort."""
        if target_paths is None:
            target_paths = ["app.py", "shared_utils.py", "app_elements/", "app_utils/", "callbacks/"]
        
        results = []
        for path in target_paths:
            if Path(self.project_root / path).exists():
                command = ["isort", "--diff", path]
                result = self.run_command(command, f"Import sorting check: {path}")
                results.append(result)
                
                if result["success"]:
                    # Actually apply the sorting
                    apply_command = ["isort", path]
                    apply_result = self.run_command(apply_command, f"Import sorting apply: {path}")
                    results.append(apply_result)
        
        return {"tool": "isort", "results": results}
    
    def lint_with_flake8(self, target_paths: List[str] = None) -> Dict[str, Any]:
        """Run flake8 linting."""
        if target_paths is None:
            target_paths = ["app.py", "shared_utils.py", "app_elements/", "app_utils/", "callbacks/"]
        
        results = []
        for path in target_paths:
            if Path(self.project_root / path).exists():
                command = ["flake8", path]
                result = self.run_command(command, f"Flake8 linting: {path}")
                results.append(result)
        
        return {"tool": "flake8", "results": results}
    
    def type_check_with_mypy(self, target_paths: List[str] = None) -> Dict[str, Any]:
        """Run mypy type checking."""
        if target_paths is None:
            target_paths = ["app.py", "shared_utils.py"]  # Start with main files
        
        results = []
        for path in target_paths:
            if Path(self.project_root / path).exists():
                command = ["mypy", path]
                result = self.run_command(command, f"MyPy type checking: {path}")
                results.append(result)
        
        return {"tool": "mypy", "results": results}
    
    def security_scan_with_bandit(self) -> Dict[str, Any]:
        """Run bandit security scanning."""
        command = ["bandit", "-r", ".", "-f", "json", "-o", "bandit_report.json"]
        result = self.run_command(command, "Bandit security scan")
        return {"tool": "bandit", "results": [result]}
    
    def run_full_pipeline(self, skip_tools: List[str] = None) -> Dict[str, Any]:
        """Run the complete code quality pipeline."""
        if skip_tools is None:
            skip_tools = []
        
        print(f"🚀 Starting comprehensive code quality pipeline at {self.timestamp}")
        print(f"Project root: {self.project_root.absolute()}")
        
        pipeline_results = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root.absolute()),
            "tools": {}
        }
        
        # Step 1: Import sorting (must come before Black)
        if "isort" not in skip_tools:
            print("\n" + "="*60)
            print("STEP 1: Import Sorting with isort")
            print("="*60)
            pipeline_results["tools"]["isort"] = self.sort_imports_with_isort()
        
        # Step 2: Code formatting with Black
        if "black" not in skip_tools:
            print("\n" + "="*60)
            print("STEP 2: Code Formatting with Black")
            print("="*60)
            pipeline_results["tools"]["black"] = self.format_with_black()
        
        # Step 3: Linting with flake8
        if "flake8" not in skip_tools:
            print("\n" + "="*60)
            print("STEP 3: Linting with flake8")
            print("="*60)
            pipeline_results["tools"]["flake8"] = self.lint_with_flake8()
        
        # Step 4: Type checking (optional, can be skipped if too many errors)
        if "mypy" not in skip_tools:
            print("\n" + "="*60)
            print("STEP 4: Type Checking with MyPy")
            print("="*60)
            pipeline_results["tools"]["mypy"] = self.type_check_with_mypy()
        
        # Step 5: Security scanning
        if "bandit" not in skip_tools:
            print("\n" + "="*60)
            print("STEP 5: Security Scanning with Bandit")
            print("="*60)
            pipeline_results["tools"]["bandit"] = self.security_scan_with_bandit()
        
        # Save results
        results_file = f"quality_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\n📊 Results saved to: {results_file}")
        self._print_summary(pipeline_results)
        
        return pipeline_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of all results."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        for tool_name, tool_results in results["tools"].items():
            total_checks = len(tool_results["results"])
            successful_checks = sum(1 for r in tool_results["results"] if r["success"])
            
            status = " " if successful_checks == total_checks else "!"
            print(f"{status} {tool_name.upper()}: {successful_checks}/{total_checks} checks passed")
            
            # Show failures
            failures = [r for r in tool_results["results"] if not r["success"]]
            for failure in failures:
                print(f"    {failure['description']}")
                if failure.get("stderr"):
                    print(f"      Error: {failure['stderr'][:100]}...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run code quality checks and formatting")
    parser.add_argument("--skip", nargs="*", default=[], 
                       choices=["isort", "black", "flake8", "mypy", "bandit"],
                       help="Skip specific tools")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory")
    
    args = parser.parse_args()
    
    manager = CodeQualityManager(args.project_root)
    results = manager.run_full_pipeline(skip_tools=args.skip)
    
    # Exit with non-zero code if there were failures
    total_failures = 0
    for tool_results in results["tools"].values():
        total_failures += sum(1 for r in tool_results["results"] if not r["success"])
    
    if total_failures > 0:
        print(f"\n  {total_failures} checks failed. Review the output above.")
        sys.exit(1)
    else:
        print(f"\n All checks passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main() 