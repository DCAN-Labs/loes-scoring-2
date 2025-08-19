#!/usr/bin/env python
"""
Comprehensive test runner for Loes Scoring System.
Runs all unit tests with coverage reporting and detailed output.
"""

import sys
import os
import unittest
import argparse
from pathlib import Path
import time
import json
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_times = []
    
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self.test_start_time
        self.test_times.append((test, elapsed))
        if self.showAll:
            self.stream.writeln(f"{self.GREEN}✓ PASS{self.RESET} ({elapsed:.3f}s)")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ ERROR{self.RESET}")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ FAIL{self.RESET}")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"{self.YELLOW}⊘ SKIP{self.RESET}: {reason}")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""
    resultclass = ColoredTextTestResult
    
    def run(self, test):
        """Run test suite with enhanced reporting."""
        print(f"\n{ColoredTextTestResult.BOLD}{'='*70}{ColoredTextTestResult.RESET}")
        print(f"{ColoredTextTestResult.BLUE}Loes Scoring System - Test Suite{ColoredTextTestResult.RESET}")
        print(f"{ColoredTextTestResult.BOLD}{'='*70}{ColoredTextTestResult.RESET}\n")
        
        result = super().run(test)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result):
        """Print test summary with statistics."""
        print(f"\n{ColoredTextTestResult.BOLD}{'='*70}{ColoredTextTestResult.RESET}")
        print(f"{ColoredTextTestResult.BLUE}Test Summary{ColoredTextTestResult.RESET}")
        print(f"{ColoredTextTestResult.BOLD}{'='*70}{ColoredTextTestResult.RESET}\n")
        
        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        success = total - failures - errors - skipped
        
        # Calculate pass rate
        pass_rate = (success / total * 100) if total > 0 else 0
        
        # Print statistics
        print(f"Total Tests: {total}")
        print(f"{ColoredTextTestResult.GREEN}Passed: {success}{ColoredTextTestResult.RESET}")
        
        if failures > 0:
            print(f"{ColoredTextTestResult.RED}Failed: {failures}{ColoredTextTestResult.RESET}")
        
        if errors > 0:
            print(f"{ColoredTextTestResult.RED}Errors: {errors}{ColoredTextTestResult.RESET}")
        
        if skipped > 0:
            print(f"{ColoredTextTestResult.YELLOW}Skipped: {skipped}{ColoredTextTestResult.RESET}")
        
        # Pass rate with color coding
        if pass_rate >= 90:
            color = ColoredTextTestResult.GREEN
        elif pass_rate >= 70:
            color = ColoredTextTestResult.YELLOW
        else:
            color = ColoredTextTestResult.RED
        
        print(f"\nPass Rate: {color}{pass_rate:.1f}%{ColoredTextTestResult.RESET}")
        
        # Print slowest tests
        if hasattr(result, 'test_times') and result.test_times:
            print(f"\n{ColoredTextTestResult.BLUE}Slowest Tests:{ColoredTextTestResult.RESET}")
            sorted_times = sorted(result.test_times, key=lambda x: x[1], reverse=True)
            for test, elapsed in sorted_times[:5]:
                test_name = test.id().split('.')[-1]
                print(f"  {test_name}: {elapsed:.3f}s")


def discover_tests(test_dir='tests', pattern='test*.py'):
    """Discover all test files in the test directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    return suite


def run_specific_module(module_name):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    try:
        module = __import__(f'tests.unit.{module_name}', fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        return suite
    except ImportError as e:
        print(f"Error: Could not import module '{module_name}': {e}")
        return None


def generate_coverage_report():
    """Generate coverage report for the tests."""
    try:
        import coverage
        
        cov = coverage.Coverage(source=['src'])
        cov.start()
        
        # Run tests
        suite = discover_tests()
        runner = unittest.TextTestRunner(verbosity=0, stream=StringIO())
        runner.run(suite)
        
        cov.stop()
        cov.save()
        
        # Print coverage report
        print("\n" + "="*70)
        print("Coverage Report")
        print("="*70)
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='htmlcov')
        print("\nDetailed HTML coverage report generated in 'htmlcov/' directory")
        
    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='Run Loes Scoring System tests')
    parser.add_argument(
        '--module', '-m',
        help='Run specific test module (e.g., test_metrics)',
        default=None
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--failfast', '-f',
        action='store_true',
        help='Stop on first failure'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='test*.py',
        help='Test file pattern (default: test*.py)'
    )
    
    args = parser.parse_args()
    
    # Generate coverage report if requested
    if args.coverage:
        generate_coverage_report()
        return
    
    # Determine test suite
    if args.module:
        suite = run_specific_module(args.module)
        if suite is None:
            sys.exit(1)
    else:
        suite = discover_tests(pattern=args.pattern)
    
    # Configure runner
    verbosity = 2 if args.verbose else 1
    runner = ColoredTextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Exit with appropriate code
    if result.failures or result.errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()