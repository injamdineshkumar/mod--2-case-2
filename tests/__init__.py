"""
Test suite for Fuzzy Logic Medical Diagnosis System
"""

import unittest

# Test discovery
def load_tests(loader, tests, pattern):
    """
    Load all test cases in the tests directory
    """
    return loader.discover('.', pattern='test_*.py')


if __name__ == '__main__':
    unittest.main()