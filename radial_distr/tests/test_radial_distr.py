"""
Unit and regression test for the radial_distr package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import radial_distr


def test_radial_distr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "radial_distr" in sys.modules
