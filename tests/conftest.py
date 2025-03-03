"""
Pytest configuration file for the vision-playground test suite.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the main application
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants for test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def test_image_path():
    """Fixture that returns the path to a test image."""
    return os.path.join(TEST_DATA_DIR, 'debug_screenshot.png')


@pytest.fixture
def test_person_image_path():
    """Fixture that returns the path to a test image containing a person."""
    return os.path.join(TEST_DATA_DIR, '20250302_202714_329652_person_0.89.jpg')


@pytest.fixture
def project_root():
    """Fixture that returns the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))