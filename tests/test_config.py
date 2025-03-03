"""
Unit tests for the configuration utilities.
"""

import os
import pytest
from modules.utils.config import load_categories, get_default_categories


class TestConfigUtils:
    """Test suite for configuration utilities."""
    
    def test_load_categories_from_file(self):
        """Test loading categories from a file."""
        test_file = os.path.join('tests', 'data', 'categories', 'test_categories.txt')
        categories = load_categories(test_file)
        
        # Test file has 20 categories
        assert len(categories) == 20
        assert 'person' in categories
        assert 'dog' in categories
        assert 'cat' in categories
    
    def test_load_categories_with_max(self):
        """Test loading categories with a maximum limit."""
        test_file = os.path.join('tests', 'data', 'categories', 'test_categories.txt')
        categories = load_categories(test_file, max_categories=5)
        
        # Should only return the first 5 categories
        assert len(categories) == 5
        assert categories == ['person', 'dog', 'cat', 'chair', 'table']
    
    def test_load_categories_nonexistent_file(self):
        """Test loading categories from a non-existent file."""
        categories = load_categories('nonexistent_file.txt')
        assert len(categories) == 0  # Should return empty list
    
    def test_get_default_categories(self):
        """Test getting default categories."""
        # This test will use the real data file, so just verify it returns something
        categories = get_default_categories()
        assert len(categories) > 0
        
        # Test with a limit
        limited_categories = get_default_categories(max_categories=10)
        assert len(limited_categories) <= 10