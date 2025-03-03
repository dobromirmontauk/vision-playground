"""
Configuration utilities for the application.
"""

import os
from typing import List, Dict, Any, Optional
import importlib.resources

# Define constants
DEFAULT_CATEGORIES_FILE = os.path.join('data', 'categories', 'household_items.txt')
DEFAULT_MAX_CATEGORIES = 1000  # Maximum number of categories to load


def load_categories(categories_file: Optional[str] = None, 
                   max_categories: int = DEFAULT_MAX_CATEGORIES) -> List[str]:
    """
    Load a list of categories from a file.
    
    Args:
        categories_file: Path to the categories file (if None, uses default)
        max_categories: Maximum number of categories to load
        
    Returns:
        List of category names
    """
    # Use default file if none specified
    if categories_file is None:
        categories_file = DEFAULT_CATEGORIES_FILE
        
    # Check if file exists
    if not os.path.exists(categories_file):
        print(f"Warning: Categories file not found: {categories_file}")
        return []
        
    try:
        # Read the file
        with open(categories_file, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
            
        # Apply max limit
        if max_categories > 0 and len(categories) > max_categories:
            categories = categories[:max_categories]
            
        print(f"Loaded {len(categories)} categories from {categories_file}")
        return categories
        
    except Exception as e:
        print(f"Error loading categories: {e}")
        return []


def get_default_categories(max_categories: int = DEFAULT_MAX_CATEGORIES) -> List[str]:
    """
    Get the default categories for object detection.
    
    Args:
        max_categories: Maximum number of categories to return
        
    Returns:
        List of default category names
    """
    return load_categories(DEFAULT_CATEGORIES_FILE, max_categories)