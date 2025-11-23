"""
Data conversion utilities for robust handling of heterogeneous data formats.

This module provides safe conversion functions that handle various data types
from multiple sources (JARVIS-DFT, NIST-JANAF, Materials Project, literature)
without raising exceptions.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np


def safe_float(value: Any) -> Optional[float]:
    """
    Safely convert any value to float without raising exceptions.
    
    Handles None, NaN, int, float, string, dict, list, tuple, and numpy array types.
    Returns None if conversion fails or value is invalid.
    
    Args:
        value: Value to convert to float
        
    Returns:
        Float value if conversion successful, None otherwise
        
    Examples:
        >>> safe_float(42)
        42.0
        >>> safe_float("3.14")
        3.14
        >>> safe_float({"value": 2.5})
        2.5
        >>> safe_float([1, 2, 3])
        2.0
        >>> safe_float(None)
        None
    """
    # Handle None
    if value is None:
        return None
    
    # Handle NaN (both float and numpy)
    try:
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    
    # Handle int and float directly
    if isinstance(value, (int, float)):
        try:
            result = float(value)
            if np.isinf(result):
                return None
            return result
        except (ValueError, OverflowError):
            return None
    
    # Handle string
    if isinstance(value, str):
        try:
            result = float(value)
            if np.isinf(result) or np.isnan(result):
                return None
            return result
        except (ValueError, TypeError):
            return None
    
    # Handle dictionary
    if isinstance(value, dict):
        return extract_numeric_from_dict(value)
    
    # Handle list, tuple, or numpy array
    if isinstance(value, (list, tuple, np.ndarray)):
        return extract_numeric_from_list(value)
    
    # Handle numpy scalar types
    if hasattr(value, 'item'):
        try:
            result = float(value.item())
            if np.isinf(result) or np.isnan(result):
                return None
            return result
        except (ValueError, TypeError, AttributeError):
            return None
    
    # Unable to convert
    return None


def safe_int(value: Any) -> Optional[int]:
    """
    Safely convert any value to int without raising exceptions.
    
    Handles None, NaN, int, float, string, dict, list, tuple, and numpy array types.
    Returns None if conversion fails or value is invalid.
    
    Args:
        value: Value to convert to int
        
    Returns:
        Integer value if conversion successful, None otherwise
        
    Examples:
        >>> safe_int(42)
        42
        >>> safe_int("123")
        123
        >>> safe_int(3.7)
        3
        >>> safe_int({"value": 5})
        5
        >>> safe_int(None)
        None
    """
    # First try to get a float
    float_value = safe_float(value)
    
    if float_value is None:
        return None
    
    try:
        return int(float_value)
    except (ValueError, TypeError, OverflowError):
        return None


def extract_numeric_from_dict(data: Dict, keys: Optional[List[str]] = None) -> Optional[float]:
    """
    Extract numeric value from dictionary by checking common keys.
    
    Tries common keys in order: "value", "magnitude", "mean", "0", or numeric keys.
    If keys parameter is provided, tries those first.
    
    Args:
        data: Dictionary containing numeric data
        keys: Optional list of keys to try first
        
    Returns:
        Float value if found, None otherwise
        
    Examples:
        >>> extract_numeric_from_dict({"value": 3.14})
        3.14
        >>> extract_numeric_from_dict({"magnitude": 2.5, "unit": "eV"})
        2.5
        >>> extract_numeric_from_dict({"mean": 1.5, "std": 0.2})
        1.5
    """
    if not isinstance(data, dict):
        return None
    
    # Common keys to check
    common_keys = ["value", "magnitude", "mean", "0"]
    
    # If custom keys provided, try them first
    if keys:
        all_keys = keys + common_keys
    else:
        all_keys = common_keys
    
    # Try each key
    for key in all_keys:
        if key in data:
            result = safe_float(data[key])
            if result is not None:
                return result
    
    # Try numeric keys (0, 1, 2, etc.)
    for key in data.keys():
        if isinstance(key, (int, float)):
            result = safe_float(data[key])
            if result is not None:
                return result
    
    # If only one value in dict, try to extract it
    if len(data) == 1:
        single_value = next(iter(data.values()))
        return safe_float(single_value)
    
    return None


def extract_numeric_from_list(data: Union[List, tuple, np.ndarray]) -> Optional[float]:
    """
    Extract numeric value from list, tuple, or array by computing mean.
    
    Computes mean of all numeric elements, ignoring non-numeric values.
    Returns None if no numeric values found.
    
    Args:
        data: List, tuple, or numpy array containing data
        
    Returns:
        Mean of numeric elements if any found, None otherwise
        
    Examples:
        >>> extract_numeric_from_list([1, 2, 3])
        2.0
        >>> extract_numeric_from_list([1.5, 2.5, 3.5])
        2.5
        >>> extract_numeric_from_list([1, "invalid", 3])
        2.0
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        return None
    
    # Handle empty sequences
    if len(data) == 0:
        return None
    
    # Convert to list if numpy array
    if isinstance(data, np.ndarray):
        try:
            data = data.tolist()
        except (ValueError, TypeError):
            return None
    
    # Extract numeric values
    numeric_values = []
    for item in data:
        value = safe_float(item)
        if value is not None:
            numeric_values.append(value)
    
    # Return mean if any numeric values found
    if numeric_values:
        return float(np.mean(numeric_values))
    
    return None
