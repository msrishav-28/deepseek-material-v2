"""Test utilities and helpers."""

import json
from pathlib import Path
from typing import Dict, Any, List


def load_test_data(filename: str) -> Dict[str, Any]:
    """Load test data from JSON file.
    
    Args:
        filename: Relative path to test data file (e.g., "materials/baseline_ceramics.json")
        
    Returns:
        Loaded JSON data as dictionary
    """
    test_data_dir = Path(__file__).parent.parent / "data"
    filepath = test_data_dir / filename
    
    with open(filepath) as f:
        return json.load(f)


def get_baseline_materials() -> List[Dict[str, Any]]:
    """Get baseline ceramic materials for testing.
    
    Returns:
        List of baseline material dictionaries
    """
    data = load_test_data("materials/baseline_ceramics.json")
    return data["materials"]


def get_material_by_formula(formula: str) -> Dict[str, Any]:
    """Get a specific material by formula.
    
    Args:
        formula: Chemical formula (e.g., "SiC", "B4C")
        
    Returns:
        Material dictionary
        
    Raises:
        ValueError: If material not found
    """
    materials = get_baseline_materials()
    
    for material in materials:
        if material["formula"] == formula:
            return material
    
    raise ValueError(f"Material with formula '{formula}' not found in test data")
