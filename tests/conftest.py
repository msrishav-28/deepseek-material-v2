"""Pytest configuration and fixtures."""

import pytest
import tempfile
import numpy as np
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "scientific: mark test as scientific accuracy test"
    )
    config.addinivalue_line(
        "markers", "reproducibility: mark test as reproducibility test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_materials_project_api():
    """Mock Materials Project API for testing."""
    # TODO: Implement in Task 2
    pass


@pytest.fixture
def sample_material_data():
    """Provide sample material data for testing."""
    return {
        "material_id": "mp-149",
        "base_composition": "SiC",
        "dopant_element": "B",
        "dopant_concentration": 0.02,
        "energy_above_hull": 0.05,  # Metastable, should pass criteria
        "formation_energy": -1.2,
    }


@pytest.fixture
def sample_properties():
    """Provide sample material properties."""
    return {
        "hardness": 28.0,  # GPa
        "fracture_toughness": 4.5,  # MPa·m^0.5
        "density": 3.21,  # g/cm³
        "youngs_modulus": 410.0,  # GPa
        "thermal_conductivity_1000C": 45.0,  # W/m·K
    }


@pytest.fixture
def sample_training_data():
    """Provide sample ML training data."""
    np.random.seed(42)
    n_samples = 100
    
    X = {
        "hardness": np.random.uniform(20, 35, n_samples),
        "fracture_toughness": np.random.uniform(3, 6, n_samples),
        "density": np.random.uniform(2.5, 4.0, n_samples),
        "youngs_modulus": np.random.uniform(300, 500, n_samples),
        "thermal_conductivity_1000C": np.random.uniform(30, 60, n_samples),
    }
    
    # Synthetic V50 values
    y = (
        X["hardness"] * 10
        + X["fracture_toughness"] * 20
        + X["density"] * 5
        + np.random.normal(0, 10, n_samples)
    )
    
    return X, y


@pytest.fixture
def sample_screening_materials():
    """Provide sample materials for screening tests."""
    return [
        {
            "material_id": "test-sic-pure",
            "formula": "SiC",
            "energy_above_hull": 0.0,
            "formation_energy_per_atom": -0.65,
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
        },
        {
            "material_id": "test-sic-b-doped",
            "formula": "Si0.98B0.02C",
            "energy_above_hull": 0.02,
            "formation_energy_per_atom": -0.63,
            "hardness": 29.5,
            "fracture_toughness": 4.8,
            "density": 3.19,
        },
        {
            "material_id": "test-sic-unstable",
            "formula": "Si0.90Fe0.10C",
            "energy_above_hull": 0.25,
            "formation_energy_per_atom": -0.40,
            "hardness": 25.0,
            "fracture_toughness": 3.5,
            "density": 3.50,
        },
    ]
