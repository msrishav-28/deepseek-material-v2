"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
from ceramic_discovery.config import Config, DatabaseConfig, MLConfig


def test_config_initialization():
    """Test that configuration initializes with defaults."""
    config = Config()
    
    assert config.database is not None
    assert config.ml is not None
    assert config.hdf5 is not None


def test_database_config_defaults():
    """Test database configuration defaults."""
    db_config = DatabaseConfig()
    
    assert db_config.pool_size == 10
    assert db_config.max_overflow == 20
    assert "postgresql" in db_config.url


def test_ml_config_targets():
    """Test ML configuration has correct performance targets."""
    ml_config = MLConfig()
    
    # Realistic R² targets as per requirements
    assert ml_config.target_r2_min == 0.65
    assert ml_config.target_r2_max == 0.75
    assert ml_config.bootstrap_samples == 1000


def test_config_creates_directories():
    """Test that configuration creates necessary directories."""
    config = Config()
    
    assert config.hdf5.data_path.exists()
    assert config.logging.file.parent.exists()


def test_config_validation_missing_api_key(monkeypatch):
    """Test configuration validation fails without API key."""
    monkeypatch.setenv("MATERIALS_PROJECT_API_KEY", "")
    
    config = Config()
    
    with pytest.raises(ValueError, match="MATERIALS_PROJECT_API_KEY is required"):
        config.validate()


def test_config_validation_success(monkeypatch):
    """Test configuration validation succeeds with required values."""
    monkeypatch.setenv("MATERIALS_PROJECT_API_KEY", "test_key_123")
    
    config = Config()
    
    # Should not raise
    config.validate()


def test_data_sources_config_defaults():
    """Test data sources configuration defaults."""
    config = Config()
    
    # Check JARVIS config
    assert config.data_sources.jarvis.enabled is True
    assert config.data_sources.jarvis.file_path == Path("./data/jdft_3d-7-7-2018.json")
    assert "Si" in config.data_sources.jarvis.metal_elements
    assert "Ti" in config.data_sources.jarvis.metal_elements
    
    # Check NIST config
    assert config.data_sources.nist.enabled is True
    assert config.data_sources.nist.data_dir == Path("./data/nist")
    assert config.data_sources.nist.default_temperature_K == 298.15
    
    # Check literature config
    assert config.data_sources.literature.enabled is True
    
    # Check data combination settings
    assert config.data_sources.enable_deduplication is True
    assert config.data_sources.conflict_resolution_strategy == "priority"
    assert "literature" in config.data_sources.source_priority


def test_feature_engineering_config_defaults():
    """Test feature engineering configuration defaults."""
    config = Config()
    
    # Check composition descriptors
    assert config.feature_engineering.composition_descriptors.enabled is True
    assert config.feature_engineering.composition_descriptors.calculate_entropy is True
    assert config.feature_engineering.composition_descriptors.calculate_electronegativity is True
    assert config.feature_engineering.composition_descriptors.calculate_atomic_properties is True
    
    # Check structure descriptors
    assert config.feature_engineering.structure_descriptors.enabled is True
    assert config.feature_engineering.structure_descriptors.calculate_pugh_ratio is True
    assert config.feature_engineering.structure_descriptors.calculate_elastic_properties is True
    assert config.feature_engineering.structure_descriptors.calculate_energy_density is True
    assert config.feature_engineering.structure_descriptors.validate_physical_bounds is True
    
    # Check feature selection settings
    assert config.feature_engineering.min_feature_variance == 0.01
    assert config.feature_engineering.remove_correlated_features is False
    assert config.feature_engineering.correlation_threshold == 0.95


def test_application_ranking_config_defaults():
    """Test application ranking configuration defaults."""
    config = Config()
    
    assert config.application_ranking.enabled is True
    assert len(config.application_ranking.applications) == 5
    assert "aerospace_hypersonic" in config.application_ranking.applications
    assert "cutting_tools" in config.application_ranking.applications
    assert "thermal_barriers" in config.application_ranking.applications
    assert "wear_resistant" in config.application_ranking.applications
    assert "electronic" in config.application_ranking.applications
    
    assert config.application_ranking.allow_partial_scoring is True
    assert config.application_ranking.min_properties_for_ranking == 2
    assert config.application_ranking.confidence_threshold == 0.5


def test_experimental_planning_config_defaults():
    """Test experimental planning configuration defaults."""
    config = Config()
    
    assert config.experimental_planning.enabled is True
    
    # Check synthesis config
    assert len(config.experimental_planning.synthesis.default_methods) == 3
    assert "Hot Pressing" in config.experimental_planning.synthesis.default_methods
    assert "Solid State Sintering" in config.experimental_planning.synthesis.default_methods
    assert "Spark Plasma Sintering" in config.experimental_planning.synthesis.default_methods
    assert config.experimental_planning.synthesis.cost_multiplier == 1.0
    assert config.experimental_planning.synthesis.time_multiplier == 1.0
    
    # Check characterization config
    assert len(config.experimental_planning.characterization.default_techniques) == 6
    assert "XRD" in config.experimental_planning.characterization.default_techniques
    assert "SEM" in config.experimental_planning.characterization.default_techniques
    assert config.experimental_planning.characterization.cost_multiplier == 1.0
    assert config.experimental_planning.characterization.time_multiplier == 1.0
    
    # Check resource estimation settings
    assert config.experimental_planning.default_timeline_months == 6.0
    assert config.experimental_planning.default_cost_k_dollars == 50.0
    assert config.experimental_planning.confidence_level == 0.8


def test_config_validation_invalid_correlation_threshold(monkeypatch):
    """Test configuration validation fails with invalid correlation threshold."""
    monkeypatch.setenv("MATERIALS_PROJECT_API_KEY", "test_key_123")
    
    config = Config()
    config.feature_engineering.correlation_threshold = 1.5
    
    with pytest.raises(ValueError, match="Correlation threshold must be between 0 and 1"):
        config.validate()


def test_config_validation_empty_applications(monkeypatch):
    """Test configuration validation fails with no applications."""
    monkeypatch.setenv("MATERIALS_PROJECT_API_KEY", "test_key_123")
    
    config = Config()
    config.application_ranking.applications = []
    
    with pytest.raises(ValueError, match="At least one application must be specified"):
        config.validate()


def test_config_environment_variables(monkeypatch):
    """Test configuration respects environment variables."""
    monkeypatch.setenv("JARVIS_ENABLED", "false")
    monkeypatch.setenv("NIST_ENABLED", "false")
    monkeypatch.setenv("JARVIS_FILE_PATH", "/custom/path/jarvis.json")
    monkeypatch.setenv("NIST_DATA_DIR", "/custom/path/nist")
    
    config = Config()
    
    assert config.data_sources.jarvis.enabled is False
    assert config.data_sources.nist.enabled is False
    assert config.data_sources.jarvis.file_path == Path("/custom/path/jarvis.json")
    assert config.data_sources.nist.data_dir == Path("/custom/path/nist")
