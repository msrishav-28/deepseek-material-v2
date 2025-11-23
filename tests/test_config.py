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
