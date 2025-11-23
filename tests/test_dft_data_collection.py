"""Tests for DFT data collection and validation."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests

from ceramic_discovery.dft import (
    MaterialsProjectClient,
    MaterialData,
    RateLimiter,
    DataValidator,
    ValidationResult,
    PropertyExtractor,
    PropertyValue,
)


class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(rate_limit_per_second=5)
        assert limiter.rate_limit == 5
        assert limiter.min_interval == 0.2

    def test_rate_limiter_enforces_delay(self):
        """Test that rate limiter enforces minimum delay between requests."""
        limiter = RateLimiter(rate_limit_per_second=10)  # 0.1s minimum interval

        start_time = time.time()
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Second call should have waited ~0.1s
        assert elapsed >= 0.09  # Allow small tolerance


class TestMaterialsProjectClient:
    """Test Materials Project API client."""

    def test_client_initialization_requires_api_key(self):
        """Test that client requires API key."""
        with pytest.raises(ValueError, match="API key is required"):
            MaterialsProjectClient(api_key="")

    def test_client_initialization_with_valid_key(self):
        """Test client initializes with valid API key."""
        client = MaterialsProjectClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"
        assert client.timeout == 30
        assert client.rate_limiter.rate_limit == 5

    def test_client_custom_configuration(self):
        """Test client accepts custom configuration."""
        client = MaterialsProjectClient(
            api_key="test_key",
            rate_limit_per_second=10,
            timeout_seconds=60,
            max_retries=5,
        )
        assert client.rate_limiter.rate_limit == 10
        assert client.timeout == 60

    @patch("requests.Session.get")
    def test_make_request_with_rate_limiting(self, mock_get):
        """Test that requests are rate limited."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key", rate_limit_per_second=10)

        start_time = time.time()
        client._make_request("test_endpoint")
        client._make_request("test_endpoint")
        elapsed = time.time() - start_time

        # Should have rate limiting delay
        assert elapsed >= 0.09

    @patch("requests.Session.get")
    def test_make_request_handles_timeout(self, mock_get):
        """Test that timeout errors are handled properly."""
        mock_get.side_effect = requests.exceptions.Timeout()

        client = MaterialsProjectClient(api_key="test_key")

        with pytest.raises(requests.exceptions.RequestException, match="timed out"):
            client._make_request("test_endpoint")

    @patch("requests.Session.get")
    def test_make_request_handles_401_unauthorized(self, mock_get):
        """Test handling of invalid API key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="invalid_key")

        with pytest.raises(requests.exceptions.RequestException, match="Invalid API key"):
            client._make_request("test_endpoint")

    @patch("requests.Session.get")
    def test_make_request_handles_429_rate_limit(self, mock_get):
        """Test handling of rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key")

        with pytest.raises(requests.exceptions.RequestException, match="Rate limit exceeded"):
            client._make_request("test_endpoint")

    @patch("requests.Session.get")
    def test_get_material_by_id_success(self, mock_get):
        """Test successful material retrieval by ID."""
        mock_data = {
            "data": [
                {
                    "material_id": "mp-149",
                    "formula_pretty": "SiC",
                    "structure": {"lattice": {"a": 3.08}, "nsites": 2},
                    "formation_energy_per_atom": -0.65,
                    "energy_above_hull": 0.0,
                    "band_gap": 2.3,
                    "density": 3.21,
                }
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key")
        material = client.get_material_by_id("mp-149")

        assert isinstance(material, MaterialData)
        assert material.material_id == "mp-149"
        assert material.formula == "SiC"
        assert material.formation_energy == -0.65
        assert material.energy_above_hull == 0.0
        assert material.band_gap == 2.3
        assert material.density == 3.21

    @patch("requests.Session.get")
    def test_get_material_by_id_not_found(self, mock_get):
        """Test handling of material not found."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key")

        with pytest.raises(ValueError, match="No data found"):
            client.get_material_by_id("mp-invalid")

    @patch("requests.Session.get")
    def test_search_materials_with_formula(self, mock_get):
        """Test material search by formula."""
        mock_data = {
            "data": [
                {
                    "material_id": "mp-149",
                    "formula_pretty": "SiC",
                    "formation_energy_per_atom": -0.65,
                    "energy_above_hull": 0.0,
                },
                {
                    "material_id": "mp-150",
                    "formula_pretty": "SiC",
                    "formation_energy_per_atom": -0.60,
                    "energy_above_hull": 0.05,
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key")
        materials = client.search_materials(formula="SiC")

        assert len(materials) == 2
        assert all(isinstance(m, MaterialData) for m in materials)
        assert materials[0].formula == "SiC"
        assert materials[1].formula == "SiC"

    @patch("requests.Session.get")
    def test_search_materials_with_energy_filter(self, mock_get):
        """Test material search with energy above hull filter."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MaterialsProjectClient(api_key="test_key")
        client.search_materials(formula="SiC", max_energy_above_hull=0.1)

        # Verify the correct parameter was passed
        call_args = mock_get.call_args
        assert call_args[1]["params"]["energy_above_hull"] == "<0.1"

    def test_context_manager(self):
        """Test client works as context manager."""
        with MaterialsProjectClient(api_key="test_key") as client:
            assert client.api_key == "test_key"


class TestDataValidator:
    """Test data validation functionality."""

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = DataValidator(outlier_threshold_sigma=3.0)
        assert validator.outlier_threshold == 3.0

    def test_validate_material_with_valid_data(self):
        """Test validation of valid material data."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-149",
            "formula": "SiC",
            "formation_energy_per_atom": -0.65,
            "energy_above_hull": 0.0,
            "density": 3.21,
            "band_gap": 2.3,
        }

        result = validator.validate_material(material_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.quality_score > 0.9

    def test_validate_material_missing_required_properties(self):
        """Test validation fails for missing required properties."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-149",
            # Missing formula, formation_energy, energy_above_hull
        }

        result = validator.validate_material(material_data)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Missing required properties" in result.errors[0]
        assert result.quality_score < 0.8

    def test_validate_material_out_of_bounds_density(self):
        """Test validation catches out-of-bounds density."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-test",
            "formula": "Test",
            "formation_energy_per_atom": -1.0,
            "energy_above_hull": 0.0,
            "density": 50.0,  # Unrealistically high
        }

        result = validator.validate_material(material_data)

        assert result.is_valid is False
        assert any("density" in error.lower() for error in result.errors)

    def test_validate_material_negative_energy_above_hull(self):
        """Test validation catches negative energy above hull."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-test",
            "formula": "Test",
            "formation_energy_per_atom": -1.0,
            "energy_above_hull": -0.1,  # Should be >= 0
        }

        result = validator.validate_material(material_data)

        assert result.is_valid is False
        assert any("energy_above_hull" in error.lower() for error in result.errors)

    def test_physical_consistency_metal_with_band_gap(self):
        """Test detection of inconsistent metal/band gap."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-test",
            "formula": "Test",
            "formation_energy_per_atom": -1.0,
            "energy_above_hull": 0.0,
            "is_metal": True,
            "band_gap": 2.0,  # Metals should have zero band gap
        }

        result = validator.validate_material(material_data)

        assert len(result.warnings) > 0
        assert any("metal" in warning.lower() for warning in result.warnings)

    def test_physical_consistency_shear_exceeds_bulk(self):
        """Test detection of shear modulus exceeding bulk modulus."""
        validator = DataValidator()

        material_data = {
            "material_id": "mp-test",
            "formula": "Test",
            "formation_energy_per_atom": -1.0,
            "energy_above_hull": 0.0,
            "bulk_modulus": 100.0,
            "shear_modulus": 150.0,  # Should not exceed bulk modulus
        }

        result = validator.validate_material(material_data)

        assert result.is_valid is False
        assert any("shear" in error.lower() and "bulk" in error.lower() for error in result.errors)

    def test_detect_outliers(self):
        """Test outlier detection across multiple materials."""
        validator = DataValidator(outlier_threshold_sigma=2.0)

        materials = [
            {"density": 3.0},
            {"density": 3.1},
            {"density": 3.2},
            {"density": 3.0},
            {"density": 3.1},
            {"density": 3.0},
            {"density": 3.2},
            {"density": 20.0},  # Clear outlier - much further from mean
        ]

        outlier_indices = validator.detect_outliers(materials, "density")

        assert len(outlier_indices) >= 1
        assert 7 in outlier_indices

    def test_detect_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        validator = DataValidator()

        materials = [{"density": 3.0}, {"density": 3.1}]

        outlier_indices = validator.detect_outliers(materials, "density")

        assert len(outlier_indices) == 0

    def test_validate_batch(self):
        """Test batch validation of multiple materials."""
        validator = DataValidator()

        materials = [
            {
                "material_id": "mp-1",
                "formula": "SiC",
                "formation_energy_per_atom": -0.65,
                "energy_above_hull": 0.0,
            },
            {
                "material_id": "mp-2",
                "formula": "B4C",
                "formation_energy_per_atom": -0.50,
                "energy_above_hull": 0.05,
            },
            {
                "material_id": "mp-3",
                # Missing required properties
            },
        ]

        summary = validator.validate_batch(materials)

        assert summary["total_materials"] == 3
        assert summary["valid_materials"] == 2
        assert summary["invalid_materials"] == 1
        assert 0.0 <= summary["average_quality_score"] <= 1.0


class TestPropertyExtractor:
    """Test property extraction functionality."""

    def test_extractor_initialization(self):
        """Test property extractor initializes correctly."""
        extractor = PropertyExtractor()

        assert len(extractor.PROPERTY_DEFINITIONS) == 58
        assert len(extractor.tier_1_properties) > 0
        assert len(extractor.tier_2_properties) > 0
        assert len(extractor.tier_3_properties) > 0

    def test_extract_structural_properties(self):
        """Test extraction of structural properties."""
        extractor = PropertyExtractor()

        material_data = {
            "crystal_structure": {
                "lattice": {
                    "a": 3.08,
                    "b": 3.08,
                    "c": 3.08,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                    "volume": 29.2,
                },
                "space_group": {"number": 216},
                "nsites": 2,
            },
            "density": 3.21,
        }

        properties = extractor._extract_structural_properties(material_data)

        assert "lattice_a" in properties
        assert properties["lattice_a"].value == 3.08
        assert properties["lattice_a"].unit == "Å"
        assert "density" in properties
        assert properties["density"].value == 3.21

    def test_extract_thermodynamic_properties(self):
        """Test extraction of thermodynamic properties."""
        extractor = PropertyExtractor()

        material_data = {
            "formation_energy": -0.65,
            "energy_above_hull": 0.0,
            "is_stable": True,
        }

        properties = extractor._extract_thermodynamic_properties(material_data)

        assert "formation_energy_per_atom" in properties
        assert properties["formation_energy_per_atom"].value == -0.65
        assert "energy_above_hull" in properties
        assert properties["energy_above_hull"].value == 0.0

    def test_extract_electronic_properties(self):
        """Test extraction of electronic properties."""
        extractor = PropertyExtractor()

        material_data = {
            "band_gap": 2.3,
            "is_metal": False,
            "is_magnetic": False,
        }

        properties = extractor._extract_electronic_properties(material_data)

        assert "band_gap" in properties
        assert properties["band_gap"].value == 2.3
        assert properties["band_gap"].unit == "eV"

    def test_extract_mechanical_properties_with_calculation(self):
        """Test extraction and calculation of mechanical properties."""
        extractor = PropertyExtractor()

        material_data = {
            "bulk_modulus": 220.0,
            "shear_modulus": 190.0,
        }

        properties = extractor._extract_mechanical_properties(material_data)

        assert "bulk_modulus_vrh" in properties
        assert "shear_modulus_vrh" in properties
        assert "youngs_modulus" in properties
        assert "poisson_ratio" in properties

        # Verify calculated values are reasonable
        assert properties["youngs_modulus"].value > 0
        assert 0 < properties["poisson_ratio"].value < 0.5

    def test_extract_all_properties(self):
        """Test extraction of all 58 properties."""
        extractor = PropertyExtractor()

        material_data = {
            "crystal_structure": {
                "lattice": {"a": 3.08, "b": 3.08, "c": 3.08},
                "nsites": 2,
            },
            "density": 3.21,
            "formation_energy": -0.65,
            "energy_above_hull": 0.0,
            "band_gap": 2.3,
            "bulk_modulus": 220.0,
            "shear_modulus": 190.0,
        }

        all_properties = extractor.extract_all_properties(material_data)

        # Should have all 58 properties
        assert len(all_properties) == 58

        # All should be PropertyValue objects
        assert all(isinstance(prop, PropertyValue) for prop in all_properties.values())

        # Some should have values, others should be None
        available = extractor.filter_available_properties(all_properties)
        assert len(available) > 0
        assert len(available) < 58

    def test_get_tier_1_properties(self):
        """Test filtering to Tier 1 properties only."""
        extractor = PropertyExtractor()

        material_data = {
            "density": 3.21,
            "formation_energy": -0.65,
            "energy_above_hull": 0.0,
            "band_gap": 2.3,
            "bulk_modulus": 220.0,  # Tier 2
        }

        all_properties = extractor.extract_all_properties(material_data)
        tier_1 = extractor.get_tier_1_properties(all_properties)

        # Should only contain Tier 1 properties
        assert "density" in tier_1
        assert "formation_energy_per_atom" in tier_1
        assert "band_gap" in tier_1
        assert "bulk_modulus_vrh" not in tier_1  # Tier 2

    def test_filter_available_properties(self):
        """Test filtering to only available properties."""
        extractor = PropertyExtractor()

        material_data = {
            "density": 3.21,
            "formation_energy": -0.65,
        }

        all_properties = extractor.extract_all_properties(material_data)
        available = extractor.filter_available_properties(all_properties)

        # Should only contain properties with values
        assert all(prop.value is not None for prop in available.values())
        assert len(available) < len(all_properties)
