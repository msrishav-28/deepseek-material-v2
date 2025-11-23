"""Tests for database system and unit conversion."""

import pytest
from ceramic_discovery.dft import UnitConverter


class TestUnitConverter:
    """Test unit conversion and standardization."""

    def test_converter_initialization(self):
        """Test converter initializes correctly."""
        converter = UnitConverter()
        assert len(converter.STANDARD_UNITS) > 0
        assert len(converter.CONVERSIONS) > 0

    def test_fracture_toughness_standard_unit(self):
        """CRITICAL: Test that fracture toughness uses correct standard unit."""
        converter = UnitConverter()
        standard_unit = converter.get_standard_unit("fracture_toughness")
        assert standard_unit == "MPa·m^0.5"

    def test_convert_same_unit(self):
        """Test conversion with same source and target unit."""
        converter = UnitConverter()
        result = converter.convert(10.0, "GPa", "GPa")
        assert result == 10.0

    def test_convert_pressure_mpa_to_gpa(self):
        """Test pressure conversion from MPa to GPa."""
        converter = UnitConverter()
        result = converter.convert(1000.0, "MPa", "GPa")
        assert result == pytest.approx(1.0)

    def test_convert_pressure_pa_to_gpa(self):
        """Test pressure conversion from Pa to GPa."""
        converter = UnitConverter()
        result = converter.convert(1e9, "Pa", "GPa")
        assert result == pytest.approx(1.0)

    def test_convert_density_kg_m3_to_g_cm3(self):
        """Test density conversion."""
        converter = UnitConverter()
        result = converter.convert(3210.0, "kg/m³", "g/cm³")
        assert result == pytest.approx(3.21)

    def test_convert_length_nm_to_angstrom(self):
        """Test length conversion."""
        converter = UnitConverter()
        result = converter.convert(1.0, "nm", "Å")
        assert result == pytest.approx(10.0)

    def test_convert_temperature_celsius_to_kelvin(self):
        """Test temperature conversion with offset."""
        converter = UnitConverter()
        result = converter.convert(0.0, "°C", "K")
        assert result == pytest.approx(273.15)

        result = converter.convert(100.0, "°C", "K")
        assert result == pytest.approx(373.15)

    def test_convert_reverse_temperature(self):
        """Test reverse temperature conversion."""
        converter = UnitConverter()
        result = converter.convert(273.15, "K", "°C")
        assert result == pytest.approx(0.0)

    def test_convert_thermal_conductivity(self):
        """Test thermal conductivity conversion."""
        converter = UnitConverter()
        result = converter.convert(1.0, "W/cm·K", "W/m·K")
        assert result == pytest.approx(100.0)

    def test_convert_velocity(self):
        """Test velocity conversion."""
        converter = UnitConverter()
        result = converter.convert(1.0, "km/s", "m/s")
        assert result == pytest.approx(1000.0)

    def test_convert_unsupported_raises_error(self):
        """Test that unsupported conversion raises error."""
        converter = UnitConverter()
        with pytest.raises(ValueError, match="not supported"):
            converter.convert(10.0, "unsupported_unit", "GPa")

    def test_standardize_hardness(self):
        """Test standardization of hardness value."""
        converter = UnitConverter()
        value, unit = converter.standardize(28000.0, "MPa", "hardness")
        assert value == pytest.approx(28.0)
        assert unit == "GPa"

    def test_standardize_fracture_toughness(self):
        """Test standardization of fracture toughness."""
        converter = UnitConverter()
        value, unit = converter.standardize(4.5, "MPa·m^0.5", "fracture_toughness")
        assert value == pytest.approx(4.5)
        assert unit == "MPa·m^0.5"

    def test_standardize_density(self):
        """Test standardization of density."""
        converter = UnitConverter()
        value, unit = converter.standardize(3210.0, "kg/m³", "density")
        assert value == pytest.approx(3.21)
        assert unit == "g/cm³"

    def test_standardize_unknown_property_raises_error(self):
        """Test that unknown property raises error."""
        converter = UnitConverter()
        with pytest.raises(ValueError, match="Unknown property"):
            converter.standardize(10.0, "GPa", "unknown_property")

    def test_normalize_unit_variations(self):
        """Test normalization of unit string variations."""
        converter = UnitConverter()

        # Test various fracture toughness notations
        assert converter._normalize_unit("MPa·m^0.5") == "MPa·m^0.5"
        assert converter._normalize_unit("mpa*m^0.5") == "MPa·m^0.5"
        assert converter._normalize_unit("MPa√m") == "MPa·m^0.5"

        # Test pressure variations
        assert converter._normalize_unit("gpa") == "GPa"
        assert converter._normalize_unit("GPA") == "GPa"

        # Test density variations
        assert converter._normalize_unit("g/cc") == "g/cm³"
        assert converter._normalize_unit("g/cm3") == "g/cm³"

    def test_validate_unit_correct(self):
        """Test validation of correct units."""
        converter = UnitConverter()

        assert converter.validate_unit("GPa", "hardness") is True
        assert converter.validate_unit("MPa·m^0.5", "fracture_toughness") is True
        assert converter.validate_unit("g/cm³", "density") is True

    def test_validate_unit_convertible(self):
        """Test validation of convertible units."""
        converter = UnitConverter()

        assert converter.validate_unit("MPa", "hardness") is True  # Can convert to GPa
        assert converter.validate_unit("kg/m³", "density") is True  # Can convert to g/cm³

    def test_validate_unit_incorrect(self):
        """Test validation rejects incorrect units."""
        converter = UnitConverter()

        assert converter.validate_unit("invalid_unit", "hardness") is False
        assert converter.validate_unit("GPa", "unknown_property") is False

    def test_batch_standardize(self):
        """Test batch standardization of multiple properties."""
        converter = UnitConverter()

        properties = {
            "hardness": (28000.0, "MPa"),
            "fracture_toughness": (4.5, "MPa·m^0.5"),
            "density": (3210.0, "kg/m³"),
            "youngs_modulus": (410.0, "GPa"),
        }

        standardized = converter.batch_standardize(properties)

        assert standardized["hardness"][0] == pytest.approx(28.0)
        assert standardized["hardness"][1] == "GPa"

        assert standardized["fracture_toughness"][0] == pytest.approx(4.5)
        assert standardized["fracture_toughness"][1] == "MPa·m^0.5"

        assert standardized["density"][0] == pytest.approx(3.21)
        assert standardized["density"][1] == "g/cm³"

        assert standardized["youngs_modulus"][0] == pytest.approx(410.0)
        assert standardized["youngs_modulus"][1] == "GPa"

    def test_get_standard_unit(self):
        """Test retrieval of standard units."""
        converter = UnitConverter()

        assert converter.get_standard_unit("hardness") == "GPa"
        assert converter.get_standard_unit("fracture_toughness") == "MPa·m^0.5"
        assert converter.get_standard_unit("density") == "g/cm³"
        assert converter.get_standard_unit("band_gap") == "eV"
        assert converter.get_standard_unit("unknown") is None

    def test_ceramic_properties_standardization(self):
        """Test standardization of typical ceramic properties."""
        converter = UnitConverter()

        # SiC properties
        sic_properties = {
            "hardness": (28.0, "GPa"),
            "fracture_toughness": (4.6, "MPa·m^0.5"),
            "density": (3.21, "g/cm³"),
            "youngs_modulus": (410.0, "GPa"),
            "thermal_conductivity": (120.0, "W/m·K"),
        }

        standardized = converter.batch_standardize(sic_properties)

        # All should be in standard units already
        for prop_name, (value, unit) in standardized.items():
            standard_unit = converter.get_standard_unit(prop_name)
            assert unit == standard_unit

    def test_reverse_conversions_work(self):
        """Test that reverse conversions are properly generated."""
        converter = UnitConverter()

        # Test forward and reverse
        forward = converter.convert(1000.0, "MPa", "GPa")
        reverse = converter.convert(forward, "GPa", "MPa")
        assert reverse == pytest.approx(1000.0)

        # Test with density
        forward = converter.convert(3210.0, "kg/m³", "g/cm³")
        reverse = converter.convert(forward, "g/cm³", "kg/m³")
        assert reverse == pytest.approx(3210.0)


class TestDatabaseSchema:
    """Test database schema definitions."""

    def test_material_model_exists(self):
        """Test that Material model is defined."""
        from ceramic_discovery.dft import Material

        assert Material is not None
        assert hasattr(Material, "__tablename__")
        assert Material.__tablename__ == "materials"

    def test_material_property_model_exists(self):
        """Test that MaterialProperty model is defined."""
        from ceramic_discovery.dft import MaterialProperty

        assert MaterialProperty is not None
        assert hasattr(MaterialProperty, "__tablename__")
        assert MaterialProperty.__tablename__ == "material_properties"

    def test_dft_calculation_model_exists(self):
        """Test that DFTCalculation model is defined."""
        from ceramic_discovery.dft import DFTCalculation

        assert DFTCalculation is not None
        assert hasattr(DFTCalculation, "__tablename__")
        assert DFTCalculation.__tablename__ == "dft_calculations"

    def test_experimental_data_model_exists(self):
        """Test that ExperimentalData model is defined."""
        from ceramic_discovery.dft import ExperimentalData

        assert ExperimentalData is not None
        assert hasattr(ExperimentalData, "__tablename__")
        assert ExperimentalData.__tablename__ == "experimental_data"

    def test_screening_result_model_exists(self):
        """Test that ScreeningResult model is defined."""
        from ceramic_discovery.dft import ScreeningResult

        assert ScreeningResult is not None
        assert hasattr(ScreeningResult, "__tablename__")
        assert ScreeningResult.__tablename__ == "screening_results"


# Note: Full database integration tests would require a test database
# These are minimal tests to verify the schema and converter work correctly
