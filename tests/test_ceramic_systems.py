"""Tests for ceramic systems management."""

import pytest
import warnings

from ceramic_discovery.ceramics import (
    CeramicType,
    CeramicSystem,
    CeramicSystemFactory,
    ValueWithUncertainty,
    MechanicalProperties,
    ThermalProperties,
    StructuralProperties,
    CompositeSystem,
    CompositeCalculator,
)


class TestValueWithUncertainty:
    """Test ValueWithUncertainty dataclass."""
    
    def test_valid_value_creation(self):
        """Test creating valid value with uncertainty."""
        value = ValueWithUncertainty(
            value=28.0,
            uncertainty=2.0,
            unit="GPa",
            source="literature"
        )
        
        assert value.value == 28.0
        assert value.uncertainty == 2.0
        assert value.unit == "GPa"
        assert value.source == "literature"
    
    def test_negative_uncertainty_raises_error(self):
        """Test that negative uncertainty raises error."""
        with pytest.raises(ValueError, match="Uncertainty must be non-negative"):
            ValueWithUncertainty(
                value=28.0,
                uncertainty=-1.0,
                unit="GPa"
            )


class TestCeramicSystemFactory:
    """Test ceramic system factory."""
    
    def test_create_sic(self):
        """Test creating SiC system."""
        sic = CeramicSystemFactory.create_sic()
        
        assert sic.ceramic_type == CeramicType.SIC
        assert sic.mechanical.hardness.value == 28.0
        assert sic.mechanical.fracture_toughness.unit == "MPa·m^0.5"
        assert sic.structural.density.value == 3.21
        assert sic.thermal.thermal_conductivity_25C.value == 120.0
        assert not sic.is_doped()
    
    def test_create_b4c(self):
        """Test creating B₄C system."""
        b4c = CeramicSystemFactory.create_b4c()
        
        assert b4c.ceramic_type == CeramicType.B4C
        assert b4c.mechanical.hardness.value == 30.0
        assert b4c.mechanical.fracture_toughness.unit == "MPa·m^0.5"
        assert b4c.structural.density.value == 2.52
    
    def test_create_wc(self):
        """Test creating WC system."""
        wc = CeramicSystemFactory.create_wc()
        
        assert wc.ceramic_type == CeramicType.WC
        assert wc.mechanical.hardness.value == 22.0
        assert wc.structural.density.value == 15.63
    
    def test_create_tic(self):
        """Test creating TiC system."""
        tic = CeramicSystemFactory.create_tic()
        
        assert tic.ceramic_type == CeramicType.TIC
        assert tic.mechanical.hardness.value == 28.0
        assert tic.structural.density.value == 4.93
    
    def test_create_al2o3(self):
        """Test creating Al₂O₃ system."""
        al2o3 = CeramicSystemFactory.create_al2o3()
        
        assert al2o3.ceramic_type == CeramicType.AL2O3
        assert al2o3.mechanical.hardness.value == 20.0
        assert al2o3.structural.density.value == 3.98
    
    def test_create_system_by_type(self):
        """Test creating system by ceramic type."""
        sic = CeramicSystemFactory.create_system(CeramicType.SIC)
        assert sic.ceramic_type == CeramicType.SIC
        
        b4c = CeramicSystemFactory.create_system(CeramicType.B4C)
        assert b4c.ceramic_type == CeramicType.B4C


class TestCeramicSystem:
    """Test ceramic system functionality."""
    
    def test_composition_string_undoped(self):
        """Test composition string for undoped system."""
        sic = CeramicSystemFactory.create_sic()
        assert sic.get_composition_string() == "SiC"
    
    def test_doped_system_creation(self):
        """Test creating doped system."""
        sic = CeramicSystemFactory.create_sic()
        
        doped_sic = CeramicSystemFactory.create_doped_system(
            base_ceramic=sic,
            dopant_element="B",
            dopant_concentration=0.02
        )
        
        assert doped_sic.is_doped()
        assert doped_sic.dopant_element == "B"
        assert doped_sic.dopant_concentration == 0.02
        assert "B(2.0%)" in doped_sic.get_composition_string()
    
    def test_doped_system_with_property_modifications(self):
        """Test doped system with property modifications."""
        sic = CeramicSystemFactory.create_sic()
        base_hardness = sic.mechanical.hardness.value
        
        # Dopant increases hardness by 10 GPa per unit concentration
        doped_sic = CeramicSystemFactory.create_doped_system(
            base_ceramic=sic,
            dopant_element="B",
            dopant_concentration=0.02,
            property_modifications={"hardness": 10.0}
        )
        
        expected_hardness = base_hardness + 10.0 * 0.02
        assert abs(doped_sic.mechanical.hardness.value - expected_hardness) < 0.01
        assert doped_sic.mechanical.hardness.source == "calculated_doped"
    
    def test_supported_concentrations(self):
        """Test supported dopant concentrations."""
        expected = [0.01, 0.02, 0.03, 0.05, 0.10]
        assert CeramicSystem.SUPPORTED_CONCENTRATIONS == expected
    
    def test_unsupported_concentration_warning(self):
        """Test warning for unsupported concentration."""
        sic = CeramicSystemFactory.create_sic()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            doped_sic = CeramicSystemFactory.create_doped_system(
                base_ceramic=sic,
                dopant_element="B",
                dopant_concentration=0.04  # Not in standard set
            )
            
            assert len(w) == 1
            assert "not in standard set" in str(w[0].message)
    
    def test_invalid_concentration_raises_error(self):
        """Test that invalid concentration raises error."""
        sic = CeramicSystemFactory.create_sic()
        
        with pytest.raises(ValueError, match="out of valid range"):
            CeramicSystemFactory.create_doped_system(
                base_ceramic=sic,
                dopant_element="B",
                dopant_concentration=0.15  # Too high
            )


class TestPropertyValidation:
    """Test property bounds validation."""
    
    def test_mechanical_properties_validation(self):
        """Test mechanical properties bounds validation."""
        sic = CeramicSystemFactory.create_sic()
        violations = sic.mechanical.validate_bounds()
        
        # Should have no violations for baseline SiC
        assert len(violations) == 0
    
    def test_thermal_properties_validation(self):
        """Test thermal properties bounds validation."""
        sic = CeramicSystemFactory.create_sic()
        violations = sic.thermal.validate_bounds()
        
        # Should have no violations for baseline SiC
        assert len(violations) == 0
    
    def test_structural_properties_validation(self):
        """Test structural properties bounds validation."""
        sic = CeramicSystemFactory.create_sic()
        violations = sic.structural.validate_bounds()
        
        # Should have no violations for baseline SiC
        assert len(violations) == 0
    
    def test_validate_all_properties(self):
        """Test validating all properties at once."""
        sic = CeramicSystemFactory.create_sic()
        all_violations = sic.validate_all_properties()
        
        # Should have no violations for baseline SiC
        assert len(all_violations) == 0
    
    def test_hardness_out_of_bounds(self):
        """Test detection of hardness out of bounds."""
        sic = CeramicSystemFactory.create_sic()
        sic.mechanical.hardness.value = 60.0  # Too high
        
        violations = sic.mechanical.validate_bounds()
        assert len(violations) > 0
        assert "Hardness" in violations[0]
    
    def test_fracture_toughness_unit_correct(self):
        """Test that fracture toughness uses correct unit."""
        sic = CeramicSystemFactory.create_sic()
        
        # CRITICAL: Must use MPa·m^0.5 (not MPa·m^1/2 or other variants)
        assert sic.mechanical.fracture_toughness.unit == "MPa·m^0.5"


class TestCompositeCalculator:
    """Test composite materials calculator."""
    
    def test_composite_system_creation(self):
        """Test creating composite system."""
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()
        
        composite = CompositeSystem(
            component1=sic,
            component2=b4c,
            volume_fraction1=0.7,
            volume_fraction2=0.3
        )
        
        assert composite.volume_fraction1 == 0.7
        assert composite.volume_fraction2 == 0.3
        assert "SiC(70%)-B₄C(30%)" in composite.get_composition_string()
    
    def test_volume_fractions_must_sum_to_one(self):
        """Test that volume fractions must sum to 1."""
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            CompositeSystem(
                component1=sic,
                component2=b4c,
                volume_fraction1=0.6,
                volume_fraction2=0.5  # Sum = 1.1
            )
    
    def test_rule_of_mixtures_linear(self):
        """Test linear rule of mixtures."""
        calculator = CompositeCalculator()
        
        result = calculator.calculate_rule_of_mixtures(
            value1=28.0,
            value2=30.0,
            fraction1=0.7,
            fraction2=0.3
        )
        
        expected = 28.0 * 0.7 + 30.0 * 0.3
        assert abs(result - expected) < 0.01
    
    def test_inverse_rule_of_mixtures(self):
        """Test inverse rule of mixtures."""
        calculator = CompositeCalculator()
        
        result = calculator.calculate_inverse_rule_of_mixtures(
            value1=120.0,
            value2=30.0,
            fraction1=0.5,
            fraction2=0.5
        )
        
        # 1/result = 0.5/120 + 0.5/30
        expected = 1.0 / (0.5/120.0 + 0.5/30.0)
        assert abs(result - expected) < 0.01
    
    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        calculator = CompositeCalculator()
        
        result = calculator.calculate_geometric_mean(
            value1=4.0,
            value2=5.0,
            fraction1=0.5,
            fraction2=0.5
        )
        
        expected = (4.0 ** 0.5) * (5.0 ** 0.5)
        assert abs(result - expected) < 0.01
    
    def test_composite_properties_calculation(self):
        """Test calculating all composite properties."""
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()
        
        composite = CompositeSystem(
            component1=sic,
            component2=b4c,
            volume_fraction1=0.7,
            volume_fraction2=0.3
        )
        
        calculator = CompositeCalculator()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            properties = calculator.calculate_composite_properties(composite)
            
            # Should issue limitation warning
            assert len(w) > 0
            assert "COMPOSITE LIMITATION" in str(w[0].message)
        
        # Check that properties were calculated
        assert "hardness" in properties
        assert "fracture_toughness" in properties
        assert "density" in properties
        assert "thermal_conductivity_25C" in properties
        
        # Check hardness is between component values
        assert b4c.mechanical.hardness.value * 0.3 < properties["hardness"].value
        assert properties["hardness"].value < sic.mechanical.hardness.value * 0.7 + b4c.mechanical.hardness.value * 0.3 + 1.0
    
    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through mixing."""
        calculator = CompositeCalculator()
        
        uncertainty = calculator.propagate_uncertainty(
            uncertainty1=2.0,
            uncertainty2=2.5,
            fraction1=0.7,
            fraction2=0.3,
            method="linear"
        )
        
        # Should be positive and account for modeling uncertainty
        assert uncertainty > 0
        # Should be larger than input uncertainties (factor of 2 for modeling)
        assert uncertainty > 2.0
    
    def test_composite_source_marked_as_estimate(self):
        """Test that composite properties are marked as estimates."""
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()
        
        composite = CompositeSystem(
            component1=sic,
            component2=b4c,
            volume_fraction1=0.7,
            volume_fraction2=0.3
        )
        
        calculator = CompositeCalculator()
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            properties = calculator.calculate_composite_properties(composite)
        
        # All properties should be marked as estimates
        for prop in properties.values():
            assert prop.source == "composite_estimate"
    
    def test_validation_against_known_composites(self):
        """Test validation against known composite data."""
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()
        
        # Create known composite (SiC-B4C 70-30)
        composite = CompositeSystem(
            component1=sic,
            component2=b4c,
            volume_fraction1=0.7,
            volume_fraction2=0.3
        )
        
        calculator = CompositeCalculator()
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            properties = calculator.calculate_composite_properties(composite)
        
        validation = calculator.validate_against_known_composites(composite, properties)
        
        # Should have validation results for known composite
        assert len(validation) > 0


class TestConcentrationDependentProperties:
    """Test concentration-dependent property calculations."""
    
    def test_calculate_doped_property(self):
        """Test concentration-dependent property calculation."""
        sic = CeramicSystemFactory.create_sic()
        
        base_value = 28.0
        dopant_effect = 10.0  # +10 per unit concentration
        concentration = 0.02
        
        result = sic.calculate_doped_property(base_value, dopant_effect, concentration)
        
        expected = 28.0 + 10.0 * 0.02
        assert abs(result - expected) < 0.01
    
    def test_all_standard_concentrations(self):
        """Test property calculation at all standard concentrations."""
        sic = CeramicSystemFactory.create_sic()
        
        for concentration in CeramicSystem.SUPPORTED_CONCENTRATIONS:
            doped = CeramicSystemFactory.create_doped_system(
                base_ceramic=sic,
                dopant_element="B",
                dopant_concentration=concentration,
                property_modifications={"hardness": 5.0}
            )
            
            assert doped.dopant_concentration == concentration
            assert doped.is_doped()
