"""Tests for structure descriptor calculator."""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from ceramic_discovery.ml.structure_descriptors import StructureDescriptorCalculator


class TestStructureDescriptorCalculator:
    """Test structure descriptor calculator."""
    
    def test_calculate_descriptors_with_moduli(self):
        """Test descriptor calculation with bulk and shear moduli."""
        calc = StructureDescriptorCalculator()
        properties = {
            'bulk_modulus': 220.0,
            'shear_modulus': 190.0
        }
        descriptors = calc.calculate_descriptors(properties)
        
        assert 'pugh_ratio' in descriptors
        assert 'is_ductile' in descriptors
        assert 'youngs_modulus' in descriptors
        assert 'poisson_ratio' in descriptors
        
        # Check Pugh ratio calculation
        expected_pugh = 220.0 / 190.0
        assert abs(descriptors['pugh_ratio'] - expected_pugh) < 1e-10
    
    def test_calculate_descriptors_with_energy_density(self):
        """Test descriptor calculation with formation energy and density."""
        calc = StructureDescriptorCalculator()
        properties = {
            'formation_energy': -2.5,
            'density': 3.2
        }
        descriptors = calc.calculate_descriptors(properties)
        
        assert 'energy_density' in descriptors
        expected_ed = abs(-2.5) / 3.2
        assert abs(descriptors['energy_density'] - expected_ed) < 1e-10
    
    def test_calculate_descriptors_complete(self):
        """Test descriptor calculation with all properties."""
        calc = StructureDescriptorCalculator()
        properties = {
            'bulk_modulus': 220.0,
            'shear_modulus': 190.0,
            'formation_energy': -2.5,
            'density': 3.2
        }
        descriptors = calc.calculate_descriptors(properties)
        
        assert 'pugh_ratio' in descriptors
        assert 'is_ductile' in descriptors
        assert 'youngs_modulus' in descriptors
        assert 'poisson_ratio' in descriptors
        assert 'energy_density' in descriptors
    
    def test_calculate_descriptors_missing_properties(self):
        """Test descriptor calculation with missing properties."""
        calc = StructureDescriptorCalculator()
        properties = {}
        descriptors = calc.calculate_descriptors(properties)
        
        # Should return empty dict when no properties available
        assert descriptors == {}
    
    def test_calculate_pugh_ratio(self):
        """Test Pugh ratio calculation."""
        calc = StructureDescriptorCalculator()
        
        # Test normal case
        ratio = calc.calculate_pugh_ratio(220.0, 190.0)
        expected = 220.0 / 190.0
        assert abs(ratio - expected) < 1e-10
        
        # Test ductile material (B/G > 1.75)
        ratio = calc.calculate_pugh_ratio(200.0, 100.0)
        assert ratio == 2.0
        assert ratio > 1.75
        
        # Test brittle material (B/G < 1.75)
        ratio = calc.calculate_pugh_ratio(150.0, 100.0)
        assert ratio == 1.5
        assert ratio < 1.75
    
    def test_calculate_pugh_ratio_invalid_inputs(self):
        """Test Pugh ratio with invalid inputs."""
        calc = StructureDescriptorCalculator()
        
        # Zero modulus
        assert calc.calculate_pugh_ratio(0, 190.0) is None
        assert calc.calculate_pugh_ratio(220.0, 0) is None
        
        # Negative modulus
        assert calc.calculate_pugh_ratio(-220.0, 190.0) is None
        assert calc.calculate_pugh_ratio(220.0, -190.0) is None
        
        # Infinite values
        assert calc.calculate_pugh_ratio(float('inf'), 190.0) is None
        assert calc.calculate_pugh_ratio(220.0, float('inf')) is None
        
        # NaN values
        assert calc.calculate_pugh_ratio(float('nan'), 190.0) is None
        assert calc.calculate_pugh_ratio(220.0, float('nan')) is None
    
    def test_calculate_elastic_properties(self):
        """Test elastic property calculation."""
        calc = StructureDescriptorCalculator()
        
        B = 220.0
        G = 190.0
        
        props = calc.calculate_elastic_properties(B, G)
        
        assert 'youngs_modulus' in props
        assert 'poisson_ratio' in props
        
        # Check Young's modulus: E = 9BG / (3B + G)
        expected_E = (9 * B * G) / (3 * B + G)
        assert abs(props['youngs_modulus'] - expected_E) < 1e-10
        
        # Check Poisson ratio: ν = (3B - 2G) / (2(3B + G))
        expected_nu = (3 * B - 2 * G) / (2 * (3 * B + G))
        assert abs(props['poisson_ratio'] - expected_nu) < 1e-10
    
    def test_poisson_ratio_bounds(self):
        """Test that Poisson ratio stays within physical bounds."""
        calc = StructureDescriptorCalculator()
        
        # Test various B/G ratios
        test_cases = [
            (220.0, 190.0),  # Typical ceramic
            (100.0, 50.0),   # Ductile
            (150.0, 100.0),  # Brittle
            (200.0, 80.0),   # Very ductile
        ]
        
        for B, G in test_cases:
            props = calc.calculate_elastic_properties(B, G)
            if 'poisson_ratio' in props:
                nu = props['poisson_ratio']
                assert -1.0 <= nu <= 0.5, \
                    f"Poisson ratio {nu} outside bounds for B={B}, G={G}"
    
    def test_elastic_properties_invalid_inputs(self):
        """Test elastic properties with invalid inputs."""
        calc = StructureDescriptorCalculator()
        
        # Zero modulus
        props = calc.calculate_elastic_properties(0, 190.0)
        assert props == {}
        
        # Negative modulus
        props = calc.calculate_elastic_properties(-220.0, 190.0)
        assert props == {}
        
        # Infinite values
        props = calc.calculate_elastic_properties(float('inf'), 190.0)
        assert props == {}
    
    def test_calculate_energy_density(self):
        """Test energy density calculation."""
        calc = StructureDescriptorCalculator()
        
        # Test with negative formation energy
        ed = calc.calculate_energy_density(-2.5, 3.2)
        expected = abs(-2.5) / 3.2
        assert abs(ed - expected) < 1e-10
        
        # Test with positive formation energy (should still use absolute value)
        ed = calc.calculate_energy_density(2.5, 3.2)
        expected = abs(2.5) / 3.2
        assert abs(ed - expected) < 1e-10
        
        # Energy density should always be positive
        assert ed > 0
    
    def test_calculate_energy_density_invalid_inputs(self):
        """Test energy density with invalid inputs."""
        calc = StructureDescriptorCalculator()
        
        # Zero density
        assert calc.calculate_energy_density(-2.5, 0) is None
        
        # Negative density
        assert calc.calculate_energy_density(-2.5, -3.2) is None
        
        # Infinite values
        assert calc.calculate_energy_density(float('inf'), 3.2) is None
        assert calc.calculate_energy_density(-2.5, float('inf')) is None
        
        # NaN values
        assert calc.calculate_energy_density(float('nan'), 3.2) is None
        assert calc.calculate_energy_density(-2.5, float('nan')) is None
    
    def test_ductility_indicator(self):
        """Test ductility indicator based on Pugh ratio."""
        calc = StructureDescriptorCalculator()
        
        # Ductile material (B/G > 1.75)
        props = {'bulk_modulus': 200.0, 'shear_modulus': 100.0}
        descriptors = calc.calculate_descriptors(props)
        assert descriptors['is_ductile'] == 1.0
        
        # Brittle material (B/G < 1.75)
        props = {'bulk_modulus': 150.0, 'shear_modulus': 100.0}
        descriptors = calc.calculate_descriptors(props)
        assert descriptors['is_ductile'] == 0.0
        
        # Boundary case (B/G = 1.75)
        props = {'bulk_modulus': 175.0, 'shear_modulus': 100.0}
        descriptors = calc.calculate_descriptors(props)
        assert descriptors['is_ductile'] == 0.0  # Exactly at boundary is brittle


class TestStructureDescriptorProperties:
    """Property-based tests for structure descriptors."""
    
    @given(
        bulk_modulus=st.floats(min_value=1.0, max_value=500.0),
        shear_modulus=st.floats(min_value=1.0, max_value=500.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_structure_descriptor_physical_bounds(self, bulk_modulus, shear_modulus):
        """
        **Feature: sic-alloy-integration, Property 5: Structure descriptor physical bounds**
        
        Property: For any material with bulk modulus B and shear modulus G where 
        both are positive, the calculated Pugh ratio should equal B/G and Poisson 
        ratio should be in range [-1, 0.5].
        
        **Validates: Requirements 2.2**
        """
        # Filter out non-finite values
        assume(math.isfinite(bulk_modulus))
        assume(math.isfinite(shear_modulus))
        assume(bulk_modulus > 0)
        assume(shear_modulus > 0)
        
        calc = StructureDescriptorCalculator()
        
        # Test Pugh ratio
        pugh_ratio = calc.calculate_pugh_ratio(bulk_modulus, shear_modulus)
        assert pugh_ratio is not None, \
            f"Pugh ratio is None for B={bulk_modulus}, G={shear_modulus}"
        
        expected_pugh = bulk_modulus / shear_modulus
        assert abs(pugh_ratio - expected_pugh) < 1e-6, \
            f"Pugh ratio {pugh_ratio} != {expected_pugh} for B={bulk_modulus}, G={shear_modulus}"
        
        # Test elastic properties
        elastic_props = calc.calculate_elastic_properties(bulk_modulus, shear_modulus)
        
        # Poisson ratio should be in physical bounds
        if 'poisson_ratio' in elastic_props:
            poisson_ratio = elastic_props['poisson_ratio']
            assert -1.0 <= poisson_ratio <= 0.5, \
                f"Poisson ratio {poisson_ratio} outside bounds [-1, 0.5] for B={bulk_modulus}, G={shear_modulus}"
        
        # Young's modulus should be positive
        if 'youngs_modulus' in elastic_props:
            youngs_modulus = elastic_props['youngs_modulus']
            assert youngs_modulus > 0, \
                f"Young's modulus {youngs_modulus} is not positive for B={bulk_modulus}, G={shear_modulus}"
    
    @given(
        bulk_modulus=st.floats(min_value=1.0, max_value=500.0),
        shear_modulus=st.floats(min_value=1.0, max_value=500.0),
        formation_energy=st.floats(min_value=-10.0, max_value=0.0),
        density=st.floats(min_value=0.1, max_value=20.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_all_descriptors_finite(self, bulk_modulus, shear_modulus, 
                                            formation_energy, density):
        """Test that all calculated descriptors are finite."""
        # Filter out non-finite values
        assume(math.isfinite(bulk_modulus))
        assume(math.isfinite(shear_modulus))
        assume(math.isfinite(formation_energy))
        assume(math.isfinite(density))
        assume(bulk_modulus > 0)
        assume(shear_modulus > 0)
        assume(density > 0)
        
        calc = StructureDescriptorCalculator()
        properties = {
            'bulk_modulus': bulk_modulus,
            'shear_modulus': shear_modulus,
            'formation_energy': formation_energy,
            'density': density
        }
        
        descriptors = calc.calculate_descriptors(properties)
        
        # All descriptors should be finite
        for key, value in descriptors.items():
            assert isinstance(value, (int, float)), \
                f"Descriptor {key} is not numeric"
            assert math.isfinite(value), \
                f"Descriptor {key} is not finite: {value}"
            assert not math.isnan(value), \
                f"Descriptor {key} is NaN"
    
    @given(
        formation_energy=st.floats(min_value=-10.0, max_value=0.0),
        density=st.floats(min_value=0.1, max_value=20.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_energy_density_positive(self, formation_energy, density):
        """Test that energy density is always non-negative."""
        assume(math.isfinite(formation_energy))
        assume(math.isfinite(density))
        assume(density > 0)
        
        calc = StructureDescriptorCalculator()
        energy_density = calc.calculate_energy_density(formation_energy, density)
        
        if energy_density is not None:
            assert energy_density >= 0, \
                f"Energy density {energy_density} is negative for E={formation_energy}, ρ={density}"
    
    @given(
        bulk_modulus=st.floats(min_value=1.0, max_value=500.0),
        shear_modulus=st.floats(min_value=1.0, max_value=500.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_pugh_ratio_consistency(self, bulk_modulus, shear_modulus):
        """Test that Pugh ratio calculation is consistent."""
        assume(math.isfinite(bulk_modulus))
        assume(math.isfinite(shear_modulus))
        assume(bulk_modulus > 0)
        assume(shear_modulus > 0)
        
        calc = StructureDescriptorCalculator()
        
        # Calculate twice
        ratio1 = calc.calculate_pugh_ratio(bulk_modulus, shear_modulus)
        ratio2 = calc.calculate_pugh_ratio(bulk_modulus, shear_modulus)
        
        assert ratio1 == ratio2, \
            f"Pugh ratio differs between runs: {ratio1} vs {ratio2}"
    
    @given(
        bulk_modulus=st.floats(min_value=1.0, max_value=500.0),
        shear_modulus=st.floats(min_value=1.0, max_value=500.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_ductility_indicator_consistency(self, bulk_modulus, shear_modulus):
        """Test that ductility indicator is consistent with Pugh ratio."""
        assume(math.isfinite(bulk_modulus))
        assume(math.isfinite(shear_modulus))
        assume(bulk_modulus > 0)
        assume(shear_modulus > 0)
        
        calc = StructureDescriptorCalculator()
        properties = {
            'bulk_modulus': bulk_modulus,
            'shear_modulus': shear_modulus
        }
        
        descriptors = calc.calculate_descriptors(properties)
        
        if 'pugh_ratio' in descriptors and 'is_ductile' in descriptors:
            pugh_ratio = descriptors['pugh_ratio']
            is_ductile = descriptors['is_ductile']
            
            # Check consistency
            if pugh_ratio > 1.75:
                assert is_ductile == 1.0, \
                    f"Material with Pugh ratio {pugh_ratio} should be ductile"
            else:
                assert is_ductile == 0.0, \
                    f"Material with Pugh ratio {pugh_ratio} should be brittle"
