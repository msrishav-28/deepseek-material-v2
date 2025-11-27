"""Tests for composition descriptor calculator."""

import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from pymatgen.core import Composition

from ceramic_discovery.ml.composition_descriptors import CompositionDescriptorCalculator


class TestCompositionDescriptorCalculator:
    """Test composition descriptor calculator."""
    
    def test_calculate_descriptors_sic(self):
        """Test descriptor calculation for SiC."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("SiC")
        
        assert descriptors['num_elements'] == 2.0
        assert descriptors['num_atoms'] == 2.0
        assert 'composition_entropy' in descriptors
        assert descriptors['composition_entropy'] > 0
        assert 'mean_electronegativity' in descriptors
        assert 'mean_atomic_radius' in descriptors
        assert 'mean_atomic_mass' in descriptors
        assert 'mean_atomic_number' in descriptors
    
    def test_calculate_descriptors_tic(self):
        """Test descriptor calculation for TiC."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("TiC")
        
        assert descriptors['num_elements'] == 2.0
        assert descriptors['num_atoms'] == 2.0
        assert 'composition_entropy' in descriptors
        assert 'mean_electronegativity' in descriptors
    
    def test_calculate_descriptors_b4c(self):
        """Test descriptor calculation for B4C."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("B4C")
        
        assert descriptors['num_elements'] == 2.0
        assert descriptors['num_atoms'] == 5.0
        assert 'composition_entropy' in descriptors
    
    def test_single_element_composition(self):
        """Test handling of single-element compositions."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("Si")
        
        assert descriptors['num_elements'] == 1.0
        assert descriptors['num_atoms'] == 1.0
        # Entropy should be 0 for single element
        assert descriptors['composition_entropy'] == 0.0
        # Delta values should be 0 for single element
        assert descriptors['delta_electronegativity'] == 0.0
        assert descriptors['delta_atomic_radius'] == 0.0
    
    def test_complex_composition(self):
        """Test handling of complex compositions."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("Ti0.5Zr0.5C")
        
        assert descriptors['num_elements'] == 3.0
        assert 'composition_entropy' in descriptors
        assert descriptors['composition_entropy'] > 0
    
    def test_invalid_formula(self):
        """Test handling of invalid formula."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("InvalidFormula123")
        
        # Should return empty dict for invalid formula
        assert descriptors == {}
    
    def test_composition_entropy_calculation(self):
        """Test composition entropy calculation."""
        calc = CompositionDescriptorCalculator()
        
        # Equal composition should have maximum entropy for 2 elements
        comp = Composition("AB")
        entropy = calc.calculate_composition_entropy(comp)
        expected = -2 * (0.5 * math.log(0.5))
        assert abs(entropy - expected) < 1e-10
    
    def test_electronegativity_features(self):
        """Test electronegativity feature calculation."""
        calc = CompositionDescriptorCalculator()
        comp = Composition("SiC")
        
        features = calc.calculate_electronegativity_features(comp)
        
        assert 'mean_electronegativity' in features
        assert 'std_electronegativity' in features
        assert 'delta_electronegativity' in features
        assert features['mean_electronegativity'] > 0
        assert features['std_electronegativity'] >= 0
        assert features['delta_electronegativity'] >= 0
    
    def test_atomic_property_features(self):
        """Test atomic property feature calculation."""
        calc = CompositionDescriptorCalculator()
        comp = Composition("SiC")
        
        features = calc.calculate_atomic_property_features(comp)
        
        assert 'mean_atomic_radius' in features
        assert 'std_atomic_radius' in features
        assert 'delta_atomic_radius' in features
        assert 'mean_atomic_mass' in features
        assert 'std_atomic_mass' in features
        assert 'mean_atomic_number' in features
        assert 'weighted_atomic_number' in features
        
        # All values should be positive
        for key, value in features.items():
            assert value >= 0, f"{key} should be non-negative"


class TestCompositionDescriptorProperties:
    """Property-based tests for composition descriptors."""
    
    # Generate valid chemical formulas
    @st.composite
    def chemical_formula(draw):
        """Generate valid chemical formulas."""
        # Common elements in ceramics
        elements = ['Si', 'C', 'Ti', 'Zr', 'Hf', 'B', 'N', 'O', 'Al', 'Mg']
        
        # Pick 1-3 elements
        num_elements = draw(st.integers(min_value=1, max_value=3))
        selected_elements = draw(st.lists(
            st.sampled_from(elements),
            min_size=num_elements,
            max_size=num_elements,
            unique=True
        ))
        
        # Build formula
        formula_parts = []
        for element in selected_elements:
            # Stoichiometry between 1 and 4
            stoich = draw(st.integers(min_value=1, max_value=4))
            if stoich == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{stoich}")
        
        return "".join(formula_parts)
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_property_composition_descriptor_validity(self, formula):
        """
        **Feature: sic-alloy-integration, Property 4: Composition descriptor validity**
        
        Property: For any valid chemical formula, all calculated composition 
        descriptors should be finite numeric values (not NaN or infinite).
        
        **Validates: Requirements 2.1**
        """
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors(formula)
        
        # All descriptors should be finite
        for key, value in descriptors.items():
            assert isinstance(value, (int, float)), \
                f"Descriptor {key} is not numeric for formula {formula}"
            assert math.isfinite(value), \
                f"Descriptor {key} is not finite for formula {formula}: {value}"
            assert not math.isnan(value), \
                f"Descriptor {key} is NaN for formula {formula}"
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_property_feature_consistency(self, formula):
        """
        **Feature: sic-alloy-integration, Property 12: Feature engineering consistency**
        
        Property: For any material, if the same material is processed twice with 
        the same configuration, the calculated features should be identical.
        
        **Validates: Requirements 2.5**
        """
        calc = CompositionDescriptorCalculator()
        
        # Calculate descriptors twice
        descriptors1 = calc.calculate_descriptors(formula)
        descriptors2 = calc.calculate_descriptors(formula)
        
        # Should have same keys
        assert set(descriptors1.keys()) == set(descriptors2.keys()), \
            f"Descriptor keys differ for formula {formula}"
        
        # Should have same values
        for key in descriptors1.keys():
            assert descriptors1[key] == descriptors2[key], \
                f"Descriptor {key} differs between runs for formula {formula}: " \
                f"{descriptors1[key]} vs {descriptors2[key]}"
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_num_elements_positive(self, formula):
        """Test that number of elements is always positive."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors(formula)
        
        if descriptors:  # If parsing succeeded
            assert descriptors['num_elements'] >= 1.0
            assert descriptors['num_atoms'] >= 1.0
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_entropy_non_negative(self, formula):
        """Test that composition entropy is always non-negative."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors(formula)
        
        if 'composition_entropy' in descriptors:
            assert descriptors['composition_entropy'] >= 0.0, \
                f"Entropy is negative for formula {formula}"
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_delta_values_non_negative(self, formula):
        """Test that delta values are always non-negative."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors(formula)
        
        delta_keys = [k for k in descriptors.keys() if k.startswith('delta_')]
        for key in delta_keys:
            assert descriptors[key] >= 0.0, \
                f"Delta value {key} is negative for formula {formula}"
    
    @given(formula=chemical_formula())
    @settings(max_examples=100, deadline=None)
    def test_std_values_non_negative(self, formula):
        """Test that standard deviation values are always non-negative."""
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors(formula)
        
        std_keys = [k for k in descriptors.keys() if k.startswith('std_')]
        for key in std_keys:
            assert descriptors[key] >= 0.0, \
                f"Std value {key} is negative for formula {formula}"
