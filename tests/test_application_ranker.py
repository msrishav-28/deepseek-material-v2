"""Tests for application ranker."""

import math
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from ceramic_discovery.screening.application_ranker import (
    ApplicationRanker,
    ApplicationSpec,
    RankedMaterial
)


class TestApplicationRanker:
    """Test application ranker functionality."""
    
    def test_initialization_default_applications(self):
        """Test initialization with default applications."""
        ranker = ApplicationRanker()
        
        # Should have 5 default applications
        assert len(ranker.applications) == 5
        assert 'aerospace_hypersonic' in ranker.applications
        assert 'cutting_tools' in ranker.applications
        assert 'thermal_barriers' in ranker.applications
        assert 'wear_resistant' in ranker.applications
        assert 'electronic' in ranker.applications
    
    def test_initialization_custom_applications(self):
        """Test initialization with custom applications."""
        custom_app = ApplicationSpec(
            name='custom',
            description='Custom application',
            target_hardness=(20.0, 30.0),
            weight_hardness=1.0
        )
        
        ranker = ApplicationRanker(applications={'custom': custom_app})
        
        assert len(ranker.applications) == 1
        assert 'custom' in ranker.applications
    
    def test_initialization_invalid_application(self):
        """Test initialization with invalid application (no weights)."""
        invalid_app = ApplicationSpec(
            name='invalid',
            description='Invalid application',
            target_hardness=(20.0, 30.0),
            weight_hardness=0.0  # No weights!
        )
        
        with pytest.raises(ValueError, match="has no property weights"):
            ApplicationRanker(applications={'invalid': invalid_app})
    
    def test_calculate_application_score_perfect_match(self):
        """Test scoring with perfect property match."""
        ranker = ApplicationRanker()
        
        # Material with properties in target range for aerospace
        properties = {
            'hardness': 30.0,  # Target: 28-35
            'thermal_conductivity': 100.0,  # Target: 50-150
            'melting_point': 3000.0,  # Target: 2500-4000
        }
        
        score, components = ranker.calculate_application_score(
            properties, 'aerospace_hypersonic'
        )
        
        # Should get perfect score
        assert score == 1.0
        assert components['score_hardness'] == 1.0
        assert components['score_thermal_conductivity'] == 1.0
        assert components['score_melting_point'] == 1.0
    
    def test_calculate_application_score_partial_match(self):
        """Test scoring with partial property match."""
        ranker = ApplicationRanker()
        
        # Material with some properties outside target range
        properties = {
            'hardness': 25.0,  # Below target: 28-35
            'thermal_conductivity': 100.0,  # In target: 50-150
            'melting_point': 3000.0,  # In target: 2500-4000
        }
        
        score, components = ranker.calculate_application_score(
            properties, 'aerospace_hypersonic'
        )
        
        # Should get partial score
        assert 0.0 < score < 1.0
        assert components['score_hardness'] < 1.0
        assert components['score_thermal_conductivity'] == 1.0
        assert components['score_melting_point'] == 1.0
    
    def test_calculate_application_score_missing_properties(self):
        """Test scoring with missing properties."""
        ranker = ApplicationRanker()
        
        # Material with only one property
        properties = {
            'hardness': 30.0,
            'thermal_conductivity': None,
            'melting_point': None,
        }
        
        score, components = ranker.calculate_application_score(
            properties, 'aerospace_hypersonic'
        )
        
        # Should still calculate score based on available properties
        assert 0.0 <= score <= 1.0
        assert components['score_hardness'] == 1.0
        assert components['score_thermal_conductivity'] is None
        assert components['score_melting_point'] is None
    
    def test_calculate_application_score_unknown_application(self):
        """Test scoring with unknown application."""
        ranker = ApplicationRanker()
        
        properties = {'hardness': 30.0}
        
        with pytest.raises(ValueError, match="Unknown application"):
            ranker.calculate_application_score(properties, 'unknown_app')
    
    def test_rank_materials_aerospace(self):
        """Test ranking materials for aerospace application."""
        ranker = ApplicationRanker()
        
        materials = [
            {
                'material_id': 'mat1',
                'formula': 'SiC',
                'hardness': 30.0,
                'thermal_conductivity': 100.0,
                'melting_point': 3000.0,
            },
            {
                'material_id': 'mat2',
                'formula': 'TiC',
                'hardness': 25.0,
                'thermal_conductivity': 80.0,
                'melting_point': 2800.0,
            },
            {
                'material_id': 'mat3',
                'formula': 'B4C',
                'hardness': 32.0,
                'thermal_conductivity': 120.0,
                'melting_point': 3200.0,
            },
        ]
        
        ranked = ranker.rank_materials(materials, 'aerospace_hypersonic')
        
        # Should have 3 ranked materials
        assert len(ranked) == 3
        
        # Should be sorted by score (descending)
        assert ranked[0].overall_score >= ranked[1].overall_score
        assert ranked[1].overall_score >= ranked[2].overall_score
        
        # Ranks should be assigned
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3
        
        # Best materials should have high scores (both SiC and B4C are in range)
        assert ranked[0].overall_score >= 0.9
    
    def test_rank_materials_cutting_tools(self):
        """Test ranking materials for cutting tools application."""
        ranker = ApplicationRanker()
        
        materials = [
            {
                'material_id': 'mat1',
                'formula': 'SiC',
                'hardness': 35.0,
                'formation_energy': -3.0,
                'bulk_modulus': 400.0,
            },
            {
                'material_id': 'mat2',
                'formula': 'TiC',
                'hardness': 32.0,
                'formation_energy': -2.5,
                'bulk_modulus': 350.0,
            },
        ]
        
        ranked = ranker.rank_materials(materials, 'cutting_tools')
        
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
    
    def test_rank_for_all_applications(self):
        """Test ranking materials for all applications."""
        ranker = ApplicationRanker()
        
        materials = [
            {
                'material_id': 'mat1',
                'formula': 'SiC',
                'hardness': 30.0,
                'thermal_conductivity': 100.0,
                'melting_point': 3000.0,
                'formation_energy': -3.0,
                'bulk_modulus': 400.0,
                'band_gap': 2.5,
                'density': 3.2,
            },
        ]
        
        df = ranker.rank_for_all_applications(materials)
        
        # Should have 5 rows (one per application)
        assert len(df) == 5
        
        # Should have all applications
        assert set(df['application'].unique()) == set(ranker.applications.keys())
        
        # All scores should be in [0, 1]
        assert (df['overall_score'] >= 0.0).all()
        assert (df['overall_score'] <= 1.0).all()
    
    def test_get_top_candidates(self):
        """Test getting top candidates."""
        ranker = ApplicationRanker()
        
        materials = [
            {
                'material_id': f'mat{i}',
                'formula': f'Mat{i}',
                'hardness': 30.0 + i,
                'thermal_conductivity': 100.0,
                'melting_point': 3000.0,
            }
            for i in range(20)
        ]
        
        top = ranker.get_top_candidates(
            materials, 'aerospace_hypersonic', n=5
        )
        
        # Should return 5 candidates
        assert len(top) == 5
        
        # Should be sorted by score
        for i in range(len(top) - 1):
            assert top[i].overall_score >= top[i + 1].overall_score
    
    def test_get_top_candidates_with_confidence_filter(self):
        """Test getting top candidates with confidence filter."""
        ranker = ApplicationRanker()
        
        materials = [
            {
                'material_id': 'mat1',
                'formula': 'SiC',
                'hardness': 30.0,
                'thermal_conductivity': 100.0,
                'melting_point': 3000.0,
            },
            {
                'material_id': 'mat2',
                'formula': 'TiC',
                'hardness': 30.0,
                # Missing other properties
            },
        ]
        
        top = ranker.get_top_candidates(
            materials, 'aerospace_hypersonic', n=10, min_confidence=0.8
        )
        
        # Should only include materials with high confidence
        assert all(m.confidence >= 0.8 for m in top)
    
    def test_compare_materials(self):
        """Test comparing two materials."""
        ranker = ApplicationRanker()
        
        mat1 = {
            'material_id': 'mat1',
            'formula': 'SiC',
            'hardness': 32.0,
            'thermal_conductivity': 120.0,
            'melting_point': 3100.0,
        }
        
        mat2 = {
            'material_id': 'mat2',
            'formula': 'TiC',
            'hardness': 28.0,
            'thermal_conductivity': 80.0,
            'melting_point': 2900.0,
        }
        
        comparison = ranker.compare_materials(mat1, mat2, 'aerospace_hypersonic')
        
        assert 'application' in comparison
        assert 'material1' in comparison
        assert 'material2' in comparison
        assert 'winner' in comparison
        assert 'score_difference' in comparison
    
    def test_list_applications(self):
        """Test listing applications."""
        ranker = ApplicationRanker()
        
        apps = ranker.list_applications()
        
        assert len(apps) == 5
        assert 'aerospace_hypersonic' in apps
    
    def test_get_application_spec(self):
        """Test getting application specification."""
        ranker = ApplicationRanker()
        
        spec = ranker.get_application_spec('aerospace_hypersonic')
        
        assert spec.name == 'aerospace_hypersonic'
        assert spec.target_hardness is not None
        assert spec.weight_hardness > 0
    
    def test_get_application_spec_unknown(self):
        """Test getting unknown application specification."""
        ranker = ApplicationRanker()
        
        with pytest.raises(ValueError, match="Unknown application"):
            ranker.get_application_spec('unknown_app')


class TestApplicationRankerProperties:
    """Property-based tests for application ranker."""
    
    # Strategy for generating valid property values
    @st.composite
    def material_properties(draw):
        """Generate valid material properties."""
        return {
            'material_id': draw(st.text(min_size=1, max_size=20)),
            'formula': draw(st.text(min_size=1, max_size=20)),
            'hardness': draw(st.one_of(
                st.none(),
                st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
            )),
            'thermal_conductivity': draw(st.one_of(
                st.none(),
                st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
            )),
            'melting_point': draw(st.one_of(
                st.none(),
                st.floats(min_value=500.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
            )),
            'formation_energy': draw(st.one_of(
                st.none(),
                st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False)
            )),
            'bulk_modulus': draw(st.one_of(
                st.none(),
                st.floats(min_value=50.0, max_value=600.0, allow_nan=False, allow_infinity=False)
            )),
            'band_gap': draw(st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
            )),
            'density': draw(st.one_of(
                st.none(),
                st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False)
            )),
        }
    
    @given(material=material_properties())
    @settings(max_examples=100, deadline=None)
    def test_property_application_scoring_monotonicity(self, material):
        """
        **Feature: sic-alloy-integration, Property 6: Application scoring monotonicity**
        
        Property: For any material and application, if all property values move 
        closer to target ranges, the application score should not decrease.
        
        **Validates: Requirements 3.4**
        """
        ranker = ApplicationRanker()
        
        # Test for aerospace application
        app = 'aerospace_hypersonic'
        spec = ranker.get_application_spec(app)
        
        # Extract properties
        properties = {
            'hardness': material['hardness'],
            'thermal_conductivity': material['thermal_conductivity'],
            'melting_point': material['melting_point'],
            'formation_energy': material['formation_energy'],
            'bulk_modulus': material['bulk_modulus'],
            'band_gap': material['band_gap'],
            'density': material['density'],
        }
        
        # Calculate initial score
        score1, _ = ranker.calculate_application_score(properties, app)
        
        # Create improved properties (move towards target ranges)
        improved_properties = properties.copy()
        
        # Improve hardness if present
        if properties['hardness'] is not None and spec.target_hardness is not None:
            min_h, max_h = spec.target_hardness
            target_h = (min_h + max_h) / 2
            current_h = properties['hardness']
            # Move 50% closer to target
            improved_properties['hardness'] = current_h + 0.5 * (target_h - current_h)
        
        # Improve thermal conductivity if present
        if properties['thermal_conductivity'] is not None and spec.target_thermal_cond is not None:
            min_tc, max_tc = spec.target_thermal_cond
            target_tc = (min_tc + max_tc) / 2
            current_tc = properties['thermal_conductivity']
            improved_properties['thermal_conductivity'] = current_tc + 0.5 * (target_tc - current_tc)
        
        # Improve melting point if present
        if properties['melting_point'] is not None and spec.target_melting_point is not None:
            min_mp, max_mp = spec.target_melting_point
            target_mp = (min_mp + max_mp) / 2
            current_mp = properties['melting_point']
            improved_properties['melting_point'] = current_mp + 0.5 * (target_mp - current_mp)
        
        # Calculate improved score
        score2, _ = ranker.calculate_application_score(improved_properties, app)
        
        # Score should not decrease (allowing for small numerical errors)
        assert score2 >= score1 - 1e-10, \
            f"Score decreased when properties improved: {score1} -> {score2}"
    
    @given(material=material_properties())
    @settings(max_examples=100, deadline=None)
    def test_property_application_score_bounds(self, material):
        """
        **Feature: sic-alloy-integration, Property 7: Application score bounds**
        
        Property: For any material and application, the calculated application 
        score should be in range [0, 1].
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        ranker = ApplicationRanker()
        
        # Test for all applications
        for app_name in ranker.list_applications():
            properties = {
                'hardness': material['hardness'],
                'thermal_conductivity': material['thermal_conductivity'],
                'melting_point': material['melting_point'],
                'formation_energy': material['formation_energy'],
                'bulk_modulus': material['bulk_modulus'],
                'band_gap': material['band_gap'],
                'density': material['density'],
            }
            
            score, _ = ranker.calculate_application_score(properties, app_name)
            
            assert 0.0 <= score <= 1.0, \
                f"Score {score} out of bounds [0, 1] for {app_name}"
            assert math.isfinite(score), \
                f"Score {score} is not finite for {app_name}"
    
    @given(material=material_properties())
    @settings(max_examples=100, deadline=None)
    def test_property_missing_property_handling(self, material):
        """
        **Feature: sic-alloy-integration, Property 8: Missing property handling**
        
        Property: For any material with missing properties, the application 
        scoring should compute partial scores based on available properties 
        without failing.
        
        **Validates: Requirements 3.5**
        """
        ranker = ApplicationRanker()
        
        # Test for all applications
        for app_name in ranker.list_applications():
            properties = {
                'hardness': material['hardness'],
                'thermal_conductivity': material['thermal_conductivity'],
                'melting_point': material['melting_point'],
                'formation_energy': material['formation_energy'],
                'bulk_modulus': material['bulk_modulus'],
                'band_gap': material['band_gap'],
                'density': material['density'],
            }
            
            # Should not raise exception even with missing properties
            try:
                score, components = ranker.calculate_application_score(properties, app_name)
                
                # Score should be valid
                assert 0.0 <= score <= 1.0
                assert math.isfinite(score)
                
                # Component scores should be None for missing properties
                for prop_name, prop_value in properties.items():
                    score_key = f'score_{prop_name}'
                    if score_key in components:
                        if prop_value is None or not np.isfinite(prop_value):
                            # Missing property should have None score
                            assert components[score_key] is None, \
                                f"Missing property {prop_name} should have None score"
                
            except Exception as e:
                pytest.fail(
                    f"Scoring failed with missing properties for {app_name}: {e}"
                )
    
    @given(
        materials=st.lists(material_properties(), min_size=1, max_size=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_ranking_consistency(self, materials):
        """Test that ranking is consistent and deterministic."""
        ranker = ApplicationRanker()
        
        # Rank materials twice
        ranked1 = ranker.rank_materials(materials, 'aerospace_hypersonic')
        ranked2 = ranker.rank_materials(materials, 'aerospace_hypersonic')
        
        # Should have same order
        assert len(ranked1) == len(ranked2)
        for i in range(len(ranked1)):
            assert ranked1[i].material_id == ranked2[i].material_id
            assert ranked1[i].overall_score == ranked2[i].overall_score
    
    @given(material=material_properties())
    @settings(max_examples=100, deadline=None)
    def test_property_confidence_bounds(self, material):
        """Test that confidence is always in [0, 1]."""
        ranker = ApplicationRanker()
        
        materials = [material]
        
        for app_name in ranker.list_applications():
            ranked = ranker.rank_materials(materials, app_name)
            
            for mat in ranked:
                assert 0.0 <= mat.confidence <= 1.0, \
                    f"Confidence {mat.confidence} out of bounds for {app_name}"
                assert math.isfinite(mat.confidence), \
                    f"Confidence {mat.confidence} is not finite for {app_name}"
