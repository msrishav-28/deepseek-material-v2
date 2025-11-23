"""Tests for mechanism analysis system."""

import pytest
import numpy as np
import pandas as pd

from ceramic_discovery.ballistics import (
    MechanismAnalyzer,
    ThermalConductivityAnalysis,
    PropertyInteraction,
    HypothesisTest,
)


@pytest.fixture
def sample_mechanism_data():
    """Create sample data for mechanism analysis."""
    np.random.seed(42)
    n_samples = 50
    
    # Generate synthetic material properties with known relationships
    hardness = np.random.uniform(20, 35, n_samples)
    fracture_toughness = np.random.uniform(3, 6, n_samples)
    density = np.random.uniform(2.5, 4.0, n_samples)
    thermal_conductivity = np.random.uniform(30, 60, n_samples)
    youngs_modulus = np.random.uniform(350, 450, n_samples)
    
    # Generate V50 with known contributions
    v50 = (
        20 * hardness +  # Strong positive effect
        50 * fracture_toughness +  # Strong positive effect
        100 * density +  # Strong positive effect
        2 * thermal_conductivity +  # Weak positive effect
        0.5 * youngs_modulus +  # Weak positive effect
        np.random.normal(0, 50, n_samples)
    )
    
    materials = pd.DataFrame({
        'hardness': hardness,
        'fracture_toughness': fracture_toughness,
        'density': density,
        'thermal_conductivity_1000C': thermal_conductivity,
        'youngs_modulus': youngs_modulus,
    })
    
    v50_series = pd.Series(v50, name='v50')
    
    return materials, v50_series


class TestThermalConductivityAnalysis:
    """Test thermal conductivity analysis."""
    
    def test_thermal_analysis_creation(self):
        """Test creating thermal conductivity analysis."""
        analysis = ThermalConductivityAnalysis(
            correlation_with_v50=0.45,
            p_value=0.01,
            contribution_score=0.15,
            mechanism_description="Test mechanism",
        )
        
        assert analysis.correlation_with_v50 == 0.45
        assert analysis.p_value == 0.01
        assert analysis.contribution_score == 0.15
        assert analysis.mechanism_description == "Test mechanism"
    
    def test_is_significant(self):
        """Test significance checking."""
        # Significant
        analysis1 = ThermalConductivityAnalysis(
            correlation_with_v50=0.45,
            p_value=0.01,
            contribution_score=0.15,
            mechanism_description="Test",
        )
        assert analysis1.is_significant(alpha=0.05) is True
        
        # Not significant
        analysis2 = ThermalConductivityAnalysis(
            correlation_with_v50=0.10,
            p_value=0.20,
            contribution_score=0.05,
            mechanism_description="Test",
        )
        assert analysis2.is_significant(alpha=0.05) is False


class TestPropertyInteraction:
    """Test property interaction analysis."""
    
    def test_interaction_creation(self):
        """Test creating property interaction."""
        interaction = PropertyInteraction(
            property1='hardness',
            property2='fracture_toughness',
            interaction_strength=0.3,
            combined_effect_on_v50=0.8,
            synergy_score=0.2,
        )
        
        assert interaction.property1 == 'hardness'
        assert interaction.property2 == 'fracture_toughness'
        assert interaction.interaction_strength == 0.3
        assert interaction.combined_effect_on_v50 == 0.8
        assert interaction.synergy_score == 0.2
    
    def test_is_synergistic(self):
        """Test synergy checking."""
        # Synergistic
        interaction1 = PropertyInteraction(
            property1='hardness',
            property2='toughness',
            interaction_strength=0.3,
            combined_effect_on_v50=0.8,
            synergy_score=0.2,
        )
        assert interaction1.is_synergistic() is True
        
        # Not synergistic
        interaction2 = PropertyInteraction(
            property1='hardness',
            property2='density',
            interaction_strength=0.1,
            combined_effect_on_v50=0.5,
            synergy_score=0.05,
        )
        assert interaction2.is_synergistic() is False


class TestHypothesisTest:
    """Test hypothesis testing."""
    
    def test_hypothesis_creation(self):
        """Test creating hypothesis test."""
        test = HypothesisTest(
            hypothesis="Hardness correlates with V50",
            test_statistic=0.65,
            p_value=0.01,
            conclusion="Significant correlation found",
        )
        
        assert test.hypothesis == "Hardness correlates with V50"
        assert test.test_statistic == 0.65
        assert test.p_value == 0.01
        assert test.conclusion == "Significant correlation found"
    
    def test_is_significant(self):
        """Test significance checking."""
        # Significant
        test1 = HypothesisTest(
            hypothesis="Test",
            test_statistic=2.5,
            p_value=0.01,
            conclusion="Significant",
        )
        assert test1.is_significant() is True
        
        # Not significant
        test2 = HypothesisTest(
            hypothesis="Test",
            test_statistic=0.5,
            p_value=0.20,
            conclusion="Not significant",
        )
        assert test2.is_significant() is False


class TestMechanismAnalyzer:
    """Test MechanismAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test initializing mechanism analyzer."""
        analyzer = MechanismAnalyzer(random_seed=42)
        
        assert analyzer.random_seed == 42
    
    def test_thermal_conductivity_impact(self, sample_mechanism_data):
        """Test thermal conductivity impact analysis."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        analysis = analyzer.analyze_thermal_conductivity_impact(materials, v50_values)
        
        assert isinstance(analysis, ThermalConductivityAnalysis)
        assert -1 <= analysis.correlation_with_v50 <= 1
        assert 0 <= analysis.p_value <= 1
        assert 0 <= analysis.contribution_score <= 1
        assert len(analysis.mechanism_description) > 0
    
    def test_thermal_conductivity_missing_data(self):
        """Test thermal analysis with missing data."""
        materials = pd.DataFrame({'hardness': [28.0, 30.0]})
        v50_values = pd.Series([800.0, 850.0])
        analyzer = MechanismAnalyzer()
        
        analysis = analyzer.analyze_thermal_conductivity_impact(materials, v50_values)
        
        assert analysis.correlation_with_v50 == 0.0
        assert analysis.p_value == 1.0
        assert "not available" in analysis.mechanism_description
    
    def test_property_contributions(self, sample_mechanism_data):
        """Test property contribution analysis."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        contributions = analyzer.analyze_property_contributions(materials, v50_values)
        
        assert isinstance(contributions, pd.DataFrame)
        assert len(contributions) > 0
        assert 'property' in contributions.columns
        assert 'pearson_correlation' in contributions.columns
        assert 'effect_size' in contributions.columns
        
        # Check all correlations are in valid range
        assert all(-1 <= corr <= 1 for corr in contributions['pearson_correlation'])
    
    def test_property_contributions_with_importances(self, sample_mechanism_data):
        """Test property contributions with feature importances."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        # Mock feature importances
        feature_names = list(materials.columns)
        feature_importances = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        contributions = analyzer.analyze_property_contributions(
            materials, v50_values, feature_importances, feature_names
        )
        
        assert 'feature_importance' in contributions.columns
        assert contributions['feature_importance'].sum() > 0
    
    def test_property_interactions(self, sample_mechanism_data):
        """Test property interaction analysis."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        interactions = analyzer.analyze_property_interactions(materials, v50_values, top_n=3)
        
        assert isinstance(interactions, list)
        assert len(interactions) <= 3
        
        for interaction in interactions:
            assert isinstance(interaction, PropertyInteraction)
            assert interaction.property1 in materials.columns
            assert interaction.property2 in materials.columns
            assert -1 <= interaction.interaction_strength <= 1
    
    def test_property_interactions_insufficient_data(self):
        """Test property interactions with insufficient data."""
        materials = pd.DataFrame({'hardness': [28.0]})
        v50_values = pd.Series([800.0])
        analyzer = MechanismAnalyzer()
        
        interactions = analyzer.analyze_property_interactions(materials, v50_values)
        
        assert len(interactions) == 0
    
    def test_hypothesis_testing(self, sample_mechanism_data):
        """Test performance hypothesis testing."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        hypotheses = analyzer.test_performance_hypotheses(materials, v50_values)
        
        assert isinstance(hypotheses, list)
        assert len(hypotheses) > 0
        
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, HypothesisTest)
            assert len(hypothesis.hypothesis) > 0
            assert len(hypothesis.conclusion) > 0
            assert 0 <= hypothesis.p_value <= 1
    
    def test_hypothesis_hardness_correlation(self, sample_mechanism_data):
        """Test hardness correlation hypothesis."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        hypotheses = analyzer.test_performance_hypotheses(materials, v50_values)
        
        # Find hardness hypothesis
        hardness_hyp = next(
            (h for h in hypotheses if 'Hardness' in h.hypothesis),
            None
        )
        
        assert hardness_hyp is not None
        # Should be significant since we generated data with strong hardness effect
        assert hardness_hyp.is_significant()
    
    def test_generate_mechanism_report(self, sample_mechanism_data):
        """Test comprehensive mechanism report generation."""
        materials, v50_values = sample_mechanism_data
        analyzer = MechanismAnalyzer()
        
        report = analyzer.generate_mechanism_report(materials, v50_values)
        
        # Check report structure
        assert 'thermal_conductivity_analysis' in report
        assert 'property_contributions' in report
        assert 'property_interactions' in report
        assert 'hypothesis_tests' in report
        assert 'summary' in report
        
        # Check thermal analysis
        assert isinstance(report['thermal_conductivity_analysis'], ThermalConductivityAnalysis)
        
        # Check property contributions
        assert isinstance(report['property_contributions'], pd.DataFrame)
        assert len(report['property_contributions']) > 0
        
        # Check interactions
        assert isinstance(report['property_interactions'], list)
        
        # Check hypotheses
        assert isinstance(report['hypothesis_tests'], list)
        assert len(report['hypothesis_tests']) > 0
        
        # Check summary
        assert 'n_materials' in report['summary']
        assert 'v50_mean' in report['summary']
        assert report['summary']['n_materials'] == len(materials)
