"""Tests for experimental planner."""

import pytest
from hypothesis import given, strategies as st, settings, assume

from ceramic_discovery.validation.experimental_planner import (
    ExperimentalPlanner,
    SynthesisMethod,
    CharacterizationTechnique,
    ResourceEstimate,
    SynthesisMethodType,
    DifficultyLevel
)


class TestExperimentalPlanner:
    """Test experimental planner functionality."""
    
    def test_initialization(self):
        """Test planner initialization."""
        planner = ExperimentalPlanner()
        
        # Should have synthesis templates
        assert len(planner.synthesis_templates) == 3
        assert SynthesisMethodType.HOT_PRESSING in planner.synthesis_templates
        assert SynthesisMethodType.SOLID_STATE_SINTERING in planner.synthesis_templates
        assert SynthesisMethodType.SPARK_PLASMA_SINTERING in planner.synthesis_templates
        
        # Should have characterization techniques
        assert len(planner.characterization_techniques) >= 6
        assert "XRD" in planner.characterization_techniques
        assert "SEM" in planner.characterization_techniques
        assert "Hardness Testing" in planner.characterization_techniques
    
    def test_design_synthesis_protocol_sic(self):
        """Test synthesis protocol design for SiC."""
        planner = ExperimentalPlanner()
        
        properties = {
            "melting_point": 3100.0,
            "density": 3.2
        }
        
        methods = planner.design_synthesis_protocol("SiC", properties)
        
        # Should return 3 methods
        assert len(methods) == 3
        
        # Check method types
        method_names = [m.method for m in methods]
        assert SynthesisMethodType.HOT_PRESSING.value in method_names
        assert SynthesisMethodType.SOLID_STATE_SINTERING.value in method_names
        assert SynthesisMethodType.SPARK_PLASMA_SINTERING.value in method_names
        
        # All methods should have valid parameters
        for method in methods:
            assert method.temperature_K > 0
            assert method.pressure_MPa >= 0
            assert method.time_hours > 0
            assert method.cost_factor > 0
            assert len(method.precursors) > 0
    
    def test_design_synthesis_protocol_temperature_scaling(self):
        """Test that synthesis temperatures scale with melting point."""
        planner = ExperimentalPlanner()
        
        # Low melting point material
        low_mp_props = {"melting_point": 2000.0, "density": 3.0}
        low_methods = planner.design_synthesis_protocol("TestMat1", low_mp_props)
        
        # High melting point material
        high_mp_props = {"melting_point": 3500.0, "density": 3.0}
        high_methods = planner.design_synthesis_protocol("TestMat2", high_mp_props)
        
        # Higher melting point should generally lead to higher synthesis temps
        for i in range(len(low_methods)):
            # Allow some flexibility due to capping
            assert high_methods[i].temperature_K >= low_methods[i].temperature_K * 0.9
    
    def test_design_characterization_plan_basic(self):
        """Test basic characterization plan."""
        planner = ExperimentalPlanner()
        
        target_properties = ["hardness", "density"]
        
        techniques = planner.design_characterization_plan("SiC", target_properties)
        
        # Should have at least essential techniques
        assert len(techniques) >= 4
        
        technique_names = [t.technique for t in techniques]
        
        # Essential techniques should be present
        assert "XRD" in technique_names
        assert "SEM" in technique_names
        assert "Density Measurement" in technique_names
        assert "Hardness Testing" in technique_names
        
        # All techniques should have valid parameters
        for technique in techniques:
            assert technique.estimated_time_hours > 0
            assert technique.estimated_cost_dollars > 0
            assert len(technique.purpose) > 0
    
    def test_design_characterization_plan_thermal_properties(self):
        """Test characterization plan with thermal properties."""
        planner = ExperimentalPlanner()
        
        target_properties = ["thermal_conductivity", "thermal_conductivity_25C"]
        
        techniques = planner.design_characterization_plan("SiC", target_properties)
        
        technique_names = [t.technique for t in techniques]
        
        # Should include thermal conductivity measurement
        assert "Thermal Conductivity" in technique_names
    
    def test_design_characterization_plan_fracture_toughness(self):
        """Test characterization plan with fracture toughness."""
        planner = ExperimentalPlanner()
        
        target_properties = ["fracture_toughness"]
        
        techniques = planner.design_characterization_plan("SiC", target_properties)
        
        technique_names = [t.technique for t in techniques]
        
        # Should include fracture toughness measurement
        assert "Fracture Toughness" in technique_names
    
    def test_estimate_resources_basic(self):
        """Test basic resource estimation."""
        planner = ExperimentalPlanner()
        
        # Create simple synthesis and characterization plans
        synthesis_methods = [
            SynthesisMethod(
                method="Hot Pressing",
                temperature_K=2000.0,
                pressure_MPa=30.0,
                time_hours=1.0,
                difficulty="Medium",
                cost_factor=1.0
            )
        ]
        
        characterization_plan = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase identification",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            ),
            CharacterizationTechnique(
                technique="SEM",
                purpose="Microstructure",
                estimated_time_hours=3.0,
                estimated_cost_dollars=300
            )
        ]
        
        estimate = planner.estimate_resources(synthesis_methods, characterization_plan)
        
        # Should have positive values
        assert estimate.timeline_months > 0
        assert estimate.estimated_cost_k_dollars > 0
        assert len(estimate.required_equipment) > 0
        assert len(estimate.required_expertise) > 0
        assert 0 < estimate.confidence_level <= 1.0
    
    def test_estimate_resources_scales_with_complexity(self):
        """Test that resource estimates scale with complexity."""
        planner = ExperimentalPlanner()
        
        # Simple plan
        simple_synthesis = [
            SynthesisMethod(
                method="Solid State Sintering",
                temperature_K=1800.0,
                pressure_MPa=0.0,
                time_hours=4.0,
                difficulty="Easy",
                cost_factor=0.6
            )
        ]
        
        simple_char = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            )
        ]
        
        simple_estimate = planner.estimate_resources(simple_synthesis, simple_char)
        
        # Complex plan
        complex_synthesis = [
            SynthesisMethod(
                method="Spark Plasma Sintering",
                temperature_K=2000.0,
                pressure_MPa=50.0,
                time_hours=0.25,
                difficulty="Hard",
                cost_factor=1.5
            ),
            SynthesisMethod(
                method="Hot Pressing",
                temperature_K=2100.0,
                pressure_MPa=40.0,
                time_hours=1.0,
                difficulty="Medium",
                cost_factor=1.0
            )
        ]
        
        complex_char = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            ),
            CharacterizationTechnique(
                technique="TEM",
                purpose="Microstructure",
                estimated_time_hours=8.0,
                estimated_cost_dollars=1000
            ),
            CharacterizationTechnique(
                technique="XPS",
                purpose="Chemistry",
                estimated_time_hours=4.0,
                estimated_cost_dollars=600
            )
        ]
        
        complex_estimate = planner.estimate_resources(complex_synthesis, complex_char)
        
        # Complex plan should cost more and take longer
        assert complex_estimate.estimated_cost_k_dollars > simple_estimate.estimated_cost_k_dollars
        assert complex_estimate.timeline_months >= simple_estimate.timeline_months
    
    def test_validate_protocol_valid(self):
        """Test protocol validation with valid protocol."""
        planner = ExperimentalPlanner()
        
        synthesis_methods = [
            SynthesisMethod(
                method="Hot Pressing",
                temperature_K=2000.0,
                pressure_MPa=30.0,
                time_hours=1.0,
                difficulty="Medium",
                cost_factor=1.0
            )
        ]
        
        characterization_plan = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            ),
            CharacterizationTechnique(
                technique="SEM",
                purpose="Microstructure",
                estimated_time_hours=3.0,
                estimated_cost_dollars=300
            ),
            CharacterizationTechnique(
                technique="Hardness Testing",
                purpose="Hardness",
                estimated_time_hours=1.0,
                estimated_cost_dollars=150
            )
        ]
        
        validation = planner.validate_protocol(synthesis_methods, characterization_plan)
        
        assert validation["is_valid"] is True
        assert validation["completeness_score"] > 0
        assert len(validation["errors"]) == 0
    
    def test_validate_protocol_no_synthesis(self):
        """Test protocol validation with no synthesis methods."""
        planner = ExperimentalPlanner()
        
        characterization_plan = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            )
        ]
        
        validation = planner.validate_protocol([], characterization_plan)
        
        assert validation["is_valid"] is False
        assert len(validation["errors"]) > 0
        assert any("synthesis" in err.lower() for err in validation["errors"])
    
    def test_validate_protocol_no_characterization(self):
        """Test protocol validation with no characterization."""
        planner = ExperimentalPlanner()
        
        synthesis_methods = [
            SynthesisMethod(
                method="Hot Pressing",
                temperature_K=2000.0,
                pressure_MPa=30.0,
                time_hours=1.0,
                difficulty="Medium",
                cost_factor=1.0
            )
        ]
        
        validation = planner.validate_protocol(synthesis_methods, [])
        
        assert validation["is_valid"] is False
        assert len(validation["errors"]) > 0
        assert any("characterization" in err.lower() for err in validation["errors"])
    
    def test_validate_protocol_extreme_temperature(self):
        """Test protocol validation with extreme temperature."""
        planner = ExperimentalPlanner()
        
        synthesis_methods = [
            SynthesisMethod(
                method="Hot Pressing",
                temperature_K=5000.0,  # Too high!
                pressure_MPa=30.0,
                time_hours=1.0,
                difficulty="Medium",
                cost_factor=1.0
            )
        ]
        
        characterization_plan = [
            CharacterizationTechnique(
                technique="XRD",
                purpose="Phase",
                estimated_time_hours=2.0,
                estimated_cost_dollars=200
            )
        ]
        
        validation = planner.validate_protocol(synthesis_methods, characterization_plan)
        
        # Should have warnings about temperature
        assert len(validation["warnings"]) > 0
        assert any("temperature" in warn.lower() for warn in validation["warnings"])
    
    def test_generate_experimental_protocol_complete(self):
        """Test complete protocol generation."""
        planner = ExperimentalPlanner()
        
        properties = {
            "melting_point": 3100.0,
            "density": 3.2,
            "hardness": 30.0
        }
        
        target_properties = ["hardness", "thermal_conductivity", "density"]
        
        protocol = planner.generate_experimental_protocol(
            "SiC", properties, target_properties
        )
        
        # Should have all components
        assert "formula" in protocol
        assert "synthesis_methods" in protocol
        assert "characterization_plan" in protocol
        assert "resource_estimate" in protocol
        assert "validation" in protocol
        assert "is_valid" in protocol
        assert "completeness_score" in protocol
        
        # Should be valid
        assert protocol["is_valid"] is True
        
        # Should have synthesis methods
        assert len(protocol["synthesis_methods"]) >= 3
        
        # Should have characterization techniques
        assert len(protocol["characterization_plan"]) >= 4
        
        # Should have resource estimate
        assert protocol["resource_estimate"].timeline_months > 0
        assert protocol["resource_estimate"].estimated_cost_k_dollars > 0


class TestExperimentalPlannerProperties:
    """Property-based tests for experimental planner."""
    
    # Strategy for generating valid material properties
    @st.composite
    def material_properties(draw):
        """Generate valid material properties."""
        return {
            'melting_point': draw(st.floats(
                min_value=1500.0, max_value=4000.0, 
                allow_nan=False, allow_infinity=False
            )),
            'density': draw(st.floats(
                min_value=1.0, max_value=20.0,
                allow_nan=False, allow_infinity=False
            )),
            'hardness': draw(st.one_of(
                st.none(),
                st.floats(min_value=10.0, max_value=50.0, 
                         allow_nan=False, allow_infinity=False)
            )),
            'thermal_conductivity': draw(st.one_of(
                st.none(),
                st.floats(min_value=1.0, max_value=500.0,
                         allow_nan=False, allow_infinity=False)
            ))
        }
    
    @st.composite
    def target_property_list(draw):
        """Generate list of target properties."""
        all_properties = [
            "hardness", "density", "thermal_conductivity",
            "fracture_toughness", "band_gap", "oxidation"
        ]
        # Select 1-6 properties
        n = draw(st.integers(min_value=1, max_value=6))
        return draw(st.lists(
            st.sampled_from(all_properties),
            min_size=n, max_size=n, unique=True
        ))
    
    @given(
        formula=st.text(min_size=2, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd')
        )),
        properties=material_properties()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_experimental_protocol_completeness(self, formula, properties):
        """
        **Feature: sic-alloy-integration, Property 9: Experimental protocol completeness**
        
        Property: For any material candidate, the generated experimental protocol 
        should include at least one synthesis method and at least one characterization 
        technique.
        
        **Validates: Requirements 4.1, 4.3**
        """
        # Skip empty formulas
        assume(len(formula.strip()) > 0)
        
        planner = ExperimentalPlanner()
        
        # Generate synthesis protocol
        synthesis_methods = planner.design_synthesis_protocol(formula, properties)
        
        # Property: Should have at least one synthesis method
        assert len(synthesis_methods) >= 1, \
            f"Expected at least 1 synthesis method, got {len(synthesis_methods)}"
        
        # Generate characterization plan
        target_properties = ["hardness", "density"]
        characterization_plan = planner.design_characterization_plan(
            formula, target_properties
        )
        
        # Property: Should have at least one characterization technique
        assert len(characterization_plan) >= 1, \
            f"Expected at least 1 characterization technique, got {len(characterization_plan)}"
        
        # Additional check: All synthesis methods should have valid parameters
        for method in synthesis_methods:
            assert method.temperature_K > 0, \
                f"Temperature must be positive, got {method.temperature_K}"
            assert method.pressure_MPa >= 0, \
                f"Pressure must be non-negative, got {method.pressure_MPa}"
            assert method.time_hours > 0, \
                f"Time must be positive, got {method.time_hours}"
            assert method.cost_factor > 0, \
                f"Cost factor must be positive, got {method.cost_factor}"
        
        # Additional check: All characterization techniques should have valid parameters
        for technique in characterization_plan:
            assert technique.estimated_time_hours > 0, \
                f"Time must be positive, got {technique.estimated_time_hours}"
            assert technique.estimated_cost_dollars > 0, \
                f"Cost must be positive, got {technique.estimated_cost_dollars}"
    
    @given(
        formula=st.text(min_size=2, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd')
        )),
        properties=material_properties(),
        target_props=target_property_list()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_resource_estimate_positivity(self, formula, properties, target_props):
        """
        **Feature: sic-alloy-integration, Property 10: Resource estimate positivity**
        
        Property: For any experimental protocol, the estimated timeline and cost 
        should be positive values.
        
        **Validates: Requirements 4.4**
        """
        # Skip empty formulas
        assume(len(formula.strip()) > 0)
        
        planner = ExperimentalPlanner()
        
        # Generate complete protocol
        synthesis_methods = planner.design_synthesis_protocol(formula, properties)
        characterization_plan = planner.design_characterization_plan(formula, target_props)
        
        # Estimate resources
        estimate = planner.estimate_resources(synthesis_methods, characterization_plan)
        
        # Property: Timeline must be positive
        assert estimate.timeline_months > 0, \
            f"Timeline must be positive, got {estimate.timeline_months}"
        
        # Property: Cost must be positive
        assert estimate.estimated_cost_k_dollars > 0, \
            f"Cost must be positive, got {estimate.estimated_cost_k_dollars}"
        
        # Additional checks: Confidence level should be in [0, 1]
        assert 0 < estimate.confidence_level <= 1.0, \
            f"Confidence must be in (0, 1], got {estimate.confidence_level}"
        
        # Additional checks: Should have equipment and expertise lists
        assert len(estimate.required_equipment) > 0, \
            "Should have at least one required equipment"
        assert len(estimate.required_expertise) > 0, \
            "Should have at least one required expertise"
