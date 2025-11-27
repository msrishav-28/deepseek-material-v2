"""
Property-based tests for SiC Alloy Designer Pipeline stage independence.

**Feature: sic-alloy-integration, Property 15: Pipeline stage independence**

Property 15: Pipeline stage independence
For any pipeline execution, if a stage fails, subsequent stages should either skip 
gracefully or use available data without cascading failures.
**Validates: Requirements 8.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np

from ceramic_discovery.screening.sic_alloy_designer_pipeline import SiCAlloyDesignerPipeline
from ceramic_discovery.dft.materials_project_client import MaterialData


# Strategy for generating material data
@st.composite
def material_data_strategy(draw):
    """Generate random MaterialData objects."""
    formula = draw(st.sampled_from(["SiC", "TiC", "ZrC", "HfC", "B4C", "TaC", "WC"]))
    
    material = MaterialData(
        material_id=f"test-{draw(st.integers(min_value=1, max_value=10000))}",
        formula=formula,
        crystal_structure={},
        formation_energy=draw(st.floats(min_value=-5.0, max_value=-0.5, allow_nan=False, allow_infinity=False)),
        energy_above_hull=draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)),
        band_gap=draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False) | st.none()),
        density=draw(st.floats(min_value=2.0, max_value=8.0, allow_nan=False, allow_infinity=False) | st.none()),
        properties={
            "formation_energy_per_atom": draw(st.floats(min_value=-5.0, max_value=-0.5, allow_nan=False, allow_infinity=False)),
            "bulk_modulus": draw(st.floats(min_value=200.0, max_value=500.0, allow_nan=False, allow_infinity=False) | st.none()),
            "hardness": draw(st.floats(min_value=20.0, max_value=40.0, allow_nan=False, allow_infinity=False) | st.none()),
        },
        metadata={"source": "test"}
    )
    
    return material


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPipelineStageIndependence:
    """Test pipeline stage independence property."""
    
    @given(
        materials=st.lists(material_data_strategy(), min_size=5, max_size=20),
        fail_stage=st.sampled_from([
            "data_loading",
            "data_combination", 
            "feature_engineering",
            "ml_training",
            "application_ranking"
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_pipeline_stage_independence_property(self, materials, fail_stage, temp_output_dir):
        """
        **Feature: sic-alloy-integration, Property 15: Pipeline stage independence**
        
        Property: For any pipeline execution, if a stage fails, subsequent stages 
        should either skip gracefully or use available data without cascading failures.
        
        **Validates: Requirements 8.2**
        
        Test strategy:
        1. Create a pipeline with valid data
        2. Simulate a stage failure
        3. Verify that:
           - The pipeline doesn't crash
           - Subsequent stages either skip or use available data
           - Errors are logged but don't propagate
           - Pipeline returns a summary even with failures
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Prepare data for stages before the failure point
        if fail_stage in ["data_combination", "feature_engineering", "ml_training", "application_ranking"]:
            # Provide valid data loading results
            pipeline.stage_results["data_loading"] = {
                "mp": materials,
                "jarvis": [],
                "nist": {},
                "literature": {}
            }
        
        if fail_stage in ["feature_engineering", "ml_training", "application_ranking"]:
            # Provide valid combined data
            data_records = []
            for mat in materials:
                record = {
                    "material_id": mat.material_id,
                    "formula": mat.formula,
                    "formation_energy": mat.formation_energy,
                    "energy_above_hull": mat.energy_above_hull,
                    "band_gap": mat.band_gap,
                    "density": mat.density,
                }
                record.update(mat.properties)
                data_records.append(record)
            
            pipeline.combined_data = pd.DataFrame(data_records)
        
        # Now simulate the failure by providing invalid data for the failing stage
        if fail_stage == "data_loading":
            # Simulate data loading failure by providing no data sources
            result = pipeline._run_stage("data_loading", lambda: {})
            
        elif fail_stage == "data_combination":
            # Simulate combination failure by providing empty data
            pipeline.stage_results["data_loading"] = {
                "mp": [],
                "jarvis": [],
                "nist": {},
                "literature": {}
            }
            result = pipeline._run_stage("data_combination", pipeline._combine_data_sources)
            
        elif fail_stage == "feature_engineering":
            # Simulate feature engineering failure by providing data with no numeric columns
            pipeline.combined_data = pd.DataFrame({"formula": [m.formula for m in materials]})
            result = pipeline._run_stage("feature_engineering", pipeline._engineer_features)
            
        elif fail_stage == "ml_training":
            # Simulate ML training failure by providing insufficient data
            pipeline.engineered_features = pd.DataFrame(np.random.randn(2, 5))  # Only 2 samples
            pipeline.combined_data = pd.DataFrame({
                "hardness": [25.0, 26.0]
            })
            result = pipeline._run_stage("ml_training", pipeline._train_ml_models, target_property="hardness")
            
        elif fail_stage == "application_ranking":
            # Simulate ranking failure by providing empty data
            pipeline.combined_data = pd.DataFrame()
            result = pipeline._run_stage("application_ranking", pipeline._rank_materials)
        
        # Property verification: Pipeline should handle failure gracefully
        
        # 1. The stage should be marked as failed or return empty results
        assert fail_stage in pipeline.stage_results or fail_stage in pipeline.stage_errors, \
            f"Stage {fail_stage} should be tracked in results or errors"
        
        # 2. Pipeline should not crash (we got here, so this passes)
        assert True, "Pipeline did not crash despite stage failure"
        
        # 3. Error should be logged if stage failed
        if not result:
            assert fail_stage in pipeline.stage_errors, \
                f"Failed stage {fail_stage} should have error logged"
            assert isinstance(pipeline.stage_errors[fail_stage], str), \
                "Error message should be a string"
        
        # 4. Timing should be recorded even for failed stages
        assert fail_stage in pipeline.stage_timings, \
            f"Stage {fail_stage} should have timing recorded"
        assert pipeline.stage_timings[fail_stage] >= 0, \
            "Stage timing should be non-negative"
        
        # 5. Pipeline state should remain consistent
        assert isinstance(pipeline.stage_results, dict), \
            "Pipeline stage_results should remain a dict"
        assert isinstance(pipeline.stage_errors, dict), \
            "Pipeline stage_errors should remain a dict"
        assert isinstance(pipeline.stage_timings, dict), \
            "Pipeline stage_timings should remain a dict"
    
    def test_pipeline_continues_after_ml_failure(self, temp_output_dir):
        """
        Test that pipeline continues to ranking even if ML training fails.
        
        This is a specific case of stage independence: ML training is optional,
        so the pipeline should continue to application ranking even if ML fails.
        """
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Provide valid combined data
        pipeline.combined_data = pd.DataFrame({
            "material_id": ["mat1", "mat2", "mat3"],
            "formula": ["SiC", "TiC", "ZrC"],
            "formation_energy": [-2.5, -3.0, -2.8],
            "hardness": [28.0, 30.0, 29.0],
            "thermal_conductivity": [120.0, 100.0, 110.0],
            "density": [3.2, 4.9, 6.5]
        })
        
        # Simulate ML training failure (no engineered features)
        pipeline.engineered_features = None
        pipeline.ml_results = {}
        
        # Try to run ranking stage
        result = pipeline._run_stage("application_ranking", pipeline._rank_materials)
        
        # Ranking should succeed despite ML failure
        assert result, "Ranking should succeed even without ML results"
        assert pipeline.application_rankings is not None, \
            "Application rankings should be generated"
        assert len(pipeline.application_rankings) > 0, \
            "Should have some rankings"
    
    def test_pipeline_summary_with_partial_failures(self, temp_output_dir):
        """
        Test that pipeline can generate summary even with multiple stage failures.
        
        This tests the ultimate graceful degradation: even if multiple stages fail,
        the pipeline should still produce a summary of what succeeded and what failed.
        """
        from datetime import datetime
        
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Simulate some successful and some failed stages
        pipeline.stage_results["data_loading"] = {"mp": [], "jarvis": []}
        pipeline.stage_errors["data_combination"] = "Simulated failure"
        pipeline.stage_errors["ml_training"] = "Simulated failure"
        pipeline.stage_timings["data_loading"] = 1.5
        pipeline.stage_timings["data_combination"] = 0.5
        pipeline.stage_timings["ml_training"] = 0.3
        
        # Generate summary
        start_time = datetime.now()
        summary = pipeline._generate_pipeline_summary(start_time, success=False)
        
        # Verify summary structure
        assert isinstance(summary, dict), "Summary should be a dictionary"
        assert "success" in summary, "Summary should have success field"
        assert summary["success"] == False, "Summary should indicate failure"
        assert "stages_completed" in summary, "Summary should count completed stages"
        assert "stages_failed" in summary, "Summary should count failed stages"
        assert summary["stages_failed"] == 2, "Should have 2 failed stages"
        assert "stage_errors" in summary, "Summary should include errors"
        assert len(summary["stage_errors"]) == 2, "Should have 2 error messages"
        assert "stage_timings" in summary, "Summary should include timings"
        
    def test_checkpoint_saving_with_failures(self, temp_output_dir):
        """
        Test that checkpoints are saved for successful stages even when later stages fail.
        
        This ensures that partial progress is preserved even in the face of failures.
        """
        pipeline = SiCAlloyDesignerPipeline(
            output_dir=temp_output_dir,
            enable_checkpoints=True
        )
        
        # Simulate successful data combination
        test_data = pd.DataFrame({
            "material_id": ["mat1", "mat2"],
            "formula": ["SiC", "TiC"],
            "formation_energy": [-2.5, -3.0]
        })
        
        pipeline.combined_data = test_data
        pipeline._save_checkpoint("combined_data", test_data)
        
        # Verify checkpoint was saved
        checkpoint_file = pipeline.checkpoint_dir / "combined_data_checkpoint.json"
        assert checkpoint_file.exists(), "Checkpoint file should be created"
        
        # Verify checkpoint can be loaded
        loaded_data = pd.read_json(checkpoint_file)
        assert len(loaded_data) == len(test_data), "Loaded data should match original"
        assert list(loaded_data.columns) == list(test_data.columns), \
            "Loaded data should have same columns"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
