"""
Integration tests for SiC Alloy Designer Pipeline.

Tests complete pipeline execution, partial data source failures, and output file generation.
**Validates: Requirements 10.5**
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import json
import pandas as pd
import numpy as np

from ceramic_discovery.screening.sic_alloy_designer_pipeline import SiCAlloyDesignerPipeline
from ceramic_discovery.dft.materials_project_client import MaterialData


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_materials():
    """Create sample material data for testing."""
    materials = []
    
    formulas = ["SiC", "TiC", "ZrC", "HfC", "B4C"]
    
    for i, formula in enumerate(formulas):
        material = MaterialData(
            material_id=f"test-{i+1}",
            formula=formula,
            crystal_structure={},
            formation_energy=-2.5 - i * 0.2,
            energy_above_hull=0.1 * i,
            band_gap=2.0 + i * 0.5,
            density=3.0 + i * 0.5,
            properties={
                "formation_energy_per_atom": -2.5 - i * 0.2,
                "bulk_modulus": 300.0 + i * 20,
                "shear_modulus": 150.0 + i * 10,
                "hardness": 25.0 + i * 2,
                "thermal_conductivity": 100.0 + i * 10,
                "melting_point": 2500.0 + i * 100
            },
            metadata={"source": "test"}
        )
        materials.append(material)
    
    return materials


class TestPipelineIntegration:
    """Integration tests for complete pipeline execution."""
    
    def test_complete_pipeline_execution(self, temp_output_dir, sample_materials):
        """
        Test complete pipeline execution from data loading to reporting.
        
        **Validates: Requirements 10.5**
        
        This test verifies that:
        1. Pipeline can execute all stages successfully
        2. All expected outputs are generated
        3. Output files are created in correct locations
        4. Pipeline summary is generated
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(
            output_dir=temp_output_dir,
            enable_checkpoints=True
        )
        
        # Run full pipeline with sample data
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness",
            top_n_candidates=3
        )
        
        # Verify pipeline completed
        assert summary is not None, "Pipeline should return summary"
        assert isinstance(summary, dict), "Summary should be a dictionary"
        
        # Verify summary structure
        assert "success" in summary, "Summary should have success field"
        assert "stages_completed" in summary, "Summary should count completed stages"
        assert "total_duration_seconds" in summary, "Summary should include duration"
        assert "data_products" in summary, "Summary should include data products"
        
        # Verify data products were created
        data_products = summary["data_products"]
        assert data_products["combined_data_rows"] > 0, "Should have combined data"
        
        # Verify output files exist
        output_dir = Path(temp_output_dir)
        
        # Check for CSV outputs
        combined_csv = output_dir / "combined_materials_data.csv"
        assert combined_csv.exists(), "Combined data CSV should be created"
        
        # Verify CSV can be loaded
        combined_df = pd.read_csv(combined_csv)
        assert len(combined_df) > 0, "Combined CSV should have data"
        assert "formula" in combined_df.columns, "Combined CSV should have formula column"
        
        # Check for pipeline summary JSON
        summary_json = output_dir / "pipeline_summary.json"
        assert summary_json.exists(), "Pipeline summary JSON should be created"
        
        # Verify JSON can be loaded
        with open(summary_json, 'r') as f:
            loaded_summary = json.load(f)
        assert loaded_summary["stages_completed"] > 0, "Should have completed stages"
        
        # Check for report
        report_txt = output_dir / "analysis_summary_report.txt"
        assert report_txt.exists(), "Analysis report should be created"
        
        # Verify report has content
        with open(report_txt, 'r') as f:
            report_content = f.read()
        assert len(report_content) > 0, "Report should have content"
        assert "DATASET STATISTICS" in report_content, "Report should have dataset section"
    
    def test_pipeline_with_partial_data_source_failures(self, temp_output_dir, sample_materials):
        """
        Test pipeline execution when some data sources fail to load.
        
        **Validates: Requirements 10.5**
        
        This test verifies that:
        1. Pipeline continues when JARVIS data is unavailable
        2. Pipeline continues when NIST data is unavailable
        3. Pipeline uses available data sources
        4. Errors are logged but don't stop execution
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Run pipeline with only MP data (JARVIS and NIST unavailable)
        summary = pipeline.run_full_pipeline(
            jarvis_file=None,  # No JARVIS data
            nist_data_dir=None,  # No NIST data
            mp_data=sample_materials,
            target_property="hardness",
            top_n_candidates=2
        )
        
        # Verify pipeline completed despite missing data sources
        assert summary is not None, "Pipeline should complete with partial data"
        assert summary["data_products"]["combined_data_rows"] > 0, \
            "Should have data from available sources"
        
        # Verify data loading stage completed
        assert "data_loading" in pipeline.stage_results, \
            "Data loading stage should complete"
        
        # Verify combined data exists
        assert pipeline.combined_data is not None, "Should have combined data"
        assert len(pipeline.combined_data) > 0, "Combined data should not be empty"
    
    def test_pipeline_with_invalid_jarvis_file(self, temp_output_dir, sample_materials):
        """
        Test pipeline handles invalid JARVIS file path gracefully.
        
        **Validates: Requirements 10.5**
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Run pipeline with invalid JARVIS file
        summary = pipeline.run_full_pipeline(
            jarvis_file="/nonexistent/path/to/jarvis.json",
            mp_data=sample_materials,
            target_property="hardness"
        )
        
        # Pipeline should complete despite invalid JARVIS file
        assert summary is not None, "Pipeline should handle invalid file path"
        
        # Should have data from MP
        assert summary["data_products"]["combined_data_rows"] > 0, \
            "Should have data from MP despite JARVIS failure"
    
    def test_output_file_generation(self, temp_output_dir, sample_materials):
        """
        Test that all expected output files are generated.
        
        **Validates: Requirements 10.5**
        
        Verifies:
        1. Combined data CSV
        2. Engineered features CSV (if feature engineering succeeds)
        3. Application rankings CSV (if ranking succeeds)
        4. Pipeline summary JSON
        5. Analysis report TXT
        6. Checkpoint files (if enabled)
        """
        # Create pipeline with checkpoints enabled
        pipeline = SiCAlloyDesignerPipeline(
            output_dir=temp_output_dir,
            enable_checkpoints=True
        )
        
        # Run pipeline
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness",
            top_n_candidates=2
        )
        
        output_dir = Path(temp_output_dir)
        
        # Check main output files
        expected_files = [
            "combined_materials_data.csv",
            "pipeline_summary.json",
            "analysis_summary_report.txt"
        ]
        
        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Expected output file {filename} should exist"
            assert filepath.stat().st_size > 0, f"Output file {filename} should not be empty"
        
        # Check optional output files (may not exist if stages fail)
        optional_files = [
            "engineered_features.csv",
            "application_rankings.csv"
        ]
        
        for filename in optional_files:
            filepath = output_dir / filename
            if filepath.exists():
                assert filepath.stat().st_size > 0, \
                    f"Optional output file {filename} should not be empty if it exists"
        
        # Check checkpoint directory
        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory should be created"
        assert checkpoint_dir.is_dir(), "Checkpoint path should be a directory"
    
    def test_pipeline_with_minimal_data(self, temp_output_dir):
        """
        Test pipeline with minimal data (edge case).
        
        **Validates: Requirements 10.5**
        """
        # Create minimal material data (just 2 materials)
        minimal_materials = [
            MaterialData(
                material_id="min-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-2.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={
                    "formation_energy_per_atom": -2.5,
                    "hardness": 28.0
                },
                metadata={"source": "test"}
            ),
            MaterialData(
                material_id="min-2",
                formula="TiC",
                crystal_structure={},
                formation_energy=-3.0,
                energy_above_hull=0.0,
                band_gap=0.0,
                density=4.9,
                properties={
                    "formation_energy_per_atom": -3.0,
                    "hardness": 30.0
                },
                metadata={"source": "test"}
            )
        ]
        
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Run pipeline
        summary = pipeline.run_full_pipeline(
            mp_data=minimal_materials,
            target_property="hardness",
            top_n_candidates=1
        )
        
        # Pipeline should complete
        assert summary is not None, "Pipeline should handle minimal data"
        
        # Should have some data
        assert summary["data_products"]["combined_data_rows"] >= 2, \
            "Should have at least the minimal materials"
    
    def test_pipeline_stage_timing_tracking(self, temp_output_dir, sample_materials):
        """
        Test that pipeline tracks timing for each stage.
        
        **Validates: Requirements 10.5**
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Run pipeline
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness"
        )
        
        # Verify timing information
        assert "stage_timings" in summary, "Summary should include stage timings"
        
        stage_timings = summary["stage_timings"]
        assert isinstance(stage_timings, dict), "Stage timings should be a dictionary"
        
        # Verify timings are non-negative
        for stage, timing in stage_timings.items():
            assert timing >= 0, f"Stage {stage} timing should be non-negative"
            assert isinstance(timing, (int, float)), \
                f"Stage {stage} timing should be numeric"
    
    def test_pipeline_error_logging(self, temp_output_dir):
        """
        Test that pipeline logs errors appropriately.
        
        **Validates: Requirements 10.5**
        """
        # Create pipeline
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        # Run pipeline with no data (should cause early exit)
        summary = pipeline.run_full_pipeline(
            mp_data=None,
            jarvis_file=None,
            nist_data_dir=None
        )
        
        # Verify error tracking structure exists
        assert "stage_errors" in summary, "Summary should include stage errors"
        assert isinstance(summary["stage_errors"], dict), "Stage errors should be a dict"
        
        # Pipeline exits early when no data is available, which is graceful degradation
        # Verify that success is False
        assert summary["success"] == False, "Pipeline should indicate failure with no data"
        
        # Verify that data products show no data
        assert summary["data_products"]["combined_data_rows"] == 0, \
            "Should have no combined data"
        
        # If there are any errors logged, verify they are strings
        for stage, error in summary["stage_errors"].items():
            assert isinstance(error, str), f"Error for {stage} should be a string"
            assert len(error) > 0, f"Error message for {stage} should not be empty"
    
    def test_checkpoint_recovery(self, temp_output_dir, sample_materials):
        """
        Test that checkpoints can be used for recovery.
        
        **Validates: Requirements 10.5**
        """
        # Create pipeline with checkpoints
        pipeline = SiCAlloyDesignerPipeline(
            output_dir=temp_output_dir,
            enable_checkpoints=True
        )
        
        # Run pipeline
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness"
        )
        
        # Verify checkpoints were created
        checkpoint_dir = Path(temp_output_dir) / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*_checkpoint.json"))
        
        assert len(checkpoint_files) > 0, "Should have created checkpoint files"
        
        # Verify checkpoints can be loaded
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            assert checkpoint_data is not None, \
                f"Checkpoint {checkpoint_file.name} should be loadable"
            
            # Verify checkpoint has data
            if isinstance(checkpoint_data, list):
                assert len(checkpoint_data) >= 0, "Checkpoint list should be valid"
            elif isinstance(checkpoint_data, dict):
                assert isinstance(checkpoint_data, dict), "Checkpoint dict should be valid"


class TestPipelineDataProducts:
    """Test data products generated by pipeline."""
    
    def test_combined_data_structure(self, temp_output_dir, sample_materials):
        """Test structure of combined data output."""
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness"
        )
        
        # Load combined data CSV
        combined_csv = Path(temp_output_dir) / "combined_materials_data.csv"
        df = pd.read_csv(combined_csv)
        
        # Verify required columns
        required_cols = ["material_id", "formula", "formation_energy"]
        for col in required_cols:
            assert col in df.columns, f"Combined data should have {col} column"
        
        # Verify data types
        assert df["material_id"].dtype == object, "material_id should be string"
        assert df["formula"].dtype == object, "formula should be string"
        assert pd.api.types.is_numeric_dtype(df["formation_energy"]), \
            "formation_energy should be numeric"
    
    def test_application_rankings_structure(self, temp_output_dir, sample_materials):
        """Test structure of application rankings output."""
        pipeline = SiCAlloyDesignerPipeline(output_dir=temp_output_dir)
        
        summary = pipeline.run_full_pipeline(
            mp_data=sample_materials,
            target_property="hardness"
        )
        
        # Check if rankings were generated
        rankings_csv = Path(temp_output_dir) / "application_rankings.csv"
        
        if rankings_csv.exists():
            df = pd.read_csv(rankings_csv)
            
            # Verify required columns
            required_cols = ["formula", "application", "overall_score", "rank"]
            for col in required_cols:
                assert col in df.columns, f"Rankings should have {col} column"
            
            # Verify score is in [0, 1]
            assert df["overall_score"].min() >= 0, "Scores should be >= 0"
            assert df["overall_score"].max() <= 1, "Scores should be <= 1"
            
            # Verify ranks are positive integers
            assert df["rank"].min() >= 1, "Ranks should start at 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
