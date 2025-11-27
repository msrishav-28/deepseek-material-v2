"""Tests for integration example scripts."""

import pytest
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd


class TestIntegrationExamples:
    """Test suite for integration example scripts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.examples_dir = Path("examples")
        self.results_dir = Path("results")
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def test_jarvis_data_loading_example_executes(self):
        """Test that JARVIS data loading example executes successfully."""
        script_path = self.examples_dir / "jarvis_data_loading_example.py"
        
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check expected output messages
        assert "JARVIS-DFT Data Loading Example" in result.stdout
        assert "JARVIS client initialized" in result.stdout
        assert "carbide materials" in result.stdout.lower()
        assert "completed successfully" in result.stdout.lower()
        
        # Check output file was created
        output_file = self.results_dir / "jarvis_carbides.csv"
        assert output_file.exists(), "Output CSV file not created"
        
        # Validate output file
        df = pd.read_csv(output_file)
        assert len(df) > 0, "Output CSV is empty"
        assert 'formula' in df.columns, "Missing 'formula' column"
    
    def test_multi_source_combination_example_executes(self):
        """Test that multi-source data combination example executes successfully."""
        script_path = self.examples_dir / "multi_source_data_combination_example.py"
        
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check expected output messages
        assert "Multi-Source Data Combination Example" in result.stdout
        assert "Loading JARVIS-DFT data" in result.stdout
        assert "Loading literature database" in result.stdout
        assert "Combining data from all sources" in result.stdout
        assert "completed successfully" in result.stdout.lower()
        
        # Check output file was created
        output_file = self.results_dir / "combined_materials_data.csv"
        assert output_file.exists(), "Output CSV file not created"
        
        # Validate output file
        df = pd.read_csv(output_file)
        assert len(df) > 0, "Output CSV is empty"
        assert 'formula' in df.columns, "Missing 'formula' column"
    
    def test_application_ranking_example_executes(self):
        """Test that application ranking example executes successfully."""
        script_path = self.examples_dir / "application_ranking_example.py"
        
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check expected output messages
        assert "Application-Specific Ranking Example" in result.stdout
        assert "Application ranker initialized" in result.stdout
        assert "Ranking for Aerospace" in result.stdout
        assert "Ranking for Cutting Tools" in result.stdout
        assert "completed successfully" in result.stdout.lower()
        
        # Check output file was created
        output_file = self.results_dir / "application_rankings.csv"
        assert output_file.exists(), "Output CSV file not created"
        
        # Validate output file
        df = pd.read_csv(output_file)
        assert len(df) > 0, "Output CSV is empty"
        assert 'formula' in df.columns, "Missing 'formula' column"
        
        # Check for application score columns
        score_columns = [col for col in df.columns if col.endswith('_score')]
        assert len(score_columns) > 0, "No application score columns found"
    
    def test_experimental_planning_example_executes(self):
        """Test that experimental planning example executes successfully."""
        script_path = self.examples_dir / "experimental_planning_example.py"
        
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check expected output messages
        assert "Experimental Planning Example" in result.stdout
        assert "Experimental planner initialized" in result.stdout
        assert "Designing synthesis protocols" in result.stdout
        assert "Designing characterization plans" in result.stdout
        assert "Estimating resources" in result.stdout
        assert "completed successfully" in result.stdout.lower()
        
        # Check output files were created
        output_dir = self.results_dir / "experimental_protocols"
        assert output_dir.exists(), "Output directory not created"
        
        comparison_file = output_dir / "candidate_comparison.csv"
        assert comparison_file.exists(), "Comparison CSV not created"
        
        # Validate comparison file
        df = pd.read_csv(comparison_file)
        assert len(df) > 0, "Comparison CSV is empty"
        assert 'formula' in df.columns, "Missing 'formula' column"
        assert 'timeline_months' in df.columns, "Missing 'timeline_months' column"
        assert 'cost_k_dollars' in df.columns, "Missing 'cost_k_dollars' column"
    
    def test_workflow_notebook_creation(self):
        """Test that workflow notebook creation script executes successfully."""
        script_path = self.examples_dir / "create_workflow_notebook.py"
        
        assert script_path.exists(), f"Script not found: {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check execution was successful
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check notebook was created
        notebook_path = Path("notebooks/06_sic_alloy_integration_workflow.ipynb")
        assert notebook_path.exists(), "Notebook file not created"
        
        # Validate notebook structure
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        assert 'cells' in notebook, "Notebook missing 'cells' key"
        assert len(notebook['cells']) > 0, "Notebook has no cells"
        assert 'metadata' in notebook, "Notebook missing 'metadata' key"
        assert notebook['nbformat'] == 4, "Incorrect notebook format version"
        
        # Check for key sections
        cell_sources = [
            ''.join(cell.get('source', [])) 
            for cell in notebook['cells']
        ]
        all_content = ' '.join(cell_sources)
        
        assert 'SiC Alloy Designer Integration Workflow' in all_content
        assert 'Multi-source data loading' in all_content
        assert 'Application-specific ranking' in all_content
        assert 'Experimental planning' in all_content
    
    def test_all_examples_produce_valid_output(self):
        """Test that all examples produce valid, non-empty output."""
        example_scripts = [
            "jarvis_data_loading_example.py",
            "multi_source_data_combination_example.py",
            "application_ranking_example.py",
            "experimental_planning_example.py",
        ]
        
        for script_name in example_scripts:
            script_path = self.examples_dir / script_name
            
            if not script_path.exists():
                pytest.skip(f"Script not found: {script_path}")
            
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check execution was successful
            assert result.returncode == 0, (
                f"Script {script_name} failed with error: {result.stderr}"
            )
            
            # Check output is not empty
            assert len(result.stdout) > 0, f"Script {script_name} produced no output"
            
            # Check for success indicators
            assert (
                "completed successfully" in result.stdout.lower() or
                "✓" in result.stdout
            ), f"Script {script_name} did not indicate successful completion"
    
    def test_example_outputs_are_consistent(self):
        """Test that running examples multiple times produces consistent results."""
        script_path = self.examples_dir / "application_ranking_example.py"
        
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")
        
        # Run the script twice
        results = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            assert result.returncode == 0
            results.append(result.stdout)
        
        # Check outputs are identical (deterministic)
        # Note: Some variation is acceptable due to timestamps, but key results should match
        output_file = self.results_dir / "application_rankings.csv"
        
        # Read the output file after first run
        df1 = pd.read_csv(output_file)
        
        # Run again
        subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Read the output file after second run
        df2 = pd.read_csv(output_file)
        
        # Check that the data is consistent
        assert len(df1) == len(df2), "Different number of materials in outputs"
        assert list(df1.columns) == list(df2.columns), "Different columns in outputs"
        
        # Check that scores are identical (within floating point tolerance)
        score_columns = [col for col in df1.columns if col.endswith('_score')]
        for col in score_columns:
            if col in df1.columns and col in df2.columns:
                pd.testing.assert_series_equal(
                    df1[col], 
                    df2[col], 
                    check_names=False,
                    rtol=1e-5
                )


class TestExampleOutputValidation:
    """Test suite for validating example outputs."""
    
    def test_jarvis_output_has_required_properties(self):
        """Test that JARVIS example output contains required properties."""
        output_file = Path("results/jarvis_carbides.csv")
        
        if not output_file.exists():
            pytest.skip("JARVIS output file not found")
        
        df = pd.read_csv(output_file)
        
        # Check required columns
        required_columns = ['formula', 'formation_energy_peratom']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types
        assert df['formula'].dtype == object, "Formula should be string type"
        
        # Check for valid values
        assert df['formula'].notna().all(), "Some formulas are missing"
    
    def test_application_rankings_have_valid_scores(self):
        """Test that application rankings have valid score values."""
        output_file = Path("results/application_rankings.csv")
        
        if not output_file.exists():
            pytest.skip("Application rankings file not found")
        
        df = pd.read_csv(output_file)
        
        # Find score columns
        score_columns = [col for col in df.columns if col.endswith('_score')]
        
        assert len(score_columns) > 0, "No score columns found"
        
        # Check that all scores are in valid range [0, 1]
        for col in score_columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                assert scores.min() >= 0, f"Score below 0 in {col}"
                assert scores.max() <= 1, f"Score above 1 in {col}"
    
    def test_experimental_protocols_have_positive_estimates(self):
        """Test that experimental protocols have positive resource estimates."""
        output_file = Path("results/experimental_protocols/candidate_comparison.csv")
        
        if not output_file.exists():
            pytest.skip("Experimental protocols file not found")
        
        df = pd.read_csv(output_file)
        
        # Check timeline is positive
        if 'timeline_months' in df.columns:
            timelines = df['timeline_months'].dropna()
            assert (timelines > 0).all(), "Some timelines are not positive"
        
        # Check cost is positive
        if 'cost_k_dollars' in df.columns:
            costs = df['cost_k_dollars'].dropna()
            assert (costs > 0).all(), "Some costs are not positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
