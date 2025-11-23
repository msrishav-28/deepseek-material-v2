"""Tests for CI/CD configuration."""

import pytest
import yaml
from pathlib import Path


class TestCIConfiguration:
    """Test CI/CD pipeline configuration."""

    def test_github_actions_workflow_exists(self):
        """Test that GitHub Actions workflow file exists."""
        workflow_file = Path(".github/workflows/ci.yml")
        assert workflow_file.exists(), "CI workflow file not found"

    def test_github_actions_workflow_valid_yaml(self):
        """Test that GitHub Actions workflow is valid YAML."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in CI workflow: {e}")

        assert config is not None
        assert "name" in config
        # "on" is parsed as True by YAML
        assert True in config or "on" in config
        assert "jobs" in config

    def test_ci_workflow_has_required_jobs(self):
        """Test that CI workflow has all required jobs."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        required_jobs = [
            "test",
            "integration-tests",
            "scientific-validation",
            "reproducibility-tests",
            "code-quality",
        ]

        jobs = config.get("jobs", {})
        
        for job in required_jobs:
            assert job in jobs, f"Required job '{job}' not found in CI workflow"

    def test_ci_workflow_tests_multiple_python_versions(self):
        """Test that CI tests multiple Python versions."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        test_job = config["jobs"]["test"]
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        
        python_versions = matrix.get("python-version", [])
        
        assert len(python_versions) >= 2, "Should test at least 2 Python versions"
        assert "3.11" in python_versions, "Should test Python 3.11"

    def test_ci_workflow_tests_multiple_os(self):
        """Test that CI tests multiple operating systems."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        test_job = config["jobs"]["test"]
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        
        os_list = matrix.get("os", [])
        
        assert len(os_list) >= 2, "Should test at least 2 operating systems"

    def test_pytest_configuration_exists(self):
        """Test that pytest configuration exists."""
        pytest_ini = Path("pytest.ini")
        assert pytest_ini.exists(), "pytest.ini not found"

    def test_pytest_configuration_valid(self):
        """Test that pytest configuration is valid."""
        pytest_ini = Path("pytest.ini")
        
        with open(pytest_ini) as f:
            content = f.read()

        # Check for required sections
        assert "[pytest]" in content
        assert "testpaths" in content
        assert "markers" in content

    def test_pytest_markers_defined(self):
        """Test that all required pytest markers are defined."""
        pytest_ini = Path("pytest.ini")
        
        with open(pytest_ini) as f:
            content = f.read()

        required_markers = [
            "integration",
            "performance",
            "scientific",
            "reproducibility",
        ]

        for marker in required_markers:
            assert marker in content, f"Marker '{marker}' not defined in pytest.ini"

    def test_coverage_configuration_exists(self):
        """Test that coverage configuration exists."""
        coveragerc = Path(".coveragerc")
        assert coveragerc.exists(), "Coverage configuration not found"

    def test_coverage_configuration_valid(self):
        """Test that coverage configuration is valid."""
        coveragerc = Path(".coveragerc")
        
        with open(coveragerc) as f:
            content = f.read()

        # Check for required sections
        assert "[run]" in content
        assert "[report]" in content
        assert "source = ceramic_discovery" in content

    def test_makefile_has_test_targets(self):
        """Test that Makefile has test targets."""
        makefile = Path("Makefile")
        assert makefile.exists(), "Makefile not found"
        
        with open(makefile) as f:
            content = f.read()

        required_targets = [
            "test",
            "test-unit",
            "test-integration",
            "test-scientific",
            "coverage",
        ]

        for target in required_targets:
            assert f"{target}:" in content, f"Target '{target}' not found in Makefile"


class TestTestDataManagement:
    """Test test data management system."""

    def test_test_data_directory_exists(self):
        """Test that test data directory exists."""
        test_data_dir = Path("tests/data")
        assert test_data_dir.exists(), "Test data directory not found"

    def test_test_data_readme_exists(self):
        """Test that test data README exists."""
        readme = Path("tests/data/README.md")
        assert readme.exists(), "Test data README not found"

    def test_baseline_materials_data_exists(self):
        """Test that baseline materials data exists."""
        baseline_data = Path("tests/data/materials/baseline_ceramics.json")
        assert baseline_data.exists(), "Baseline materials data not found"

    def test_test_data_validator_exists(self):
        """Test that test data validator exists."""
        validator = Path("tests/utils/validate_test_data.py")
        assert validator.exists(), "Test data validator not found"

    def test_test_utils_module_exists(self):
        """Test that test utils module exists."""
        utils_init = Path("tests/utils/__init__.py")
        assert utils_init.exists(), "Test utils module not found"

    def test_can_import_test_utils(self):
        """Test that test utils can be imported."""
        try:
            from tests.utils import load_test_data, get_baseline_materials
        except ImportError as e:
            pytest.fail(f"Cannot import test utils: {e}")

    def test_can_load_baseline_materials(self):
        """Test that baseline materials can be loaded."""
        from tests.utils import get_baseline_materials

        materials = get_baseline_materials()
        
        assert len(materials) > 0, "No baseline materials loaded"
        assert all("material_id" in m for m in materials)
        assert all("formula" in m for m in materials)


class TestCIIntegration:
    """Test CI integration features."""

    def test_ci_runs_scientific_validation(self):
        """Test that CI includes scientific validation."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        assert "scientific-validation" in config["jobs"]
        
        job = config["jobs"]["scientific-validation"]
        steps = job.get("steps", [])
        
        # Check for scientific test execution
        step_names = [step.get("name", "") for step in steps]
        assert any("scientific" in name.lower() for name in step_names)

    def test_ci_runs_reproducibility_tests(self):
        """Test that CI includes reproducibility tests."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        assert "reproducibility-tests" in config["jobs"]

    def test_ci_validates_test_data(self):
        """Test that CI validates test data."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        assert "test-data-validation" in config["jobs"]

    def test_ci_checks_code_quality(self):
        """Test that CI checks code quality."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            config = yaml.safe_load(f)

        assert "code-quality" in config["jobs"]
        
        job = config["jobs"]["code-quality"]
        steps = job.get("steps", [])
        
        # Check for linting steps
        step_names = [step.get("name", "") for step in steps]
        assert any("black" in name.lower() for name in step_names)
        assert any("flake8" in name.lower() for name in step_names)

    def test_ci_uploads_coverage(self):
        """Test that CI uploads coverage reports."""
        workflow_file = Path(".github/workflows/ci.yml")
        
        with open(workflow_file) as f:
            content = f.read()

        # Check for codecov upload
        assert "codecov" in content.lower()
