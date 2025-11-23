"""Tests for CLI functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ceramic_discovery.cli import main


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_materials_file(tmp_path):
    """Create temporary materials configuration file."""
    materials = {
        "base_systems": ["SiC", "B4C"],
        "dopants": ["Ti", "Zr"],
        "concentrations": [0.01, 0.05]
    }
    file_path = tmp_path / "materials.json"
    with open(file_path, "w") as f:
        json.dump(materials, f)
    return file_path


@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary data file."""
    file_path = tmp_path / "data.csv"
    file_path.write_text("property1,property2,v50\n1.0,2.0,500\n")
    return file_path


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Ceramic Armor Discovery Framework" in result.output
    
    def test_cli_version(self, cli_runner):
        """Test CLI version command."""
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    @patch("ceramic_discovery.cli.config")
    def test_init_command(self, mock_config, cli_runner):
        """Test init command."""
        mock_config.validate.return_value = None
        mock_config.hdf5.data_path = Path("./data/hdf5")
        mock_config.database.url = "postgresql://localhost/test"
        
        result = cli_runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert "initialized successfully" in result.output
    
    @patch("ceramic_discovery.cli.config")
    def test_status_command(self, mock_config, cli_runner):
        """Test status command."""
        mock_config.database.url = "postgresql://localhost/test"
        mock_config.redis.url = "redis://localhost"
        mock_config.hdf5.data_path = Path("./data")
        mock_config.computational.max_parallel_jobs = 4
        mock_config.ml.random_seed = 42
        mock_config.ml.cross_validation_folds = 5
        mock_config.ml.target_r2_min = 0.65
        mock_config.ml.target_r2_max = 0.75
        mock_config.ml.bootstrap_samples = 1000
        
        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Status" in result.output


class TestScreeningCommands:
    """Test screening-related commands."""
    
    @patch("ceramic_discovery.screening.screening_engine.ScreeningEngine")
    def test_screen_dopants_command(self, mock_engine_class, cli_runner, tmp_path):
        """Test dopant screening command."""
        mock_engine = MagicMock()
        mock_engine.screen_dopants.return_value = {
            "viable_count": 5,
            "results": []
        }
        mock_engine_class.return_value = mock_engine
        
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(main, [
            "screen", "dopants",
            "-b", "SiC",
            "-d", "Ti,Zr",
            "-c", "0.01,0.05",
            "-o", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Screening complete" in result.output
    
    @patch("ceramic_discovery.screening.workflow_orchestrator.WorkflowOrchestrator")
    def test_screen_batch_command(self, mock_orchestrator_class, cli_runner,
                                   temp_materials_file, tmp_path):
        """Test batch screening command."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_batch_screening.return_value = {
            "total_screened": 10
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(main, [
            "screen", "batch",
            "-m", str(temp_materials_file),
            "-o", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Batch screening complete" in result.output


class TestMLCommands:
    """Test machine learning commands."""
    
    @patch("ceramic_discovery.ml.model_trainer.ModelTrainer")
    def test_ml_train_command(self, mock_trainer_class, cli_runner,
                              temp_data_file, tmp_path):
        """Test ML training command."""
        mock_trainer = MagicMock()
        mock_trainer.train_from_file.return_value = {
            "r2_score": 0.72,
            "rmse": 25.5
        }
        mock_trainer_class.return_value = mock_trainer
        
        output_dir = tmp_path / "models"
        
        result = cli_runner.invoke(main, [
            "ml", "train",
            "-d", str(temp_data_file),
            "-t", "v50",
            "-m", "ensemble",
            "-o", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Training complete" in result.output
    
    @patch("ceramic_discovery.ballistics.ballistic_predictor.BallisticPredictor")
    def test_ml_predict_command(self, mock_predictor_class, cli_runner,
                                 temp_data_file, tmp_path):
        """Test ML prediction command."""
        mock_predictor = MagicMock()
        mock_predictor.predict_from_file.return_value = {
            "predictions": [500, 520]
        }
        mock_predictor_class.load_model.return_value = mock_predictor
        
        model_file = tmp_path / "model.pkl"
        model_file.write_text("mock model")
        output_dir = tmp_path / "predictions"
        
        result = cli_runner.invoke(main, [
            "ml", "predict",
            "-m", str(model_file),
            "-d", str(temp_data_file),
            "-o", str(output_dir),
            "--uncertainty"
        ])
        
        assert result.exit_code == 0
        assert "Predictions complete" in result.output


class TestAnalysisCommands:
    """Test analysis commands."""
    
    @patch("ceramic_discovery.analysis.property_analyzer.PropertyAnalyzer")
    def test_analyze_properties_command(self, mock_analyzer_class, cli_runner,
                                        temp_data_file, tmp_path):
        """Test property analysis command."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_from_file.return_value = {
            "plots_generated": 3
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        output_dir = tmp_path / "analysis"
        
        result = cli_runner.invoke(main, [
            "analyze", "properties",
            "-d", str(temp_data_file),
            "-o", str(output_dir),
            "--plot-types", "correlation,pca"
        ])
        
        assert result.exit_code == 0
        assert "Analysis complete" in result.output
    
    @patch("ceramic_discovery.ballistics.mechanism_analyzer.MechanismAnalyzer")
    def test_analyze_mechanisms_command(self, mock_analyzer_class, cli_runner,
                                        temp_data_file, tmp_path):
        """Test mechanism analysis command."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_from_file.return_value = {
            "key_drivers": ["hardness", "thermal_conductivity"]
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        output_dir = tmp_path / "mechanisms"
        
        result = cli_runner.invoke(main, [
            "analyze", "mechanisms",
            "-d", str(temp_data_file),
            "-o", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Mechanism analysis complete" in result.output


class TestExportCommands:
    """Test export commands."""
    
    @patch("ceramic_discovery.reporting.data_exporter.DataExporter")
    def test_export_results_command(self, mock_exporter_class, cli_runner,
                                     temp_data_file, tmp_path):
        """Test results export command."""
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        
        output_file = tmp_path / "results.csv"
        
        result = cli_runner.invoke(main, [
            "export", "results",
            "-d", str(temp_data_file),
            "-f", "csv",
            "-o", str(output_file),
            "--include-uncertainty"
        ])
        
        assert result.exit_code == 0
        assert "Export complete" in result.output
    
    @patch("ceramic_discovery.reporting.report_generator.ReportGenerator")
    def test_export_report_command(self, mock_generator_class, cli_runner,
                                    temp_data_file, tmp_path):
        """Test report generation command."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        output_file = tmp_path / "report.pdf"
        
        result = cli_runner.invoke(main, [
            "export", "report",
            "-d", str(temp_data_file),
            "-t", "full",
            "-o", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "Report generated" in result.output


class TestConfigurationCommands:
    """Test configuration commands."""
    
    def test_configure_development(self, cli_runner, tmp_path):
        """Test development configuration generation."""
        output_file = tmp_path / ".env.dev"
        
        result = cli_runner.invoke(main, [
            "configure",
            "-s", "development",
            "-o", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "DATABASE_URL" in content
        assert "development" in result.output
    
    def test_configure_hpc(self, cli_runner, tmp_path):
        """Test HPC configuration generation."""
        output_file = tmp_path / ".env.hpc"
        
        result = cli_runner.invoke(main, [
            "configure",
            "-s", "hpc",
            "-o", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "HPC_SCHEDULER" in content
