"""Tests for research output generation and reporting."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ceramic_discovery.reporting import ReportGenerator, ReportSection, DataExporter
from ceramic_discovery.validation.reproducibility import (
    ExperimentSnapshot,
    ExperimentParameters,
    ComputationalEnvironment,
    DataProvenance,
    SoftwareVersion
)


@pytest.fixture
def sample_data():
    """Create sample material data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'material_id': [f'MAT_{i:03d}' for i in range(50)],
        'base_composition': np.random.choice(['SiC', 'B4C', 'WC'], 50),
        'hardness': np.random.uniform(20, 35, 50),
        'fracture_toughness': np.random.uniform(3, 7, 50),
        'density': np.random.uniform(3.0, 6.0, 50),
        'thermal_conductivity': np.random.uniform(20, 100, 50),
        'v50_predicted': np.random.uniform(800, 1200, 50),
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestReportGenerator:
    """Tests for ReportGenerator."""
    
    def test_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator(
            title="Test Report",
            authors=["Author 1", "Author 2"],
            significance_level=0.05
        )
        
        assert generator.title == "Test Report"
        assert len(generator.authors) == 2
        assert generator.significance_level == 0.05
    
    def test_generate_summary_statistics(self, sample_data):
        """Test summary statistics generation."""
        generator = ReportGenerator(title="Test Report")
        
        section = generator.generate_summary_statistics(
            sample_data,
            title="Material Properties Summary"
        )
        
        assert section.title == "Material Properties Summary"
        assert len(section.tables) > 0
        assert 'property' in section.tables[0].columns
    
    def test_generate_comparative_analysis(self, sample_data):
        """Test comparative analysis generation."""
        generator = ReportGenerator(title="Test Report")
        
        section = generator.generate_comparative_analysis(
            sample_data,
            group_by='base_composition',
            properties=['hardness', 'fracture_toughness'],
            title="Composition Comparison"
        )
        
        assert section.title == "Composition Comparison"
        assert len(section.tables) > 0
    
    def test_generate_correlation_analysis(self, sample_data):
        """Test correlation analysis generation."""
        generator = ReportGenerator(title="Test Report")
        
        section = generator.generate_correlation_analysis(
            sample_data,
            properties=['hardness', 'fracture_toughness', 'density'],
            title="Property Correlations"
        )
        
        assert section.title == "Property Correlations"
        assert len(section.figures) > 0
    
    def test_save_html_report(self, sample_data, temp_output_dir):
        """Test HTML report generation."""
        generator = ReportGenerator(title="Test Report")
        
        # Add a section
        section = generator.generate_summary_statistics(sample_data)
        generator.add_section(section)
        
        # Save report
        report_path = generator.save_report(
            temp_output_dir,
            format='html',
            include_figures=False
        )
        
        assert report_path.exists()
        assert report_path.suffix == '.html'
        
        # Verify content
        content = report_path.read_text()
        assert "Test Report" in content


class TestDataExporter:
    """Tests for DataExporter."""
    
    def test_initialization(self, temp_output_dir):
        """Test data exporter initialization."""
        exporter = DataExporter(temp_output_dir)
        
        assert exporter.output_dir == temp_output_dir
        assert temp_output_dir.exists()
    
    def test_export_csv(self, sample_data, temp_output_dir):
        """Test CSV export."""
        exporter = DataExporter(temp_output_dir)
        
        output_path = exporter.export_dataframe(
            sample_data,
            filename="test_data",
            format='csv'
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        
        # Verify data
        loaded_data = pd.read_csv(output_path)
        assert len(loaded_data) == len(sample_data)
    
    def test_export_json(self, sample_data, temp_output_dir):
        """Test JSON export."""
        exporter = DataExporter(temp_output_dir)
        
        metadata = {'description': 'Test dataset', 'version': '1.0'}
        
        output_path = exporter.export_dataframe(
            sample_data,
            filename="test_data",
            format='json',
            metadata=metadata
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.json'
        
        # Verify data
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert 'data' in loaded
        assert 'metadata' in loaded
        assert loaded['metadata']['description'] == 'Test dataset'
    
    def test_export_excel(self, sample_data, temp_output_dir):
        """Test Excel export."""
        exporter = DataExporter(temp_output_dir)
        
        output_path = exporter.export_dataframe(
            sample_data,
            filename="test_data",
            format='excel'
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.xlsx'
        
        # Verify data
        loaded_data = pd.read_excel(output_path, sheet_name='Data')
        assert len(loaded_data) == len(sample_data)
    
    def test_anonymize_data(self, sample_data, temp_output_dir):
        """Test data anonymization."""
        exporter = DataExporter(temp_output_dir)
        
        # Anonymize by removing sensitive columns
        anonymized = exporter.anonymize_data(
            sample_data,
            sensitive_columns=['material_id'],
            method='remove'
        )
        
        assert 'material_id' not in anonymized.columns
        assert len(anonymized) == len(sample_data)
    
    def test_generate_data_dictionary(self, sample_data, temp_output_dir):
        """Test data dictionary generation."""
        exporter = DataExporter(temp_output_dir)
        
        column_descriptions = {
            'hardness': 'Material hardness in GPa',
            'fracture_toughness': 'Fracture toughness in MPa·m^0.5'
        }
        
        units = {
            'hardness': 'GPa',
            'fracture_toughness': 'MPa·m^0.5'
        }
        
        output_path = exporter.generate_data_dictionary(
            sample_data,
            column_descriptions=column_descriptions,
            units=units
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.xlsx'
    
    def test_create_reproducibility_package(self, temp_output_dir):
        """Test reproducibility package creation."""
        exporter = DataExporter(temp_output_dir)
        
        # Create mock experiment snapshot
        params = ExperimentParameters(
            experiment_id='test_exp_001',
            experiment_type='screening',
            parameters={'n_materials': 100},
            random_seed=42
        )
        
        env = ComputationalEnvironment(
            python_version='3.11.0',
            platform='Linux',
            platform_version='5.15.0',
            processor='x86_64',
            dependencies=[
                SoftwareVersion('numpy', '1.24.0'),
                SoftwareVersion('pandas', '2.0.0')
            ]
        )
        
        prov = DataProvenance(source_data='test_data.csv')
        prov.add_transformation('Filter stable materials', 'hash1')
        prov.finalize('final_hash')
        
        snapshot = ExperimentSnapshot(
            experiment_id='test_exp_001',
            parameters=params,
            environment=env,
            provenance=prov,
            results={'r2_score': 0.75}
        )
        
        # Create package
        package_path = exporter.create_reproducibility_package(
            experiment_id='test_exp_001',
            experiment_snapshot=snapshot
        )
        
        assert package_path.exists()
        assert package_path.suffix == '.zip'


class TestReportSection:
    """Tests for ReportSection."""
    
    def test_section_creation(self):
        """Test report section creation."""
        section = ReportSection(
            title="Test Section",
            content="This is test content.",
            level=1
        )
        
        assert section.title == "Test Section"
        assert section.content == "This is test content."
        assert section.level == 1
        assert len(section.figures) == 0
        assert len(section.tables) == 0
    
    def test_add_table(self, sample_data):
        """Test adding table to section."""
        section = ReportSection(
            title="Test Section",
            content="Content"
        )
        
        section.add_table(sample_data)
        
        assert len(section.tables) == 1
        assert isinstance(section.tables[0], pd.DataFrame)
    
    def test_add_subsection(self):
        """Test adding subsection."""
        parent = ReportSection(
            title="Parent Section",
            content="Parent content",
            level=1
        )
        
        child = ReportSection(
            title="Child Section",
            content="Child content"
        )
        
        parent.add_subsection(child)
        
        assert len(parent.subsections) == 1
        assert parent.subsections[0].level == 2


class TestSummaryReportGeneration:
    """Tests for generate_summary_report functionality."""
    
    def test_generate_summary_report_with_complete_data(self, sample_data):
        """Test report generation with complete data."""
        generator = ReportGenerator(title="Complete Data Report")
        
        # Create ML results
        ml_results = {
            'model_type': 'RandomForest',
            'r2_score': 0.85,
            'mae': 15.3,
            'rmse': 22.1,
            'n_features': 10,
            'n_samples': 100
        }
        
        # Create application rankings
        app_rankings = pd.DataFrame({
            'application': ['aerospace_hypersonic', 'cutting_tools'] * 5,
            'formula': ['SiC', 'TiC', 'B4C', 'WC', 'ZrC'] * 2,
            'rank': list(range(1, 11)),
            'overall_score': [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50],
            'confidence': [0.95] * 10
        })
        
        report = generator.generate_summary_report(
            data=sample_data,
            ml_results=ml_results,
            application_rankings=app_rankings,
            top_n=5
        )
        
        # Verify report structure
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Verify sections are present
        assert "DATASET STATISTICS" in report
        assert "MACHINE LEARNING MODEL PERFORMANCE" in report
        assert "TOP CANDIDATE RECOMMENDATIONS" in report
        
        # Verify dataset statistics
        assert f"Total materials: {len(sample_data)}" in report
        assert "Properties tracked:" in report
        
        # Verify ML results
        assert "RandomForest" in report
        assert "0.8500" in report  # R² score
        
        # Verify application rankings
        assert "aerospace_hypersonic" in report
        assert "cutting_tools" in report
    
    def test_generate_summary_report_with_incomplete_data(self):
        """Test report generation with incomplete data."""
        generator = ReportGenerator(title="Incomplete Data Report")
        
        # Only provide partial data
        partial_data = pd.DataFrame({
            'formula': ['SiC', 'TiC'],
            'hardness': [28.0, 30.0]
        })
        
        report = generator.generate_summary_report(
            data=partial_data,
            ml_results=None,
            application_rankings=None,
            top_n=5
        )
        
        # Verify report completes
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Verify missing data is indicated
        assert "No ML results provided" in report
        assert "No application rankings provided" in report
        
        # Verify partial data is reported
        assert "Total materials: 2" in report
    
    def test_generate_summary_report_with_empty_data(self):
        """Test report generation with empty data."""
        generator = ReportGenerator(title="Empty Data Report")
        
        report = generator.generate_summary_report(
            data=pd.DataFrame(),
            ml_results={},
            application_rankings=pd.DataFrame(),
            top_n=5
        )
        
        # Verify report completes without crashing
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Verify empty data is indicated
        assert "dataset is empty" in report or "No dataset provided" in report
        assert "No ML results provided" in report
        assert "No application rankings provided" in report
    
    def test_generate_summary_report_with_multiple_models(self):
        """Test report generation with multiple ML models."""
        generator = ReportGenerator(title="Multiple Models Report")
        
        ml_results = {
            'models': {
                'RandomForest': {
                    'r2_score': 0.85,
                    'mae': 15.3,
                    'rmse': 22.1,
                    'n_features': 10,
                    'n_samples': 100
                },
                'GradientBoosting': {
                    'r2_score': 0.88,
                    'mae': 13.5,
                    'rmse': 19.8,
                    'n_features': 10,
                    'n_samples': 100
                }
            }
        }
        
        report = generator.generate_summary_report(
            ml_results=ml_results,
            top_n=5
        )
        
        # Verify both models are reported
        assert "RandomForest" in report
        assert "GradientBoosting" in report
        assert "0.8500" in report
        assert "0.8800" in report
    
    def test_generate_summary_report_formatting(self, sample_data):
        """Test report formatting and structure."""
        generator = ReportGenerator(title="Formatting Test Report")
        
        report = generator.generate_summary_report(
            data=sample_data,
            top_n=5,
            title="Custom Report Title"
        )
        
        # Verify report has proper formatting
        assert "=" * 80 in report  # Header/footer separators
        assert "-" * 80 in report  # Section separators
        
        # Verify report has custom title
        assert "Custom Report Title" in report
        
        # Verify report has timestamp
        assert "Generated:" in report
        
        # Verify report has end marker
        assert "END OF REPORT" in report
    
    def test_generate_summary_report_with_nan_values(self):
        """Test report handles NaN values gracefully."""
        generator = ReportGenerator(title="NaN Test Report")
        
        # Create data with NaN values
        data_with_nan = pd.DataFrame({
            'formula': ['SiC', 'TiC', 'B4C'],
            'hardness': [28.0, np.nan, 32.0],
            'density': [np.nan, 4.5, 2.5]
        })
        
        ml_results = {
            'r2_score': np.nan,
            'mae': 15.3,
            'rmse': np.nan
        }
        
        report = generator.generate_summary_report(
            data=data_with_nan,
            ml_results=ml_results,
            top_n=5
        )
        
        # Verify report completes without crashing
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_generate_summary_report_property_statistics(self):
        """Test property statistics section in report."""
        generator = ReportGenerator(title="Property Stats Report")
        
        # Create data with known statistics
        test_data = pd.DataFrame({
            'hardness': [20.0, 25.0, 30.0, 35.0, 40.0],
            'density': [3.0, 3.5, 4.0, 4.5, 5.0]
        })
        
        report = generator.generate_summary_report(
            data=test_data,
            top_n=5
        )
        
        # Verify property statistics are included
        assert "Property Statistics:" in report
        assert "hardness" in report
        assert "density" in report
        
        # Verify statistics columns are present
        assert "Mean" in report
        assert "Std" in report
        assert "Min" in report
        assert "Max" in report
    
    def test_generate_summary_report_top_candidates(self):
        """Test top candidates section in report."""
        generator = ReportGenerator(title="Top Candidates Report")
        
        # Create application rankings with clear top candidates
        app_rankings = pd.DataFrame({
            'application': ['aerospace_hypersonic'] * 10,
            'formula': [f'Material_{i}' for i in range(10)],
            'rank': list(range(1, 11)),
            'overall_score': [1.0 - i*0.05 for i in range(10)],
            'confidence': [0.95] * 10
        })
        
        report = generator.generate_summary_report(
            application_rankings=app_rankings,
            top_n=3
        )
        
        # Verify top 3 candidates are shown
        assert "Material_0" in report
        assert "Material_1" in report
        assert "Material_2" in report
        
        # Verify ranking information
        assert "Rank" in report
        assert "Formula" in report
        assert "Score" in report
        assert "Confidence" in report
