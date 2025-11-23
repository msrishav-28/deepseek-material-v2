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
