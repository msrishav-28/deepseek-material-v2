"""Tests for Jupyter notebook widgets."""

import numpy as np
import pandas as pd
import pytest

from ceramic_discovery.utils.notebook_widgets import (
    MaterialExplorer,
    PropertyComparator,
    ScreeningConfigurator,
    MLModelTuner
)


@pytest.fixture
def sample_data():
    """Create sample material data for testing."""
    np.random.seed(42)
    n_samples = 20
    
    return pd.DataFrame({
        'composition': [f'SiC-Ti{i}%' for i in range(n_samples)],
        'energy_above_hull': np.random.uniform(0, 0.2, n_samples),
        'hardness': np.random.uniform(25, 35, n_samples),
        'fracture_toughness': np.random.uniform(3, 6, n_samples),
        'density': np.random.uniform(3.0, 3.5, n_samples),
        'thermal_conductivity_1000C': np.random.uniform(40, 80, n_samples),
        'v50': np.random.uniform(450, 550, n_samples)
    })


class TestMaterialExplorer:
    """Test MaterialExplorer widget."""
    
    def test_initialization(self, sample_data):
        """Test widget initialization."""
        explorer = MaterialExplorer(sample_data)
        
        assert explorer.data is not None
        assert len(explorer.data) == len(sample_data)
        assert explorer.x_axis is not None
        assert explorer.y_axis is not None
    
    def test_widget_creation(self, sample_data):
        """Test that all widgets are created."""
        explorer = MaterialExplorer(sample_data)
        
        assert explorer.x_axis is not None
        assert explorer.y_axis is not None
        assert explorer.color_by is not None
        assert explorer.stability_filter is not None
        assert explorer.plot_button is not None
        assert explorer.output is not None
    
    def test_data_filtering(self, sample_data):
        """Test data filtering by stability."""
        explorer = MaterialExplorer(sample_data)
        
        # Set filter
        explorer.stability_filter.value = 0.1
        explorer._update_plot(None)
        
        # Check filtered data
        assert len(explorer.filtered_data) <= len(sample_data)
        assert all(explorer.filtered_data['energy_above_hull'] <= 0.1)


class TestScreeningConfigurator:
    """Test ScreeningConfigurator widget."""
    
    def test_initialization(self):
        """Test widget initialization."""
        configurator = ScreeningConfigurator()
        
        assert configurator.base_systems is not None
        assert configurator.dopants is not None
        assert configurator.concentrations is not None
    
    def test_get_config(self):
        """Test configuration retrieval."""
        configurator = ScreeningConfigurator()
        
        config = configurator.get_config()
        
        assert 'base_systems' in config
        assert 'dopants' in config
        assert 'concentrations' in config
        assert 'stability_threshold' in config
        assert 'parallel_jobs' in config
    
    def test_config_values(self):
        """Test configuration values."""
        configurator = ScreeningConfigurator()
        
        # Set values
        configurator.stability_threshold.value = 0.15
        configurator.parallel_jobs.value = 8
        
        config = configurator.get_config()
        
        assert config['stability_threshold'] == 0.15
        assert config['parallel_jobs'] == 8
    
    def test_callback_execution(self):
        """Test callback execution."""
        callback_executed = {'value': False}
        
        def test_callback(config):
            callback_executed['value'] = True
        
        configurator = ScreeningConfigurator(callback=test_callback)
        configurator._on_submit(None)
        
        assert callback_executed['value'] is True


class TestPropertyComparator:
    """Test PropertyComparator widget."""
    
    def test_initialization(self, sample_data):
        """Test widget initialization."""
        comparator = PropertyComparator(sample_data)
        
        assert comparator.data is not None
        assert comparator.material1 is not None
        assert comparator.material2 is not None
        assert comparator.properties is not None
    
    def test_widget_creation(self, sample_data):
        """Test that all widgets are created."""
        comparator = PropertyComparator(sample_data)
        
        assert comparator.material1 is not None
        assert comparator.material2 is not None
        assert comparator.properties is not None
        assert comparator.compare_button is not None
        assert comparator.output is not None
    
    def test_material_selection(self, sample_data):
        """Test material selection."""
        comparator = PropertyComparator(sample_data)
        
        materials = sample_data['composition'].unique()
        assert len(comparator.material1.options) == len(materials)
        assert len(comparator.material2.options) == len(materials)


class TestMLModelTuner:
    """Test MLModelTuner widget."""
    
    def test_initialization(self):
        """Test widget initialization."""
        tuner = MLModelTuner()
        
        assert tuner.model_type is not None
        assert tuner.cv_folds is not None
        assert tuner.random_seed is not None
    
    def test_get_config(self):
        """Test configuration retrieval."""
        tuner = MLModelTuner()
        
        config = tuner.get_config()
        
        assert 'model_type' in config
        assert 'cv_folds' in config
        assert 'random_seed' in config
        assert 'target_r2' in config
        assert 'bootstrap_samples' in config
    
    def test_config_values(self):
        """Test configuration values."""
        tuner = MLModelTuner()
        
        # Set values
        tuner.cv_folds.value = 7
        tuner.random_seed.value = 123
        tuner.bootstrap_samples.value = 500
        
        config = tuner.get_config()
        
        assert config['cv_folds'] == 7
        assert config['random_seed'] == 123
        assert config['bootstrap_samples'] == 500
    
    def test_model_type_conversion(self):
        """Test model type string conversion."""
        tuner = MLModelTuner()
        
        tuner.model_type.value = 'Random Forest'
        config = tuner.get_config()
        
        assert config['model_type'] == 'random_forest'
    
    def test_callback_execution(self):
        """Test callback execution."""
        callback_executed = {'value': False}
        
        def test_callback(config):
            callback_executed['value'] = True
        
        tuner = MLModelTuner(callback=test_callback)
        tuner._on_train(None)
        
        assert callback_executed['value'] is True
