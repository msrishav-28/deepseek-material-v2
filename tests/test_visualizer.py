"""Tests for research visualization system."""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from ceramic_discovery.analysis.visualizer import MaterialsVisualizer


class TestMaterialsVisualizer:
    """Test MaterialsVisualizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample material property data."""
        np.random.seed(42)
        n_samples = 50
        
        data = pd.DataFrame({
            'material_id': [f'mat_{i}' for i in range(n_samples)],
            'base_composition': np.random.choice(['SiC', 'B4C', 'WC'], n_samples),
            'hardness': np.random.normal(28.0, 3.0, n_samples),
            'fracture_toughness': np.random.normal(4.5, 0.8, n_samples),
            'density': np.random.normal(3.2, 0.3, n_samples),
            'thermal_conductivity': np.random.normal(50.0, 10.0, n_samples),
            'dopant_concentration': np.repeat([0.01, 0.02, 0.03, 0.05, 0.10], 10),
            'predicted_v50': np.random.normal(1500, 100, n_samples),
            'actual_v50': np.random.normal(1500, 100, n_samples),
        })
        
        # Add uncertainty columns
        data['v50_uncertainty'] = np.random.uniform(50, 150, n_samples)
        
        return data
    
    @pytest.fixture
    def learning_curves_data(self):
        """Create sample learning curves data."""
        return {
            'train_scores': [0.5, 0.6, 0.65, 0.68, 0.70, 0.71, 0.72],
            'val_scores': [0.48, 0.58, 0.62, 0.64, 0.65, 0.65, 0.64],
        }
    
    @pytest.fixture
    def feature_importance_data(self):
        """Create sample feature importance data."""
        return pd.DataFrame({
            'feature': ['hardness', 'thermal_conductivity', 'density', 'fracture_toughness'],
            'importance': [0.35, 0.30, 0.20, 0.15],
        })
    
    def test_visualizer_initialization(self):
        """Test MaterialsVisualizer initialization."""
        viz = MaterialsVisualizer(theme='plotly_white')
        
        assert viz.theme == 'plotly_white'
    
    def test_create_parallel_coordinates(self, sample_data):
        """Test parallel coordinates plot creation."""
        viz = MaterialsVisualizer()
        
        properties = ['hardness', 'fracture_toughness', 'density', 'thermal_conductivity']
        fig = viz.create_parallel_coordinates(sample_data, properties)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_parallel_coordinates_with_color(self, sample_data):
        """Test parallel coordinates with color mapping."""
        viz = MaterialsVisualizer()
        
        properties = ['hardness', 'fracture_toughness', 'density']
        fig = viz.create_parallel_coordinates(
            sample_data, 
            properties, 
            color_by='thermal_conductivity'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_parallel_coordinates_missing_property(self, sample_data):
        """Test error handling for missing properties."""
        viz = MaterialsVisualizer()
        
        properties = ['hardness', 'nonexistent_property']
        
        with pytest.raises(ValueError, match="not found in data"):
            viz.create_parallel_coordinates(sample_data, properties)
    
    def test_create_structure_property_plot(self, sample_data):
        """Test structure-property scatter plot creation."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_structure_property_plot(
            sample_data,
            x_property='hardness',
            y_property='fracture_toughness'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_structure_property_plot_with_color(self, sample_data):
        """Test structure-property plot with color mapping."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_structure_property_plot(
            sample_data,
            x_property='hardness',
            y_property='fracture_toughness',
            color_by='base_composition'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_structure_property_plot_with_confidence(self, sample_data):
        """Test structure-property plot with confidence intervals."""
        viz = MaterialsVisualizer()
        
        # Add uncertainty columns
        sample_data['hardness_uncertainty'] = 0.5
        sample_data['toughness_uncertainty'] = 0.1
        
        fig = viz.create_structure_property_plot(
            sample_data,
            x_property='hardness',
            y_property='fracture_toughness',
            show_confidence=True,
            x_uncertainty='hardness_uncertainty',
            y_uncertainty='toughness_uncertainty'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_ml_performance_dashboard(self, learning_curves_data, feature_importance_data):
        """Test ML performance dashboard creation."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_ml_performance_dashboard(
            learning_curves_data,
            feature_importance_data
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least learning curves and feature importance
    
    def test_create_ml_performance_dashboard_with_residuals(
        self, 
        learning_curves_data, 
        feature_importance_data
    ):
        """Test ML dashboard with residual plot."""
        viz = MaterialsVisualizer()
        
        residuals = pd.DataFrame({
            'predicted': np.random.normal(1500, 100, 50),
            'residual': np.random.normal(0, 50, 50),
        })
        
        fig = viz.create_ml_performance_dashboard(
            learning_curves_data,
            feature_importance_data,
            residuals=residuals
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Learning curves, feature importance, and residuals
    
    def test_create_parity_plot(self, sample_data):
        """Test parity plot creation."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_parity_plot(
            sample_data,
            predicted_col='predicted_v50',
            actual_col='actual_v50'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Scatter and perfect prediction line
    
    def test_create_parity_plot_with_confidence(self, sample_data):
        """Test parity plot with confidence intervals."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_parity_plot(
            sample_data,
            predicted_col='predicted_v50',
            actual_col='actual_v50',
            show_confidence=True,
            uncertainty_col='v50_uncertainty'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_parity_plot_missing_columns(self, sample_data):
        """Test error handling for missing columns."""
        viz = MaterialsVisualizer()
        
        with pytest.raises(ValueError, match="not found in data"):
            viz.create_parity_plot(
                sample_data,
                predicted_col='nonexistent',
                actual_col='actual_v50'
            )
    
    def test_create_concentration_trend_plot(self, sample_data):
        """Test concentration-dependent trend plot."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_concentration_trend_plot(
            sample_data,
            property_name='hardness',
            concentration_col='dopant_concentration'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    
    def test_create_concentration_trend_plot_grouped(self, sample_data):
        """Test concentration trend plot with grouping."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_concentration_trend_plot(
            sample_data,
            property_name='hardness',
            concentration_col='dopant_concentration',
            group_by='base_composition'
        )
        
        assert isinstance(fig, go.Figure)
        # Should have multiple traces for different groups
        assert len(fig.data) >= 3
    
    def test_create_concentration_trend_plot_with_confidence(self, sample_data):
        """Test concentration trend plot with confidence intervals."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_concentration_trend_plot(
            sample_data,
            property_name='hardness',
            concentration_col='dopant_concentration',
            group_by='base_composition',
            show_confidence=True
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_property_distribution_histogram(self, sample_data):
        """Test property distribution histogram."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_property_distribution_plot(
            sample_data,
            property_name='hardness',
            plot_type='histogram'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_property_distribution_box(self, sample_data):
        """Test property distribution box plot."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_property_distribution_plot(
            sample_data,
            property_name='hardness',
            plot_type='box'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_property_distribution_violin(self, sample_data):
        """Test property distribution violin plot."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_property_distribution_plot(
            sample_data,
            property_name='hardness',
            plot_type='violin'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_property_distribution_grouped(self, sample_data):
        """Test grouped property distribution."""
        viz = MaterialsVisualizer()
        
        fig = viz.create_property_distribution_plot(
            sample_data,
            property_name='hardness',
            group_by='base_composition',
            plot_type='box'
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_create_property_distribution_invalid_type(self, sample_data):
        """Test error handling for invalid plot type."""
        viz = MaterialsVisualizer()
        
        with pytest.raises(ValueError, match="Unknown plot type"):
            viz.create_property_distribution_plot(
                sample_data,
                property_name='hardness',
                plot_type='invalid_type'
            )
    
    def test_create_correlation_heatmap(self, sample_data):
        """Test correlation heatmap creation."""
        viz = MaterialsVisualizer()
        
        # Create correlation matrix
        numeric_cols = ['hardness', 'fracture_toughness', 'density', 'thermal_conductivity']
        corr_matrix = sample_data[numeric_cols].corr()
        
        fig = viz.create_correlation_heatmap(corr_matrix)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
