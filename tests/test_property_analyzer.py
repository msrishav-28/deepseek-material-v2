"""Tests for materials property analysis toolkit."""

import numpy as np
import pandas as pd
import pytest

from ceramic_discovery.analysis.property_analyzer import (
    PropertyAnalyzer,
    PropertyStatistics,
    CorrelationAnalysis,
    TrendAnalysis,
    OutlierDetection,
)


class TestPropertyStatistics:
    """Test PropertyStatistics dataclass."""
    
    def test_property_statistics_creation(self):
        """Test creation of PropertyStatistics."""
        stats = PropertyStatistics(
            property_name='hardness',
            mean=28.5,
            median=28.0,
            std=2.5,
            min=24.0,
            max=33.0,
            q25=26.5,
            q75=30.5,
            skewness=0.1,
            kurtosis=-0.5,
            n_samples=100,
        )
        
        assert stats.property_name == 'hardness'
        assert stats.mean == 28.5
        assert stats.n_samples == 100
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PropertyStatistics(
            property_name='density',
            mean=3.2,
            median=3.1,
            std=0.3,
            min=2.8,
            max=3.8,
            q25=3.0,
            q75=3.4,
            skewness=0.2,
            kurtosis=0.1,
            n_samples=50,
        )
        
        stats_dict = stats.to_dict()
        assert stats_dict['property'] == 'density'
        assert stats_dict['mean'] == 3.2
        assert stats_dict['n_samples'] == 50


class TestPropertyAnalyzer:
    """Test PropertyAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample material property data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'material_id': [f'mat_{i}' for i in range(n_samples)],
            'hardness': np.random.normal(28.0, 3.0, n_samples),
            'fracture_toughness': np.random.normal(4.5, 0.8, n_samples),
            'density': np.random.normal(3.2, 0.3, n_samples),
            'thermal_conductivity': np.random.normal(50.0, 10.0, n_samples),
            'dopant_concentration': np.repeat([0.01, 0.02, 0.03, 0.05, 0.10], 20),
        })
        
        return data
    
    def test_analyzer_initialization(self):
        """Test PropertyAnalyzer initialization."""
        analyzer = PropertyAnalyzer(significance_level=0.05, outlier_threshold=3.0)
        
        assert analyzer.significance_level == 0.05
        assert analyzer.outlier_threshold == 3.0
    
    def test_compute_statistics(self, sample_data):
        """Test computation of property statistics."""
        analyzer = PropertyAnalyzer()
        
        stats = analyzer.compute_statistics(sample_data, 'hardness')
        
        assert stats.property_name == 'hardness'
        assert stats.n_samples == 100
        assert 25.0 < stats.mean < 31.0
        assert stats.std > 0
        assert stats.min < stats.q25 < stats.median < stats.q75 < stats.max
    
    def test_compute_statistics_missing_property(self, sample_data):
        """Test error handling for missing property."""
        analyzer = PropertyAnalyzer()
        
        with pytest.raises(ValueError, match="not found in data"):
            analyzer.compute_statistics(sample_data, 'nonexistent_property')
    
    def test_compute_all_statistics(self, sample_data):
        """Test computation of statistics for all properties."""
        analyzer = PropertyAnalyzer()
        
        all_stats = analyzer.compute_all_statistics(sample_data)
        
        assert len(all_stats) == 5  # 5 numeric columns
        assert 'property' in all_stats.columns
        assert 'mean' in all_stats.columns
        assert 'std' in all_stats.columns
    
    def test_analyze_correlation(self, sample_data):
        """Test correlation analysis between two properties."""
        analyzer = PropertyAnalyzer()
        
        # Add correlated property
        sample_data['correlated_prop'] = sample_data['hardness'] * 1.5 + np.random.normal(0, 1, len(sample_data))
        
        corr = analyzer.analyze_correlation(sample_data, 'hardness', 'correlated_prop')
        
        assert corr.property1 == 'hardness'
        assert corr.property2 == 'correlated_prop'
        assert -1.0 <= corr.pearson_r <= 1.0
        assert -1.0 <= corr.spearman_r <= 1.0
        assert 0.0 <= corr.pearson_p <= 1.0
        assert corr.pearson_r > 0.5  # Should be positively correlated
    
    def test_analyze_correlation_insufficient_data(self):
        """Test correlation analysis with insufficient data."""
        analyzer = PropertyAnalyzer()
        
        data = pd.DataFrame({
            'prop1': [1.0, 2.0],
            'prop2': [3.0, 4.0],
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze_correlation(data, 'prop1', 'prop2')
    
    def test_compute_correlation_matrix(self, sample_data):
        """Test computation of correlation matrix."""
        analyzer = PropertyAnalyzer()
        
        properties = ['hardness', 'fracture_toughness', 'density']
        corr_matrix = analyzer.compute_correlation_matrix(sample_data, properties, method='pearson')
        
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
    
    def test_compute_correlation_matrix_spearman(self, sample_data):
        """Test Spearman correlation matrix."""
        analyzer = PropertyAnalyzer()
        
        corr_matrix = analyzer.compute_correlation_matrix(sample_data, method='spearman')
        
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert np.allclose(np.diag(corr_matrix), 1.0)
    
    def test_analyze_trend(self, sample_data):
        """Test trend analysis for concentration-dependent effects."""
        analyzer = PropertyAnalyzer()
        
        # Create data with clear trend
        sample_data['trending_prop'] = sample_data['dopant_concentration'] * 10 + np.random.normal(0, 0.5, len(sample_data))
        
        trend = analyzer.analyze_trend(sample_data, 'trending_prop', 'dopant_concentration')
        
        assert trend.property_name == 'trending_prop'
        assert trend.slope > 0  # Should be increasing
        assert 0.0 <= trend.r_squared <= 1.0
        assert trend.trend_direction == 'increasing'
        assert trend.is_significant
    
    def test_analyze_trend_no_trend(self, sample_data):
        """Test trend analysis with no significant trend."""
        analyzer = PropertyAnalyzer()
        
        # Create data with no trend (random noise)
        sample_data['random_prop'] = np.random.normal(5.0, 1.0, len(sample_data))
        
        trend = analyzer.analyze_trend(sample_data, 'random_prop', 'dopant_concentration')
        
        assert trend.property_name == 'random_prop'
        # Trend direction should be 'no_trend' if not significant
        if not trend.is_significant:
            assert trend.trend_direction == 'no_trend'
    
    def test_analyze_all_trends(self, sample_data):
        """Test trend analysis for all properties."""
        analyzer = PropertyAnalyzer()
        
        all_trends = analyzer.analyze_all_trends(sample_data, 'dopant_concentration')
        
        assert len(all_trends) > 0
        assert 'property' in all_trends.columns
        assert 'slope' in all_trends.columns
        assert 'r_squared' in all_trends.columns
        assert 'trend_direction' in all_trends.columns
    
    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        analyzer = PropertyAnalyzer(outlier_threshold=3.0)
        
        # Add some outliers
        sample_data.loc[0, 'hardness'] = 100.0  # Clear outlier
        sample_data.loc[1, 'hardness'] = -10.0  # Clear outlier
        
        outliers = analyzer.detect_outliers(sample_data, 'hardness')
        
        assert outliers.property_name == 'hardness'
        assert len(outliers.outlier_indices) >= 2
        assert 0 in outliers.outlier_indices
        assert 1 in outliers.outlier_indices
        assert outliers.threshold == 3.0
    
    def test_detect_outliers_no_variation(self):
        """Test outlier detection with no variation."""
        analyzer = PropertyAnalyzer()
        
        data = pd.DataFrame({
            'constant_prop': [5.0] * 10,
        })
        
        outliers = analyzer.detect_outliers(data, 'constant_prop')
        
        assert len(outliers.outlier_indices) == 0
    
    def test_detect_all_outliers(self, sample_data):
        """Test outlier detection for all properties."""
        analyzer = PropertyAnalyzer()
        
        # Add outliers to multiple properties
        sample_data.loc[0, 'hardness'] = 100.0
        sample_data.loc[1, 'density'] = 50.0
        
        all_outliers = analyzer.detect_all_outliers(sample_data)
        
        assert len(all_outliers) > 0
        assert 'property' in all_outliers.columns
        assert 'n_outliers' in all_outliers.columns
    
    def test_compare_distributions(self, sample_data):
        """Test distribution comparison between datasets."""
        analyzer = PropertyAnalyzer()
        
        # Create two datasets with different means
        data1 = sample_data.iloc[:50].copy()
        data2 = sample_data.iloc[50:].copy()
        data2['hardness'] = data2['hardness'] + 5.0  # Shift mean
        
        comparison = analyzer.compare_distributions(data1, data2, 'hardness')
        
        assert comparison['property'] == 'hardness'
        assert 'dataset1_mean' in comparison
        assert 'dataset2_mean' in comparison
        assert 't_pvalue' in comparison
        assert 'ks_pvalue' in comparison
        assert comparison['dataset2_mean'] > comparison['dataset1_mean']
    
    def test_compare_distributions_insufficient_data(self):
        """Test distribution comparison with insufficient data."""
        analyzer = PropertyAnalyzer()
        
        data1 = pd.DataFrame({'prop': [1.0]})
        data2 = pd.DataFrame({'prop': [2.0]})
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.compare_distributions(data1, data2, 'prop')


class TestCorrelationAnalysis:
    """Test CorrelationAnalysis dataclass."""
    
    def test_correlation_analysis_creation(self):
        """Test creation of CorrelationAnalysis."""
        corr = CorrelationAnalysis(
            property1='hardness',
            property2='density',
            pearson_r=0.75,
            pearson_p=0.001,
            spearman_r=0.72,
            spearman_p=0.002,
            is_significant=True,
        )
        
        assert corr.property1 == 'hardness'
        assert corr.property2 == 'density'
        assert corr.is_significant
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        corr = CorrelationAnalysis(
            property1='prop1',
            property2='prop2',
            pearson_r=0.5,
            pearson_p=0.05,
            spearman_r=0.48,
            spearman_p=0.06,
            is_significant=False,
        )
        
        corr_dict = corr.to_dict()
        assert corr_dict['property1'] == 'prop1'
        assert corr_dict['pearson_r'] == 0.5


class TestTrendAnalysis:
    """Test TrendAnalysis dataclass."""
    
    def test_trend_analysis_creation(self):
        """Test creation of TrendAnalysis."""
        trend = TrendAnalysis(
            property_name='hardness',
            concentrations=[0.01, 0.02, 0.03],
            values=[28.0, 29.0, 30.0],
            slope=100.0,
            intercept=27.0,
            r_squared=0.95,
            p_value=0.001,
            is_significant=True,
            trend_direction='increasing',
        )
        
        assert trend.property_name == 'hardness'
        assert trend.trend_direction == 'increasing'
        assert trend.is_significant


class TestOutlierDetection:
    """Test OutlierDetection dataclass."""
    
    def test_outlier_detection_creation(self):
        """Test creation of OutlierDetection."""
        outliers = OutlierDetection(
            property_name='hardness',
            outlier_indices=[0, 5, 10],
            outlier_values=[100.0, 95.0, 98.0],
            z_scores=[5.2, 4.8, 5.0],
            threshold=3.0,
        )
        
        assert outliers.property_name == 'hardness'
        assert len(outliers.outlier_indices) == 3
        assert outliers.threshold == 3.0
