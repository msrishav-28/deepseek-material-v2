"""Tests for uncertainty quantification system."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ceramic_discovery.ml.uncertainty_quantification import (
    UncertaintyQuantifier,
    UncertaintyPropagator,
    ConfidenceIntervalReporter,
    PredictionWithUncertainty,
)


class TestUncertaintyQuantifier:
    """Test uncertainty quantifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'x1': np.random.uniform(0, 10, n_samples),
            'x2': np.random.uniform(0, 10, n_samples),
        })
        
        # Target with known relationship
        y = pd.Series(2 * X['x1'] + 3 * X['x2'] + np.random.normal(0, 1, n_samples))
        
        # Split
        split_idx = 80
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model."""
        X_train, _, y_train, _ = sample_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def test_bootstrap_predictions(self, trained_model, sample_data):
        """Test bootstrap prediction generation."""
        X_train, X_test, y_train, _ = sample_data
        
        quantifier = UncertaintyQuantifier(n_bootstrap=10, random_seed=42)
        bootstrap_preds = quantifier.bootstrap_predictions(
            trained_model, X_train, y_train, X_test
        )
        
        assert bootstrap_preds.shape == (10, len(X_test))
        assert not np.isnan(bootstrap_preds).any()
    
    def test_prediction_intervals(self, trained_model, sample_data):
        """Test prediction interval calculation."""
        X_train, X_test, y_train, _ = sample_data
        
        quantifier = UncertaintyQuantifier(n_bootstrap=50, confidence_level=0.95, random_seed=42)
        bootstrap_preds = quantifier.bootstrap_predictions(
            trained_model, X_train, y_train, X_test
        )
        
        lower, upper = quantifier.calculate_prediction_intervals(bootstrap_preds)
        
        assert len(lower) == len(X_test)
        assert len(upper) == len(X_test)
        assert np.all(lower <= upper)
    
    def test_reliability_score(self, sample_data):
        """Test reliability score calculation."""
        X_train, X_test, _, _ = sample_data
        
        quantifier = UncertaintyQuantifier(random_seed=42)
        prediction_std = np.random.uniform(0.1, 1.0, len(X_test))
        
        reliability = quantifier.calculate_reliability_score(
            X_test, X_train, prediction_std
        )
        
        assert len(reliability) == len(X_test)
        assert np.all((reliability >= 0) & (reliability <= 1))
    
    def test_quantify_uncertainty(self, trained_model, sample_data):
        """Test full uncertainty quantification."""
        X_train, X_test, y_train, y_test = sample_data
        
        quantifier = UncertaintyQuantifier(n_bootstrap=20, random_seed=42)
        metrics = quantifier.quantify_uncertainty(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        assert metrics.prediction_intervals.shape == (len(X_test), 2)
        assert len(metrics.prediction_std) == len(X_test)
        assert len(metrics.reliability_scores) == len(X_test)
        assert 0.0 <= metrics.coverage <= 1.0
    
    def test_predict_with_uncertainty(self, trained_model, sample_data):
        """Test predictions with uncertainty."""
        X_train, X_test, y_train, _ = sample_data
        
        quantifier = UncertaintyQuantifier(n_bootstrap=20, random_seed=42)
        predictions = quantifier.predict_with_uncertainty(
            trained_model, X_train, y_train, X_test
        )
        
        assert len(predictions) == len(X_test)
        
        for pred in predictions:
            assert isinstance(pred, PredictionWithUncertainty)
            assert pred.uncertainty >= 0
            assert pred.confidence_interval[0] <= pred.prediction <= pred.confidence_interval[1]
            assert 0.0 <= pred.reliability_score <= 1.0


class TestUncertaintyPropagator:
    """Test uncertainty propagation."""
    
    def test_propagate_addition(self):
        """Test uncertainty propagation through addition."""
        values = [10.0, 20.0]
        uncertainties = [1.0, 2.0]
        
        result, uncertainty = UncertaintyPropagator.propagate_addition(values, uncertainties)
        
        assert result == 30.0
        assert uncertainty == pytest.approx(np.sqrt(1**2 + 2**2))
    
    def test_propagate_multiplication(self):
        """Test uncertainty propagation through multiplication."""
        values = [10.0, 20.0]
        uncertainties = [1.0, 2.0]
        
        result, uncertainty = UncertaintyPropagator.propagate_multiplication(values, uncertainties)
        
        assert result == 200.0
        # Relative uncertainty: sqrt((1/10)^2 + (2/20)^2) = sqrt(0.01 + 0.01) = 0.1414
        expected_uncertainty = 200.0 * np.sqrt((1/10)**2 + (2/20)**2)
        assert uncertainty == pytest.approx(expected_uncertainty)
    
    def test_propagate_power(self):
        """Test uncertainty propagation through power."""
        value = 10.0
        uncertainty = 1.0
        exponent = 2.0
        
        result, uncertainty_result = UncertaintyPropagator.propagate_power(
            value, uncertainty, exponent
        )
        
        assert result == 100.0
        # Relative uncertainty: |2| * (1/10) = 0.2
        expected_uncertainty = 100.0 * 0.2
        assert uncertainty_result == pytest.approx(expected_uncertainty)
    
    def test_propagate_function(self):
        """Test uncertainty propagation through arbitrary function."""
        np.random.seed(42)
        
        def func(x, y):
            return x**2 + y**2
        
        values = [3.0, 4.0]
        uncertainties = [0.1, 0.1]
        
        result, uncertainty = UncertaintyPropagator.propagate_function(
            func, values, uncertainties, n_samples=1000
        )
        
        # Expected: 3^2 + 4^2 = 25
        assert result == pytest.approx(25.0, abs=0.5)
        assert uncertainty > 0


class TestConfidenceIntervalReporter:
    """Test confidence interval reporting."""
    
    def test_format_prediction_with_ci(self):
        """Test formatting prediction with confidence interval."""
        prediction = 100.0
        ci = (95.0, 105.0)
        
        formatted = ConfidenceIntervalReporter.format_prediction_with_ci(
            prediction, ci, confidence_level=0.95
        )
        
        assert "100.00" in formatted
        assert "95%" in formatted
        assert "95.00" in formatted
        assert "105.00" in formatted
    
    def test_create_prediction_report(self):
        """Test creating prediction report."""
        predictions = [
            PredictionWithUncertainty(
                prediction=100.0,
                uncertainty=5.0,
                confidence_interval=(95.0, 105.0),
                reliability_score=0.9
            ),
            PredictionWithUncertainty(
                prediction=200.0,
                uncertainty=10.0,
                confidence_interval=(190.0, 210.0),
                reliability_score=0.8
            ),
        ]
        
        report = ConfidenceIntervalReporter.create_prediction_report(predictions)
        
        assert len(report) == 2
        assert 'prediction' in report.columns
        assert 'uncertainty' in report.columns
        assert 'reliability_score' in report.columns
        assert 'ci_width' in report.columns
        
        assert report.loc[0, 'prediction'] == 100.0
        assert report.loc[0, 'ci_width'] == 10.0
        assert report.loc[1, 'reliability_score'] == 0.8
