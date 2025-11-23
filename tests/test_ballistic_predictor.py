"""Tests for ballistic performance prediction."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ceramic_discovery.ballistics import (
    BallisticPredictor,
    BallisticPrediction,
    MechanismAnalysis,
    PropertyContribution,
)
from ceramic_discovery.ml.model_trainer import ModelTrainer
from ceramic_discovery.ml.uncertainty_quantification import UncertaintyQuantifier


@pytest.fixture
def sample_training_data():
    """Create sample training data for V50 prediction."""
    np.random.seed(42)
    n_samples = 50
    
    # Generate synthetic material properties
    data = {
        'hardness': np.random.uniform(20, 35, n_samples),
        'fracture_toughness': np.random.uniform(3, 6, n_samples),
        'density': np.random.uniform(2.5, 4.0, n_samples),
        'thermal_conductivity_1000C': np.random.uniform(30, 60, n_samples),
        'youngs_modulus': np.random.uniform(350, 450, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Generate synthetic V50 values (correlated with properties)
    v50 = (
        20 * X['hardness'] +
        50 * X['fracture_toughness'] +
        100 * X['density'] +
        2 * X['thermal_conductivity_1000C'] +
        0.5 * X['youngs_modulus'] +
        np.random.normal(0, 50, n_samples)
    )
    
    y = pd.Series(v50, name='v50')
    
    return X, y


@pytest.fixture
def trained_predictor(sample_training_data):
    """Create a trained ballistic predictor."""
    X, y = sample_training_data
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    trainer = ModelTrainer(random_seed=42)
    model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Create predictor
    predictor = BallisticPredictor(
        model=model,
        random_seed=42
    )
    predictor.set_training_data(X_train, y_train)
    
    return predictor, X_test, y_test


class TestBallisticPrediction:
    """Test BallisticPrediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating a ballistic prediction."""
        prediction = BallisticPrediction(
            v50=800.0,
            confidence_interval=(750.0, 850.0),
            uncertainty=25.0,
            reliability_score=0.85,
            physical_bounds_satisfied=True,
        )
        
        assert prediction.v50 == 800.0
        assert prediction.confidence_interval == (750.0, 850.0)
        assert prediction.uncertainty == 25.0
        assert prediction.reliability_score == 0.85
        assert prediction.physical_bounds_satisfied is True
    
    def test_prediction_to_dict(self):
        """Test converting prediction to dictionary."""
        prediction = BallisticPrediction(
            v50=800.0,
            confidence_interval=(750.0, 850.0),
            uncertainty=25.0,
            reliability_score=0.85,
            physical_bounds_satisfied=True,
        )
        
        result = prediction.to_dict()
        
        assert result['v50'] == 800.0
        assert result['confidence_interval'] == (750.0, 850.0)
        assert result['uncertainty'] == 25.0
        assert result['reliability_score'] == 0.85
        assert result['physical_bounds_satisfied'] is True
    
    def test_format_prediction(self):
        """Test formatting prediction as string."""
        prediction = BallisticPrediction(
            v50=800.0,
            confidence_interval=(750.0, 850.0),
            uncertainty=25.0,
            reliability_score=0.85,
            physical_bounds_satisfied=True,
        )
        
        formatted = prediction.format_prediction()
        
        assert '800.0' in formatted
        assert '750.0' in formatted
        assert '850.0' in formatted
        assert '0.85' in formatted


class TestBallisticPredictor:
    """Test BallisticPredictor class."""
    
    def test_predictor_initialization(self):
        """Test initializing ballistic predictor."""
        predictor = BallisticPredictor(random_seed=42)
        
        assert predictor.model is None
        assert predictor.uncertainty_quantifier is not None
        assert predictor.random_seed == 42
        assert predictor.X_train is None
        assert predictor.y_train is None
    
    def test_set_training_data(self, sample_training_data):
        """Test setting training data."""
        X, y = sample_training_data
        predictor = BallisticPredictor()
        
        predictor.set_training_data(X, y)
        
        assert predictor.X_train is not None
        assert predictor.y_train is not None
        assert len(predictor.X_train) == len(X)
        assert len(predictor.y_train) == len(y)
    
    def test_check_physical_bounds(self):
        """Test physical bounds checking."""
        predictor = BallisticPredictor()
        
        # Within bounds
        assert predictor._check_physical_bounds(800.0) is True
        assert predictor._check_physical_bounds(200.0) is True
        assert predictor._check_physical_bounds(2000.0) is True
        
        # Outside bounds
        assert predictor._check_physical_bounds(100.0) is False
        assert predictor._check_physical_bounds(2500.0) is False
    
    def test_enforce_physical_bounds(self):
        """Test enforcing physical bounds."""
        predictor = BallisticPredictor()
        
        # Within bounds - no change
        assert predictor._enforce_physical_bounds(800.0) == 800.0
        
        # Below minimum - clipped
        assert predictor._enforce_physical_bounds(100.0) == 200.0
        
        # Above maximum - clipped
        assert predictor._enforce_physical_bounds(2500.0) == 2000.0
    
    def test_predict_v50(self, trained_predictor):
        """Test V50 prediction with uncertainty."""
        predictor, X_test, y_test = trained_predictor
        
        # Predict for first test sample
        material = X_test.iloc[[0]]
        prediction = predictor.predict_v50(material)
        
        # Check prediction structure
        assert isinstance(prediction, BallisticPrediction)
        assert prediction.v50 > 0
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] < prediction.v50
        assert prediction.confidence_interval[1] > prediction.v50
        assert 0 <= prediction.reliability_score <= 1
        assert isinstance(prediction.physical_bounds_satisfied, bool)
    
    def test_predict_v50_without_model(self):
        """Test prediction fails without model."""
        predictor = BallisticPredictor()
        material = pd.DataFrame({'hardness': [28.0]})
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_v50(material)
    
    def test_predict_v50_without_training_data(self, sample_training_data):
        """Test prediction fails without training data."""
        X, y = sample_training_data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        predictor = BallisticPredictor(model=model)
        material = X_test.iloc[[0]]
        
        with pytest.raises(ValueError, match="Training data not set"):
            predictor.predict_v50(material)
    
    def test_predict_batch(self, trained_predictor):
        """Test batch prediction."""
        predictor, X_test, y_test = trained_predictor
        
        # Predict for all test samples
        predictions = predictor.predict_batch(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, BallisticPrediction) for p in predictions)
        assert all(p.v50 > 0 for p in predictions)
    
    def test_physical_bounds_enforcement(self, trained_predictor):
        """Test that physical bounds are enforced."""
        predictor, X_test, y_test = trained_predictor
        
        # Create extreme material properties that might produce out-of-bounds predictions
        extreme_material = pd.DataFrame({
            'hardness': [50.0],  # Very high
            'fracture_toughness': [10.0],  # Very high
            'density': [5.0],  # Very high
            'thermal_conductivity_1000C': [100.0],  # Very high
            'youngs_modulus': [600.0],  # Very high
        })
        
        prediction = predictor.predict_v50(extreme_material, enforce_bounds=True)
        
        # Check bounds are enforced
        assert predictor.V50_MIN <= prediction.v50 <= predictor.V50_MAX
        assert predictor.V50_MIN <= prediction.confidence_interval[0] <= predictor.V50_MAX
        assert predictor.V50_MIN <= prediction.confidence_interval[1] <= predictor.V50_MAX


class TestMechanismAnalysis:
    """Test mechanism analysis functionality."""
    
    def test_analyze_mechanisms(self, trained_predictor):
        """Test mechanism analysis."""
        predictor, X_test, y_test = trained_predictor
        
        # Analyze mechanisms
        analysis = predictor.analyze_mechanisms(X_test, y_test)
        
        assert isinstance(analysis, MechanismAnalysis)
        assert len(analysis.property_contributions) > 0
        assert analysis.dominant_mechanism is not None
        assert isinstance(analysis.correlation_matrix, pd.DataFrame)
    
    def test_property_contributions(self, trained_predictor):
        """Test property contribution analysis."""
        predictor, X_test, y_test = trained_predictor
        
        analysis = predictor.analyze_mechanisms(X_test, y_test)
        
        # Check property contributions
        for contrib in analysis.property_contributions:
            assert isinstance(contrib, PropertyContribution)
            assert contrib.property_name in X_test.columns
            assert 0 <= contrib.contribution_score <= 1
            assert -1 <= contrib.correlation <= 1
            assert 0 <= contrib.p_value <= 1
    
    def test_thermal_conductivity_impact(self, trained_predictor):
        """Test thermal conductivity impact analysis."""
        predictor, X_test, y_test = trained_predictor
        
        analysis = predictor.analyze_mechanisms(X_test, y_test)
        
        # Thermal conductivity should have some impact
        assert analysis.thermal_conductivity_impact >= 0
    
    def test_get_top_contributors(self, trained_predictor):
        """Test getting top property contributors."""
        predictor, X_test, y_test = trained_predictor
        
        analysis = predictor.analyze_mechanisms(X_test, y_test)
        top_contributors = analysis.get_top_contributors(n=3)
        
        assert len(top_contributors) <= 3
        assert all(isinstance(c, PropertyContribution) for c in top_contributors)
        
        # Check they are sorted by contribution
        if len(top_contributors) > 1:
            for i in range(len(top_contributors) - 1):
                assert (abs(top_contributors[i].contribution_score) >=
                        abs(top_contributors[i + 1].contribution_score))


class TestPredictionAccuracy:
    """Test prediction accuracy validation."""
    
    def test_validate_prediction_accuracy(self, trained_predictor):
        """Test validation of prediction accuracy."""
        predictor, X_test, y_test = trained_predictor
        
        metrics = predictor.validate_prediction_accuracy(X_test, y_test)
        
        # Check metrics are present
        assert 'r2_score' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'coverage' in metrics
        assert 'mean_reliability' in metrics
        
        # Check metrics are reasonable
        assert -1 <= metrics['r2_score'] <= 1
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert 0 <= metrics['coverage'] <= 1
        assert 0 <= metrics['mean_reliability'] <= 1
