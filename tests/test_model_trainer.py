"""Tests for ML model trainer."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from ceramic_discovery.ml.model_trainer import (
    ModelTrainer,
    ModelMetrics,
    TrainedModel,
)


class TestModelMetrics:
    """Test model metrics."""
    
    def test_metrics_within_target(self):
        """Test checking if metrics are within target range."""
        metrics = ModelMetrics(
            r2_score=0.70,
            rmse=10.0,
            mae=8.0,
            r2_cv_mean=0.68,
            r2_cv_std=0.05,
            r2_confidence_interval=(0.63, 0.73)
        )
        
        assert metrics.is_within_target(0.65, 0.75)
        assert not metrics.is_within_target(0.75, 0.85)
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(
            r2_score=0.70,
            rmse=10.0,
            mae=8.0,
            r2_cv_mean=0.68,
            r2_cv_std=0.05,
            r2_confidence_interval=(0.63, 0.73)
        )
        
        metrics_dict = metrics.to_dict()
        
        assert 'r2_score' in metrics_dict
        assert metrics_dict['r2_score'] == 0.70
        assert metrics_dict['r2_confidence_interval'] == (0.63, 0.73)


class TestModelTrainer:
    """Test model trainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        # Create synthetic data with known relationship
        X = pd.DataFrame({
            'density': np.random.uniform(2.0, 5.0, n_samples),
            'hardness': np.random.uniform(20.0, 35.0, n_samples),
            'thermal_conductivity': np.random.uniform(20.0, 120.0, n_samples),
        })
        
        # Target with some noise
        y = pd.Series(
            10 * X['density'] + 2 * X['hardness'] + 0.5 * X['thermal_conductivity'] + np.random.normal(0, 5, n_samples),
            name='target'
        )
        
        # Split into train/test
        split_idx = int(0.8 * n_samples)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def test_train_random_forest(self, sample_data):
        """Test Random Forest training."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        assert model.model_type == 'RandomForest'
        assert model.metrics.r2_score > 0.5  # Should have reasonable performance
        assert len(model.feature_names) == 3
        assert model.feature_importances is not None
        assert len(model.feature_importances) == 3
    
    def test_train_svm(self, sample_data):
        """Test SVM training."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_svm(X_train, y_train, X_test, y_test)
        
        assert model.model_type == 'SVM'
        # SVM may have lower performance, just check it runs
        assert model.metrics.r2_score > 0.0
        assert len(model.feature_names) == 3
        assert model.feature_importances is None  # SVM doesn't have feature importances
    
    def test_cross_validation_metrics(self, sample_data):
        """Test that cross-validation metrics are calculated."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42, cv_folds=3)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        assert model.metrics.r2_cv_mean is not None
        assert model.metrics.r2_cv_std is not None
        assert model.metrics.r2_confidence_interval is not None
        assert len(model.metrics.r2_confidence_interval) == 2
    
    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        models = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=False)
        
        assert 'RandomForest' in models
        assert 'SVM' in models
        # XGBoost may or may not be available
    
    def test_select_best_model(self, sample_data):
        """Test selecting best model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        models = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=False)
        
        best_model = trainer.select_best_model(models)
        
        assert best_model is not None
        assert best_model.model_type in ['RandomForest', 'XGBoost', 'SVM']
    
    def test_compare_models(self, sample_data):
        """Test model comparison."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        models = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=False)
        
        comparison = trainer.compare_models(models)
        
        assert len(comparison) >= 2  # At least RF and SVM
        assert 'model_type' in comparison.columns
        assert 'r2_score' in comparison.columns
        assert 'within_target' in comparison.columns
        
        # Should be sorted by R² score
        assert comparison['r2_score'].is_monotonic_decreasing
    
    def test_model_save_load(self, sample_data):
        """Test model persistence."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model'
            
            # Save model
            model.save(model_path)
            
            # Load model
            loaded_model = TrainedModel.load(model_path)
            
            assert loaded_model.model_type == model.model_type
            assert loaded_model.metrics.r2_score == model.metrics.r2_score
            assert loaded_model.feature_names == model.feature_names
            
            # Test predictions are the same
            pred_original = model.model.predict(X_test)
            pred_loaded = loaded_model.model.predict(X_test)
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)
    
    def test_realistic_performance_target(self, sample_data):
        """Test that models can achieve realistic performance targets."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42, target_r2_min=0.65, target_r2_max=0.75)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # With synthetic data, we should be able to achieve good performance
        # In real scenarios, R² = 0.65-0.75 is the realistic target
        assert model.metrics.r2_score > 0.5
