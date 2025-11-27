"""Tests for ML model trainer."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
from hypothesis import given, strategies as st, settings, assume

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
            mape=5.0,
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
            mape=5.0,
            r2_cv_mean=0.68,
            r2_cv_std=0.05,
            r2_confidence_interval=(0.63, 0.73)
        )
        
        metrics_dict = metrics.to_dict()
        
        assert 'r2_score' in metrics_dict
        assert metrics_dict['r2_score'] == 0.70
        assert 'mape' in metrics_dict
        assert metrics_dict['mape'] == 5.0
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
    
    def test_train_gradient_boosting(self, sample_data):
        """Test Gradient Boosting training."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        assert model.model_type == 'GradientBoosting'
        assert model.metrics.r2_score > 0.5  # Should have reasonable performance
        assert len(model.feature_names) == 3
        assert model.feature_importances is not None
        assert len(model.feature_importances) == 3
        
        # Check hyperparameters match requirements
        assert model.hyperparameters['n_estimators'] == 150
        assert model.hyperparameters['learning_rate'] == 0.1
        assert model.hyperparameters['max_depth'] == 5
    
    def test_random_forest_default_hyperparameters(self, sample_data):
        """Test Random Forest uses correct default hyperparameters."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Check hyperparameters match requirements
        assert model.hyperparameters['n_estimators'] == 150
        assert model.hyperparameters['max_depth'] == 15
    
    def test_mape_metric_calculation(self, sample_data):
        """Test that MAPE metric is calculated."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # MAPE should be present in metrics
        assert hasattr(model.metrics, 'mape')
        assert isinstance(model.metrics.mape, float)
        assert model.metrics.mape >= 0  # MAPE should be non-negative
    
    def test_prepare_training_data_removes_missing_targets(self):
        """Test that prepare_training_data removes rows with missing targets."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y = pd.Series([100.0, np.nan, 200.0, np.nan, 300.0], name='target')
        
        trainer = ModelTrainer(random_seed=42)
        X_clean, y_clean = trainer.prepare_training_data(X, y)
        
        # Should have 3 rows (removed 2 with NaN targets)
        assert len(X_clean) == 3
        assert len(y_clean) == 3
        assert y_clean.notna().all()
    
    def test_prepare_training_data_removes_infinite_targets(self):
        """Test that prepare_training_data removes rows with infinite targets."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y = pd.Series([100.0, np.inf, 200.0, -np.inf, 300.0], name='target')
        
        trainer = ModelTrainer(random_seed=42)
        X_clean, y_clean = trainer.prepare_training_data(X, y)
        
        # Should have 3 rows (removed 2 with infinite targets)
        assert len(X_clean) == 3
        assert len(y_clean) == 3
        assert np.isfinite(y_clean).all()
    
    def test_prepare_training_data_filters_non_numeric_features(self):
        """Test that prepare_training_data filters out non-numeric features."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': ['a', 'b', 'c', 'd', 'e'],  # Non-numeric
            'feature3': [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], name='target')
        
        trainer = ModelTrainer(random_seed=42)
        X_clean, y_clean = trainer.prepare_training_data(X, y)
        
        # Should only have numeric features
        assert len(X_clean.columns) == 2
        assert 'feature1' in X_clean.columns
        assert 'feature3' in X_clean.columns
        assert 'feature2' not in X_clean.columns
    
    def test_prepare_training_data_removes_infinite_features(self):
        """Test that prepare_training_data removes rows with infinite feature values."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, np.inf, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, -np.inf, 50.0],
        })
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], name='target')
        
        trainer = ModelTrainer(random_seed=42)
        X_clean, y_clean = trainer.prepare_training_data(X, y)
        
        # Should have 3 rows (removed 2 with infinite features)
        assert len(X_clean) == 3
        assert len(y_clean) == 3
        assert np.isfinite(X_clean).all().all()
    
    def test_train_all_models_includes_gradient_boosting(self, sample_data):
        """Test that train_all_models includes Gradient Boosting."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        models = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=False)
        
        # Should include Gradient Boosting
        assert 'GradientBoosting' in models
        assert models['GradientBoosting'].model_type == 'GradientBoosting'
        
        # Should still include other models
        assert 'RandomForest' in models
        assert 'SVM' in models
    
    def test_batch_training_workflow(self, sample_data):
        """Test batch training workflow with all models."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_seed=42)
        
        # Train all models
        models = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=False)
        
        # Should have at least 3 models (RF, GB, SVM)
        assert len(models) >= 3
        
        # All models should have metrics
        for model_type, model in models.items():
            assert model.metrics is not None
            assert hasattr(model.metrics, 'r2_score')
            assert hasattr(model.metrics, 'mae')
            assert hasattr(model.metrics, 'rmse')
            assert hasattr(model.metrics, 'mape')
        
        # Compare models
        comparison = trainer.compare_models(models)
        assert len(comparison) >= 3
        assert 'GradientBoosting' in comparison['model_type'].values
        
        # Select best model
        best_model = trainer.select_best_model(models)
        assert best_model is not None
        assert best_model.model_type in models.keys()


class TestModelTrainerProperties:
    """Property-based tests for model trainer."""
    
    @st.composite
    def training_dataset(draw):
        """Generate training datasets with various data quality issues."""
        n_samples = draw(st.integers(min_value=50, max_value=200))
        n_features = draw(st.integers(min_value=2, max_value=10))
        
        # Generate feature data with potential issues
        X_data = {}
        for i in range(n_features):
            # Mix of clean and problematic data
            feature_type = draw(st.sampled_from(['clean', 'with_nan', 'with_inf', 'non_numeric']))
            
            if feature_type == 'clean':
                X_data[f'feature_{i}'] = draw(st.lists(
                    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=n_samples,
                    max_size=n_samples
                ))
            elif feature_type == 'with_nan':
                # Some NaN values
                values = [draw(st.one_of(
                    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    st.just(np.nan)
                )) for _ in range(n_samples)]
                X_data[f'feature_{i}'] = values
            elif feature_type == 'with_inf':
                # Some infinite values
                values = [draw(st.one_of(
                    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    st.sampled_from([np.inf, -np.inf])
                )) for _ in range(n_samples)]
                X_data[f'feature_{i}'] = values
            else:  # non_numeric
                # String column
                X_data[f'feature_{i}'] = [f'value_{j}' for j in range(n_samples)]
        
        X = pd.DataFrame(X_data)
        
        # Generate target with potential issues
        target_type = draw(st.sampled_from(['clean', 'with_nan', 'with_inf']))
        
        if target_type == 'clean':
            y = pd.Series(draw(st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=n_samples,
                max_size=n_samples
            )), name='target')
        elif target_type == 'with_nan':
            y = pd.Series([draw(st.one_of(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.just(np.nan)
            )) for _ in range(n_samples)], name='target')
        else:  # with_inf
            y = pd.Series([draw(st.one_of(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.sampled_from([np.inf, -np.inf])
            )) for _ in range(n_samples)], name='target')
        
        return X, y
    
    @given(data=training_dataset())
    @settings(max_examples=100, deadline=None)
    def test_property_ml_training_data_validation(self, data):
        """
        **Feature: sic-alloy-integration, Property 14: ML training data validation**
        
        Property: For any training dataset, after preparation, all feature values 
        should be finite and all target values should be finite.
        
        **Validates: Requirements 7.1**
        """
        X, y = data
        
        # Assume we have at least some valid data to work with
        assume(len(X) > 10)
        assume(len(y) > 10)
        
        trainer = ModelTrainer(random_seed=42)
        
        # Prepare the data
        X_clean, y_clean = trainer.prepare_training_data(X, y)
        
        # After preparation, IF there is data remaining, THEN all values should be finite
        # This handles the edge case where all data might be filtered out
        if len(y_clean) > 0:
            # Check target values
            assert y_clean.notna().all(), "Target contains NaN values after preparation"
            assert np.isfinite(y_clean).all(), "Target contains infinite values after preparation"
            
            # Check feature values
            assert len(X_clean) > 0, "X is empty but y is not"
            assert X_clean.notna().all().all(), "Features contain NaN values after preparation"
            assert np.isfinite(X_clean).all().all(), "Features contain infinite values after preparation"
            
            # Check that X and y have same length
            assert len(X_clean) == len(y_clean), "X and y have different lengths after preparation"
            
            # All columns should be numeric
            for col in X_clean.columns:
                assert pd.api.types.is_numeric_dtype(X_clean[col]), \
                    f"Column {col} is not numeric after preparation"
        else:
            # If all data was filtered out, X should also be empty
            assert len(X_clean) == 0, "X is not empty but y is empty"
