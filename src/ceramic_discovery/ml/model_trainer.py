"""Multi-algorithm ML trainer with hyperparameter optimization."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    
    r2_score: float
    rmse: float
    mae: float
    mape: float
    r2_cv_mean: float
    r2_cv_std: float
    r2_confidence_interval: Tuple[float, float]
    
    def is_within_target(self, target_min: float = 0.65, target_max: float = 0.75) -> bool:
        """Check if R² is within target range."""
        return target_min <= self.r2_score <= target_max
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'r2_score': self.r2_score,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'r2_cv_mean': self.r2_cv_mean,
            'r2_cv_std': self.r2_cv_std,
            'r2_confidence_interval': self.r2_confidence_interval,
        }


@dataclass
class TrainedModel:
    """Container for trained model with metadata."""
    
    model: Any
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: ModelMetrics
    feature_names: List[str]
    feature_importances: Optional[np.ndarray] = None
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix('.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics.to_dict(),
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None,
            'training_date': self.training_date,
            'version': self.version,
        }
        
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainedModel':
        """Load model from disk."""
        # Load model
        model_path = path.with_suffix('.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct metrics
        metrics_dict = metadata['metrics']
        metrics = ModelMetrics(
            r2_score=metrics_dict['r2_score'],
            rmse=metrics_dict['rmse'],
            mae=metrics_dict['mae'],
            mape=metrics_dict.get('mape', 0.0),  # Default for backward compatibility
            r2_cv_mean=metrics_dict['r2_cv_mean'],
            r2_cv_std=metrics_dict['r2_cv_std'],
            r2_confidence_interval=tuple(metrics_dict['r2_confidence_interval']),
        )
        
        return cls(
            model=model,
            model_type=metadata['model_type'],
            hyperparameters=metadata['hyperparameters'],
            metrics=metrics,
            feature_names=metadata['feature_names'],
            feature_importances=np.array(metadata['feature_importances']) if metadata['feature_importances'] else None,
            training_date=metadata['training_date'],
            version=metadata['version'],
        )


class ModelTrainer:
    """Multi-algorithm ML trainer with hyperparameter optimization."""
    
    def __init__(
        self,
        random_seed: int = 42,
        cv_folds: int = 5,
        target_r2_min: float = 0.65,
        target_r2_max: float = 0.75,
        n_optuna_trials: int = 50
    ):
        """
        Initialize model trainer.
        
        Args:
            random_seed: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            target_r2_min: Minimum target R² score
            target_r2_max: Maximum target R² score
            n_optuna_trials: Number of Optuna optimization trials
        """
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.target_r2_min = target_r2_min
        self.target_r2_max = target_r2_max
        self.n_optuna_trials = n_optuna_trials
        
        np.random.seed(random_seed)
    
    def prepare_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by removing invalid values.
        
        Removes rows with:
        - Missing target values
        - Infinite target values
        - Non-numeric features
        - Infinite feature values
        
        Args:
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            Cleaned X and y
        """
        # Remove rows with missing targets
        valid_target_mask = y.notna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]
        
        # Remove rows with infinite targets
        finite_target_mask = np.isfinite(y)
        X = X[finite_target_mask]
        y = y[finite_target_mask]
        
        # Filter to numeric features only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Remove rows with any infinite feature values
        finite_features_mask = np.all(np.isfinite(X), axis=1)
        X = X[finite_features_mask]
        y = y[finite_features_mask]
        
        # Reset indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        return X, y
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainedModel:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameters: Optional hyperparameters (if None, uses defaults)
        
        Returns:
            Trained model with metrics
        """
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 150,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_seed,
            }
        
        model = RandomForestRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_train, y_train, X_test, y_test)
        
        # Get feature importances
        feature_importances = model.feature_importances_
        
        return TrainedModel(
            model=model,
            model_type='RandomForest',
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_names=list(X_train.columns),
            feature_importances=feature_importances,
        )
    
    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainedModel:
        """
        Train Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameters: Optional hyperparameters (if None, uses defaults)
        
        Returns:
            Trained model with metrics
        """
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_seed,
            }
        
        model = GradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_train, y_train, X_test, y_test)
        
        # Get feature importances
        feature_importances = model.feature_importances_
        
        return TrainedModel(
            model=model,
            model_type='GradientBoosting',
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_names=list(X_train.columns),
            feature_importances=feature_importances,
        )
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainedModel:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameters: Optional hyperparameters (if None, uses defaults)
        
        Returns:
            Trained model with metrics
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_seed,
            }
        
        model = xgb.XGBRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_train, y_train, X_test, y_test)
        
        # Get feature importances
        feature_importances = model.feature_importances_
        
        return TrainedModel(
            model=model,
            model_type='XGBoost',
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_names=list(X_train.columns),
            feature_importances=feature_importances,
        )
    
    def train_svm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainedModel:
        """
        Train SVM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameters: Optional hyperparameters (if None, uses defaults)
        
        Returns:
            Trained model with metrics
        """
        if hyperparameters is None:
            hyperparameters = {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale',
            }
        
        model = SVR(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_train, y_train, X_test, y_test)
        
        return TrainedModel(
            model=model,
            model_type='SVM',
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_names=list(X_train.columns),
            feature_importances=None,  # SVM doesn't have feature importances
        )
    
    def _calculate_metrics(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        # Test set predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero by filtering out near-zero values
        mask = np.abs(y_test) > 1e-10
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.inf
        
        # Cross-validation on training set
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        
        # Confidence interval (95%)
        r2_cv_mean = cv_scores.mean()
        r2_cv_std = cv_scores.std()
        confidence_interval = (
            r2_cv_mean - 1.96 * r2_cv_std,
            r2_cv_mean + 1.96 * r2_cv_std
        )
        
        return ModelMetrics(
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2_cv_mean=r2_cv_mean,
            r2_cv_std=r2_cv_std,
            r2_confidence_interval=confidence_interval,
        )
    
    def optimize_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> TrainedModel:
        """
        Optimize Random Forest hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Trained model with optimized hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available. Using default hyperparameters.")
            return self.train_random_forest(X_train, y_train, X_test, y_test)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': self.random_seed,
            }
            
            model = RandomForestRegressor(**params)
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_seed
        
        return self.train_random_forest(X_train, y_train, X_test, y_test, best_params)
    
    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> TrainedModel:
        """
        Optimize XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Trained model with optimized hyperparameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        if not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available. Using default hyperparameters.")
            return self.train_xgboost(X_train, y_train, X_test, y_test)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_seed,
            }
            
            model = xgb.XGBRegressor(**params)
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_seed
        
        return self.train_xgboost(X_train, y_train, X_test, y_test, best_params)
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        optimize: bool = False
    ) -> Dict[str, TrainedModel]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            optimize: Whether to optimize hyperparameters
        
        Returns:
            Dictionary mapping model type to trained model
        """
        models = {}
        
        # Random Forest
        if optimize:
            models['RandomForest'] = self.optimize_random_forest(X_train, y_train, X_test, y_test)
        else:
            models['RandomForest'] = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Gradient Boosting
        models['GradientBoosting'] = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            if optimize:
                models['XGBoost'] = self.optimize_xgboost(X_train, y_train, X_test, y_test)
            else:
                models['XGBoost'] = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # SVM
        models['SVM'] = self.train_svm(X_train, y_train, X_test, y_test)
        
        return models
    
    def select_best_model(self, models: Dict[str, TrainedModel]) -> TrainedModel:
        """
        Select best model based on R² score.
        
        Args:
            models: Dictionary of trained models
        
        Returns:
            Best performing model
        """
        best_model = None
        best_r2 = -np.inf
        
        for model_type, model in models.items():
            if model.metrics.r2_score > best_r2:
                best_r2 = model.metrics.r2_score
                best_model = model
        
        return best_model
    
    def compare_models(self, models: Dict[str, TrainedModel]) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            models: Dictionary of trained models
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for model_type, model in models.items():
            comparison.append({
                'model_type': model_type,
                'r2_score': model.metrics.r2_score,
                'rmse': model.metrics.rmse,
                'mae': model.metrics.mae,
                'r2_cv_mean': model.metrics.r2_cv_mean,
                'r2_cv_std': model.metrics.r2_cv_std,
                'within_target': model.metrics.is_within_target(self.target_r2_min, self.target_r2_max),
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('r2_score', ascending=False)
        
        return df
