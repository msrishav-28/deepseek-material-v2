"""Incremental learning for ML models with large datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """Configuration for incremental learning."""
    
    batch_size: int = 100
    learning_rate: float = 0.01
    max_iter_per_batch: int = 1
    warm_start: bool = True
    shuffle: bool = True
    random_state: int = 42


class IncrementalLearner:
    """
    Incremental learning for ML models with large datasets.
    
    Supports online learning algorithms that can be trained
    incrementally without loading entire dataset into memory.
    """
    
    def __init__(
        self,
        model_type: str = 'sgd',
        config: Optional[IncrementalConfig] = None
    ):
        """
        Initialize incremental learner.
        
        Args:
            model_type: Type of model ('sgd', 'mlp')
            config: Incremental learning configuration
        """
        self.model_type = model_type
        self.config = config or IncrementalConfig()
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_samples_seen = 0
        self.feature_names: Optional[List[str]] = None
        
        self._initialize_model()
        
        logger.info(f"Initialized IncrementalLearner with model_type={model_type}")
    
    def _initialize_model(self) -> None:
        """Initialize the incremental model."""
        if self.model_type == 'sgd':
            self.model = SGDRegressor(
                learning_rate='adaptive',
                eta0=self.config.learning_rate,
                max_iter=self.config.max_iter_per_batch,
                warm_start=self.config.warm_start,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
        
        elif self.model_type == 'mlp':
            if not MLP_AVAILABLE:
                raise ImportError("MLPRegressor not available")
            
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.max_iter_per_batch,
                warm_start=self.config.warm_start,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def partial_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        update_scaler: bool = True
    ) -> None:
        """
        Incrementally fit model on a batch of data.
        
        Args:
            X: Feature batch
            y: Target batch
            update_scaler: Whether to update the scaler
        """
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        
        # Scale features
        if update_scaler:
            X_scaled = self.scaler.partial_fit(X).transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Fit model
        self.model.partial_fit(X_scaled, y)
        
        self.is_fitted = True
        self.n_samples_seen += len(X)
        
        logger.debug(f"Partial fit on {len(X)} samples (total: {self.n_samples_seen})")
    
    def fit_batches(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        show_progress: bool = False
    ) -> None:
        """
        Fit model on data in batches.
        
        Args:
            X: Features
            y: Target
            show_progress: Whether to show progress
        """
        n_samples = len(X)
        n_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size
        
        logger.info(f"Fitting model on {n_samples} samples in {n_batches} batches")
        
        for i in range(0, n_samples, self.config.batch_size):
            X_batch = X.iloc[i:i + self.config.batch_size]
            y_batch = y.iloc[i:i + self.config.batch_size]
            
            self.partial_fit(X_batch, y_batch)
            
            if show_progress and (i // self.config.batch_size) % max(1, n_batches // 10) == 0:
                progress = (i + len(X_batch)) / n_samples * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        logger.info(f"Completed fitting on {self.n_samples_seen} samples")
    
    def fit_from_file(
        self,
        file_path: Path,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        file_format: str = 'csv',
        show_progress: bool = False
    ) -> None:
        """
        Fit model incrementally from file without loading entire file.
        
        Args:
            file_path: Path to data file
            target_column: Name of target column
            feature_columns: List of feature columns (None = all except target)
            file_format: File format ('csv', 'parquet')
            show_progress: Whether to show progress
        """
        logger.info(f"Fitting model from file: {file_path}")
        
        batch_num = 0
        
        if file_format == 'csv':
            for chunk in pd.read_csv(file_path, chunksize=self.config.batch_size):
                X_batch, y_batch = self._prepare_batch(chunk, target_column, feature_columns)
                self.partial_fit(X_batch, y_batch)
                batch_num += 1
                
                if show_progress and batch_num % 10 == 0:
                    logger.info(f"Processed {batch_num} batches ({self.n_samples_seen} samples)")
        
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
            self.fit_batches(
                df.drop(columns=[target_column]),
                df[target_column],
                show_progress=show_progress
            )
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Completed fitting from file ({self.n_samples_seen} samples)")
    
    def _prepare_batch(
        self,
        batch: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare batch by separating features and target."""
        if feature_columns is None:
            feature_columns = [col for col in batch.columns if col != target_column]
        
        X = batch[feature_columns]
        y = batch[target_column]
        
        return X, y
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            R² score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Output path
        """
        import pickle
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self.n_samples_seen,
            'feature_names': self.feature_names,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved incremental model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'IncrementalLearner':
        """
        Load model from disk.
        
        Args:
            path: Input path
        
        Returns:
            Loaded incremental learner
        """
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        learner = cls(
            model_type=model_data['model_type'],
            config=model_data['config']
        )
        
        learner.model = model_data['model']
        learner.scaler = model_data['scaler']
        learner.is_fitted = model_data['is_fitted']
        learner.n_samples_seen = model_data['n_samples_seen']
        learner.feature_names = model_data['feature_names']
        
        logger.info(f"Loaded incremental model from {path}")
        
        return learner
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self.n_samples_seen,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
        }
