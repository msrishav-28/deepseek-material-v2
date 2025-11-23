"""Uncertainty quantification system for ML predictions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample


@dataclass
class PredictionWithUncertainty:
    """Container for prediction with uncertainty estimates."""
    
    prediction: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    reliability_score: float  # 0-1, higher is more reliable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'prediction': self.prediction,
            'uncertainty': self.uncertainty,
            'confidence_interval': self.confidence_interval,
            'reliability_score': self.reliability_score,
        }


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty quantification metrics."""
    
    prediction_intervals: np.ndarray  # Shape: (n_samples, 2) for lower and upper bounds
    prediction_std: np.ndarray  # Standard deviation of predictions
    reliability_scores: np.ndarray  # Reliability score for each prediction
    coverage: float  # Fraction of true values within prediction intervals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'prediction_intervals': self.prediction_intervals.tolist(),
            'prediction_std': self.prediction_std.tolist(),
            'reliability_scores': self.reliability_scores.tolist(),
            'coverage': self.coverage,
        }


class UncertaintyQuantifier:
    """Uncertainty quantification using bootstrap sampling."""
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        """
        Initialize uncertainty quantifier.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        self.lower_percentile = (alpha / 2) * 100
        self.upper_percentile = (1 - alpha / 2) * 100
    
    def bootstrap_predictions(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate bootstrap predictions.
        
        Args:
            model: Trained model (must have fit and predict methods)
            X_train: Training features
            y_train: Training target
            X_predict: Features to predict on
        
        Returns:
            Array of shape (n_bootstrap, n_samples) with bootstrap predictions
        """
        predictions = np.zeros((self.n_bootstrap, len(X_predict)))
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample with replacement
            X_boot, y_boot = resample(
                X_train, y_train,
                random_state=self.random_seed + i
            )
            
            # Clone and train model on bootstrap sample
            from sklearn.base import clone
            model_boot = clone(model)
            model_boot.fit(X_boot, y_boot)
            
            # Predict on test data
            predictions[i, :] = model_boot.predict(X_predict)
        
        return predictions
    
    def calculate_prediction_intervals(
        self,
        bootstrap_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals from bootstrap predictions.
        
        Args:
            bootstrap_predictions: Array of shape (n_bootstrap, n_samples)
        
        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        lower_bounds = np.percentile(
            bootstrap_predictions,
            self.lower_percentile,
            axis=0
        )
        upper_bounds = np.percentile(
            bootstrap_predictions,
            self.upper_percentile,
            axis=0
        )
        
        return lower_bounds, upper_bounds
    
    def calculate_reliability_score(
        self,
        X: pd.DataFrame,
        X_train: pd.DataFrame,
        prediction_std: np.ndarray
    ) -> np.ndarray:
        """
        Calculate reliability score for predictions.
        
        Reliability is based on:
        1. Distance to training data (closer = more reliable)
        2. Prediction uncertainty (lower = more reliable)
        
        Args:
            X: Features to predict on
            X_train: Training features
            prediction_std: Standard deviation of predictions
        
        Returns:
            Array of reliability scores (0-1)
        """
        # Calculate distance to nearest training sample
        from scipy.spatial.distance import cdist
        
        distances = cdist(X, X_train, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        # Normalize distances (0 = at training point, 1 = far from training)
        max_distance = min_distances.max()
        if max_distance > 0:
            normalized_distances = min_distances / max_distance
        else:
            normalized_distances = np.zeros_like(min_distances)
        
        # Normalize prediction uncertainty
        max_std = prediction_std.max()
        if max_std > 0:
            normalized_std = prediction_std / max_std
        else:
            normalized_std = np.zeros_like(prediction_std)
        
        # Combine factors (lower is better, so invert)
        reliability = 1.0 - (0.5 * normalized_distances + 0.5 * normalized_std)
        
        # Ensure in [0, 1] range
        reliability = np.clip(reliability, 0.0, 1.0)
        
        return reliability
    
    def quantify_uncertainty(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame,
        y_true: Optional[pd.Series] = None
    ) -> UncertaintyMetrics:
        """
        Quantify uncertainty for predictions.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_predict: Features to predict on
            y_true: Optional true values for coverage calculation
        
        Returns:
            Uncertainty metrics
        """
        # Generate bootstrap predictions
        bootstrap_preds = self.bootstrap_predictions(
            model, X_train, y_train, X_predict
        )
        
        # Calculate prediction intervals
        lower_bounds, upper_bounds = self.calculate_prediction_intervals(bootstrap_preds)
        prediction_intervals = np.column_stack([lower_bounds, upper_bounds])
        
        # Calculate prediction standard deviation
        prediction_std = bootstrap_preds.std(axis=0)
        
        # Calculate reliability scores
        reliability_scores = self.calculate_reliability_score(
            X_predict, X_train, prediction_std
        )
        
        # Calculate coverage if true values provided
        coverage = 0.0
        if y_true is not None:
            y_true_array = y_true.values if isinstance(y_true, pd.Series) else y_true
            within_interval = (y_true_array >= lower_bounds) & (y_true_array <= upper_bounds)
            coverage = within_interval.mean()
        
        return UncertaintyMetrics(
            prediction_intervals=prediction_intervals,
            prediction_std=prediction_std,
            reliability_scores=reliability_scores,
            coverage=coverage,
        )
    
    def predict_with_uncertainty(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame
    ) -> List[PredictionWithUncertainty]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_predict: Features to predict on
        
        Returns:
            List of predictions with uncertainty
        """
        # Get point predictions
        predictions = model.predict(X_predict)
        
        # Quantify uncertainty
        uncertainty_metrics = self.quantify_uncertainty(
            model, X_train, y_train, X_predict
        )
        
        # Create prediction objects
        results = []
        for i in range(len(X_predict)):
            results.append(PredictionWithUncertainty(
                prediction=predictions[i],
                uncertainty=uncertainty_metrics.prediction_std[i],
                confidence_interval=(
                    uncertainty_metrics.prediction_intervals[i, 0],
                    uncertainty_metrics.prediction_intervals[i, 1]
                ),
                reliability_score=uncertainty_metrics.reliability_scores[i],
            ))
        
        return results


class UncertaintyPropagator:
    """Propagate uncertainties through calculations."""
    
    @staticmethod
    def propagate_addition(
        values: List[float],
        uncertainties: List[float]
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty through addition: z = x + y.
        
        Args:
            values: List of values to add
            uncertainties: List of uncertainties
        
        Returns:
            Tuple of (result, uncertainty)
        """
        result = sum(values)
        # For addition: σ_z² = σ_x² + σ_y²
        uncertainty = np.sqrt(sum(u**2 for u in uncertainties))
        
        return result, uncertainty
    
    @staticmethod
    def propagate_multiplication(
        values: List[float],
        uncertainties: List[float]
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty through multiplication: z = x * y.
        
        Args:
            values: List of values to multiply
            uncertainties: List of uncertainties
        
        Returns:
            Tuple of (result, uncertainty)
        """
        result = np.prod(values)
        
        # For multiplication: (σ_z/z)² = (σ_x/x)² + (σ_y/y)²
        relative_uncertainties = [u / v for u, v in zip(uncertainties, values)]
        relative_uncertainty = np.sqrt(sum(ru**2 for ru in relative_uncertainties))
        uncertainty = abs(result * relative_uncertainty)
        
        return result, uncertainty
    
    @staticmethod
    def propagate_power(
        value: float,
        uncertainty: float,
        exponent: float
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty through power: z = x^n.
        
        Args:
            value: Base value
            uncertainty: Uncertainty in base
            exponent: Power exponent
        
        Returns:
            Tuple of (result, uncertainty)
        """
        result = value ** exponent
        
        # For power: σ_z/z = |n| * σ_x/x
        relative_uncertainty = abs(exponent) * (uncertainty / value)
        uncertainty_result = abs(result * relative_uncertainty)
        
        return result, uncertainty_result
    
    @staticmethod
    def propagate_function(
        func: callable,
        values: List[float],
        uncertainties: List[float],
        n_samples: int = 10000
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty through arbitrary function using Monte Carlo.
        
        Args:
            func: Function to propagate through
            values: Input values
            uncertainties: Input uncertainties
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Tuple of (result, uncertainty)
        """
        # Generate samples from normal distributions
        samples = []
        for value, uncertainty in zip(values, uncertainties):
            samples.append(np.random.normal(value, uncertainty, n_samples))
        
        # Evaluate function on samples
        results = func(*samples)
        
        # Calculate mean and standard deviation
        result = results.mean()
        uncertainty = results.std()
        
        return result, uncertainty


class ConfidenceIntervalReporter:
    """Report confidence intervals for predictions."""
    
    @staticmethod
    def format_prediction_with_ci(
        prediction: float,
        confidence_interval: Tuple[float, float],
        confidence_level: float = 0.95
    ) -> str:
        """
        Format prediction with confidence interval.
        
        Args:
            prediction: Point prediction
            confidence_interval: (lower, upper) bounds
            confidence_level: Confidence level
        
        Returns:
            Formatted string
        """
        lower, upper = confidence_interval
        ci_pct = int(confidence_level * 100)
        
        return f"{prediction:.2f} ({ci_pct}% CI: [{lower:.2f}, {upper:.2f}])"
    
    @staticmethod
    def create_prediction_report(
        predictions: List[PredictionWithUncertainty]
    ) -> pd.DataFrame:
        """
        Create report of predictions with uncertainties.
        
        Args:
            predictions: List of predictions with uncertainty
        
        Returns:
            DataFrame with prediction report
        """
        report_data = []
        
        for i, pred in enumerate(predictions):
            report_data.append({
                'sample_id': i,
                'prediction': pred.prediction,
                'uncertainty': pred.uncertainty,
                'ci_lower': pred.confidence_interval[0],
                'ci_upper': pred.confidence_interval[1],
                'ci_width': pred.confidence_interval[1] - pred.confidence_interval[0],
                'reliability_score': pred.reliability_score,
            })
        
        return pd.DataFrame(report_data)
