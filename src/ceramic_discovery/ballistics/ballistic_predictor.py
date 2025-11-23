"""V50 ballistic performance prediction engine with ML integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ceramic_discovery.ml.model_trainer import TrainedModel
from ceramic_discovery.ml.uncertainty_quantification import (
    UncertaintyQuantifier,
    PredictionWithUncertainty,
)


@dataclass
class BallisticPrediction:
    """Container for ballistic performance prediction with uncertainty."""
    
    v50: float  # Predicted V50 velocity (m/s)
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    uncertainty: float  # Standard deviation of prediction
    reliability_score: float  # 0-1, higher is more reliable
    physical_bounds_satisfied: bool  # Whether prediction is within physical bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'v50': self.v50,
            'confidence_interval': self.confidence_interval,
            'uncertainty': self.uncertainty,
            'reliability_score': self.reliability_score,
            'physical_bounds_satisfied': self.physical_bounds_satisfied,
        }
    
    def format_prediction(self) -> str:
        """Format prediction as human-readable string."""
        ci_lower, ci_upper = self.confidence_interval
        return (
            f"V50: {self.v50:.1f} m/s "
            f"(95% CI: [{ci_lower:.1f}, {ci_upper:.1f}] m/s, "
            f"reliability: {self.reliability_score:.2f})"
        )


@dataclass
class PropertyContribution:
    """Container for property contribution to ballistic performance."""
    
    property_name: str
    contribution_score: float  # Relative contribution (0-1)
    correlation: float  # Correlation with V50
    p_value: float  # Statistical significance
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if contribution is statistically significant."""
        return self.p_value < alpha


@dataclass
class MechanismAnalysis:
    """Container for ballistic performance mechanism analysis."""
    
    property_contributions: List[PropertyContribution]
    thermal_conductivity_impact: float  # Specific impact of thermal conductivity
    dominant_mechanism: str  # Primary performance driver
    correlation_matrix: pd.DataFrame  # Property correlations
    
    def get_top_contributors(self, n: int = 5) -> List[PropertyContribution]:
        """Get top N property contributors."""
        sorted_contributions = sorted(
            self.property_contributions,
            key=lambda x: abs(x.contribution_score),
            reverse=True
        )
        return sorted_contributions[:n]


class BallisticPredictor:
    """
    V50 ballistic performance prediction engine.
    
    Predicts V50 ballistic limit velocity from material properties using
    trained ML models with comprehensive uncertainty quantification.
    """
    
    # Physical bounds for V50 predictions (m/s)
    V50_MIN = 200.0  # Minimum realistic V50
    V50_MAX = 2000.0  # Maximum realistic V50
    
    # Primary features for V50 prediction (from design document)
    PRIMARY_FEATURES = [
        'hardness',
        'fracture_toughness',
        'density',
        'thermal_conductivity_1000C',
        'youngs_modulus',
    ]
    
    def __init__(
        self,
        model: Optional[TrainedModel] = None,
        uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
        random_seed: int = 42
    ):
        """
        Initialize ballistic predictor.
        
        Args:
            model: Trained ML model for V50 prediction
            uncertainty_quantifier: Uncertainty quantification system
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier or UncertaintyQuantifier(
            n_bootstrap=1000,
            confidence_level=0.95,
            random_seed=random_seed
        )
        self.random_seed = random_seed
        
        # Training data (stored for uncertainty quantification)
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        
        np.random.seed(random_seed)
    
    def load_model(self, model_path: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        self.model = TrainedModel.load(model_path)
    
    def set_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Set training data for uncertainty quantification.
        
        Args:
            X_train: Training features
            y_train: Training target (V50 values)
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def _check_physical_bounds(self, v50: float) -> bool:
        """
        Check if V50 prediction is within physical bounds.
        
        Args:
            v50: Predicted V50 value
        
        Returns:
            True if within bounds, False otherwise
        """
        return self.V50_MIN <= v50 <= self.V50_MAX
    
    def _enforce_physical_bounds(self, v50: float) -> float:
        """
        Enforce physical bounds on V50 prediction.
        
        Args:
            v50: Predicted V50 value
        
        Returns:
            V50 value clipped to physical bounds
        """
        return np.clip(v50, self.V50_MIN, self.V50_MAX)
    
    def _calculate_reliability_score(
        self,
        material_properties: pd.DataFrame,
        prediction_uncertainty: float
    ) -> float:
        """
        Calculate reliability score for prediction.
        
        Reliability is based on:
        1. Distance to training data
        2. Prediction uncertainty
        3. Data quality (completeness of input properties)
        
        Args:
            material_properties: Material properties for prediction
            prediction_uncertainty: Uncertainty of prediction
        
        Returns:
            Reliability score (0-1)
        """
        if self.X_train is None:
            # If no training data, use uncertainty only
            max_uncertainty = 200.0  # Typical max uncertainty for V50
            uncertainty_score = 1.0 - min(prediction_uncertainty / max_uncertainty, 1.0)
            return uncertainty_score
        
        # Calculate distance to training data
        from scipy.spatial.distance import cdist
        
        distances = cdist(
            material_properties[self.X_train.columns],
            self.X_train,
            metric='euclidean'
        )
        min_distance = distances.min()
        
        # Normalize distance
        typical_distance = 10.0  # Typical distance in feature space
        distance_score = np.exp(-min_distance / typical_distance)
        
        # Normalize uncertainty
        max_uncertainty = 200.0
        uncertainty_score = 1.0 - min(prediction_uncertainty / max_uncertainty, 1.0)
        
        # Check data quality (completeness)
        n_features = len(self.PRIMARY_FEATURES)
        n_available = sum(
            1 for feat in self.PRIMARY_FEATURES
            if feat in material_properties.columns and not pd.isna(material_properties[feat].iloc[0])
        )
        data_quality_score = n_available / n_features
        
        # Combine scores
        reliability = (
            0.4 * distance_score +
            0.4 * uncertainty_score +
            0.2 * data_quality_score
        )
        
        return np.clip(reliability, 0.0, 1.0)
    
    def predict_v50(
        self,
        material_properties: pd.DataFrame,
        enforce_bounds: bool = True
    ) -> BallisticPrediction:
        """
        Predict V50 ballistic limit velocity with uncertainty.
        
        Args:
            material_properties: DataFrame with material properties
            enforce_bounds: Whether to enforce physical bounds on prediction
        
        Returns:
            Ballistic prediction with uncertainty estimates
        
        Raises:
            ValueError: If model is not loaded or training data not set
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not set. Call set_training_data() first.")
        
        # Make prediction with uncertainty
        prediction_with_uncertainty = self.uncertainty_quantifier.predict_with_uncertainty(
            self.model.model,
            self.X_train,
            self.y_train,
            material_properties
        )[0]  # Get first prediction
        
        # Extract values
        v50 = prediction_with_uncertainty.prediction
        confidence_interval = prediction_with_uncertainty.confidence_interval
        uncertainty = prediction_with_uncertainty.uncertainty
        
        # Check physical bounds
        bounds_satisfied = self._check_physical_bounds(v50)
        
        # Enforce bounds if requested
        if enforce_bounds:
            v50 = self._enforce_physical_bounds(v50)
            confidence_interval = (
                self._enforce_physical_bounds(confidence_interval[0]),
                self._enforce_physical_bounds(confidence_interval[1])
            )
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(
            material_properties,
            uncertainty
        )
        
        return BallisticPrediction(
            v50=v50,
            confidence_interval=confidence_interval,
            uncertainty=uncertainty,
            reliability_score=reliability_score,
            physical_bounds_satisfied=bounds_satisfied,
        )
    
    def predict_batch(
        self,
        materials: pd.DataFrame,
        enforce_bounds: bool = True
    ) -> List[BallisticPrediction]:
        """
        Predict V50 for multiple materials.
        
        Args:
            materials: DataFrame with material properties (one row per material)
            enforce_bounds: Whether to enforce physical bounds
        
        Returns:
            List of ballistic predictions
        """
        predictions = []
        
        for idx in range(len(materials)):
            material = materials.iloc[[idx]]
            prediction = self.predict_v50(material, enforce_bounds)
            predictions.append(prediction)
        
        return predictions
    
    def analyze_mechanisms(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series
    ) -> MechanismAnalysis:
        """
        Analyze mechanisms driving ballistic performance.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
        
        Returns:
            Mechanism analysis results
        """
        # Calculate property contributions using feature importance
        property_contributions = []
        
        if self.model and self.model.feature_importances is not None:
            for feature, importance in zip(
                self.model.feature_names,
                self.model.feature_importances
            ):
                # Calculate correlation with V50
                if feature in materials.columns:
                    correlation, p_value = stats.pearsonr(
                        materials[feature].dropna(),
                        v50_values[materials[feature].notna()]
                    )
                else:
                    correlation, p_value = 0.0, 1.0
                
                property_contributions.append(PropertyContribution(
                    property_name=feature,
                    contribution_score=importance,
                    correlation=correlation,
                    p_value=p_value,
                ))
        else:
            # Fallback: use correlations only
            for feature in materials.columns:
                if feature in materials.columns and materials[feature].notna().sum() > 2:
                    correlation, p_value = stats.pearsonr(
                        materials[feature].dropna(),
                        v50_values[materials[feature].notna()]
                    )
                    
                    property_contributions.append(PropertyContribution(
                        property_name=feature,
                        contribution_score=abs(correlation),
                        correlation=correlation,
                        p_value=p_value,
                    ))
        
        # Calculate thermal conductivity specific impact
        thermal_conductivity_impact = 0.0
        if 'thermal_conductivity_1000C' in materials.columns:
            thermal_feature = next(
                (pc for pc in property_contributions if pc.property_name == 'thermal_conductivity_1000C'),
                None
            )
            if thermal_feature:
                thermal_conductivity_impact = thermal_feature.contribution_score
        
        # Identify dominant mechanism
        if property_contributions:
            dominant = max(property_contributions, key=lambda x: x.contribution_score)
            dominant_mechanism = dominant.property_name
        else:
            dominant_mechanism = "unknown"
        
        # Calculate correlation matrix
        correlation_matrix = materials.corr()
        
        return MechanismAnalysis(
            property_contributions=property_contributions,
            thermal_conductivity_impact=thermal_conductivity_impact,
            dominant_mechanism=dominant_mechanism,
            correlation_matrix=correlation_matrix,
        )
    
    def validate_prediction_accuracy(
        self,
        materials: pd.DataFrame,
        true_v50: pd.Series
    ) -> Dict[str, float]:
        """
        Validate prediction accuracy against true V50 values.
        
        Args:
            materials: DataFrame with material properties
            true_v50: Series with true V50 values
        
        Returns:
            Dictionary with accuracy metrics
        """
        predictions = self.predict_batch(materials)
        predicted_v50 = np.array([p.v50 for p in predictions])
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        r2 = r2_score(true_v50, predicted_v50)
        rmse = np.sqrt(mean_squared_error(true_v50, predicted_v50))
        mae = mean_absolute_error(true_v50, predicted_v50)
        
        # Calculate coverage (fraction within confidence intervals)
        within_ci = sum(
            1 for i, pred in enumerate(predictions)
            if pred.confidence_interval[0] <= true_v50.iloc[i] <= pred.confidence_interval[1]
        )
        coverage = within_ci / len(predictions)
        
        # Calculate mean reliability score
        mean_reliability = np.mean([p.reliability_score for p in predictions])
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'mean_reliability': mean_reliability,
        }
