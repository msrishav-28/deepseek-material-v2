"""Feature engineering pipeline for ML models with Tier 1-2 property separation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    
    name: str
    tier: int  # 1, 2, or 3
    category: str  # 'structural', 'thermodynamic', 'electronic', 'mechanical', 'thermal'
    unit: str
    is_fundamental: bool  # True for Tier 1-2, False for Tier 3 (derived)
    
    def __post_init__(self):
        """Validate feature metadata."""
        if self.tier not in [1, 2, 3]:
            raise ValueError(f"Invalid tier {self.tier}. Must be 1, 2, or 3.")
        
        # Tier 1-2 are fundamental, Tier 3 are derived
        expected_fundamental = self.tier in [1, 2]
        if self.is_fundamental != expected_fundamental:
            raise ValueError(
                f"Feature {self.name} tier {self.tier} fundamental status mismatch. "
                f"Expected {expected_fundamental}, got {self.is_fundamental}"
            )


@dataclass
class FeatureSet:
    """Container for feature data with metadata."""
    
    features: pd.DataFrame
    metadata: Dict[str, FeatureMetadata]
    scaler: Optional[StandardScaler] = None
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.columns)
    
    def get_tier_1_features(self) -> List[str]:
        """Get Tier 1 feature names."""
        return [name for name, meta in self.metadata.items() if meta.tier == 1]
    
    def get_tier_2_features(self) -> List[str]:
        """Get Tier 2 feature names."""
        return [name for name, meta in self.metadata.items() if meta.tier == 2]
    
    def get_fundamental_features(self) -> List[str]:
        """Get fundamental (Tier 1-2) feature names."""
        return [name for name, meta in self.metadata.items() if meta.is_fundamental]


class FeatureValidator:
    """Validate features to prevent circular dependencies and ensure correctness."""
    
    # Define which properties are fundamental (Tier 1-2) vs derived (Tier 3)
    TIER_1_PROPERTIES = {
        # Structural
        'lattice_a', 'lattice_b', 'lattice_c',
        'lattice_alpha', 'lattice_beta', 'lattice_gamma',
        'volume', 'density', 'space_group_number', 'nsites',
        # Thermodynamic
        'formation_energy_per_atom', 'energy_above_hull',
        'decomposition_energy', 'is_stable',
        # Electronic
        'band_gap', 'is_metal', 'is_magnetic',
        'total_magnetization', 'efermi',
    }
    
    TIER_2_PROPERTIES = {
        # Mechanical
        'bulk_modulus_vrh', 'shear_modulus_vrh', 'youngs_modulus',
        'poisson_ratio', 'elastic_anisotropy', 'hardness_vickers',
        'fracture_toughness', 'c11', 'c12', 'c44',
        # Thermal
        'thermal_conductivity_300K', 'thermal_conductivity_1000K',
        'thermal_expansion_300K', 'specific_heat_300K',
        'debye_temperature', 'melting_point',
        'thermal_diffusivity', 'phonon_band_gap',
        # Optical
        'refractive_index', 'dielectric_constant',
        'absorption_coefficient', 'reflectivity', 'transparency_range',
        # Chemical
        'oxidation_resistance', 'corrosion_resistance',
        # Microstructural
        'grain_size', 'porosity',
    }
    
    TIER_3_PROPERTIES = {
        # Derived ballistic properties - MUST NOT be used as ML inputs
        'v50_predicted', 'areal_density', 'specific_energy_absorption',
        'penetration_resistance', 'multi_hit_capability',
        'spall_resistance', 'shock_impedance',
        # Economic/manufacturing
        'cost_per_kg', 'availability_score',
        'manufacturability_score', 'environmental_impact',
    }
    
    def __init__(self):
        """Initialize feature validator."""
        self.fundamental_properties = self.TIER_1_PROPERTIES | self.TIER_2_PROPERTIES
        self.derived_properties = self.TIER_3_PROPERTIES
    
    def validate_feature_set(self, feature_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that feature set contains only fundamental properties.
        
        Args:
            feature_names: List of feature names to validate
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        
        for feature in feature_names:
            if feature in self.derived_properties:
                violations.append(
                    f"Feature '{feature}' is a Tier 3 derived property and creates "
                    f"circular dependency. Only Tier 1-2 fundamental properties allowed."
                )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def check_circular_dependencies(
        self,
        feature_names: List[str],
        target_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Check for circular dependencies between features and target.
        
        Args:
            feature_names: List of feature names
            target_name: Name of target variable
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        
        # If target is a derived property, features must be fundamental
        if target_name in self.derived_properties:
            for feature in feature_names:
                if feature in self.derived_properties:
                    violations.append(
                        f"Circular dependency: Feature '{feature}' and target '{target_name}' "
                        f"are both derived properties"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_feature_tier(self, feature_name: str) -> int:
        """Get tier level for a feature."""
        if feature_name in self.TIER_1_PROPERTIES:
            return 1
        elif feature_name in self.TIER_2_PROPERTIES:
            return 2
        elif feature_name in self.TIER_3_PROPERTIES:
            return 3
        else:
            warnings.warn(f"Unknown feature '{feature_name}', assuming Tier 2")
            return 2


class FeatureEngineeringPipeline:
    """Feature engineering pipeline with validation and scaling."""
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        handle_missing: str = 'drop',
        min_feature_variance: float = 1e-6
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            scaling_method: 'standard' (z-score) or 'minmax' (0-1 range)
            handle_missing: 'drop' (remove samples) or 'median' (impute with median)
            min_feature_variance: Minimum variance threshold for feature selection
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.min_feature_variance = min_feature_variance
        
        self.validator = FeatureValidator()
        self.scaler: Optional[StandardScaler] = None
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self.selected_features: List[str] = []
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_name: Optional[str] = None
    ) -> 'FeatureEngineeringPipeline':
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Feature dataframe
            y: Optional target variable
            target_name: Optional name of target variable for validation
        
        Returns:
            Self for method chaining
        """
        # Validate feature set
        is_valid, violations = self.validator.validate_feature_set(X.columns.tolist())
        if not is_valid:
            raise ValueError(
                f"Invalid feature set:\n" + "\n".join(violations)
            )
        
        # Check circular dependencies if target provided
        if target_name is not None:
            is_valid, violations = self.validator.check_circular_dependencies(
                X.columns.tolist(), target_name
            )
            if not is_valid:
                raise ValueError(
                    f"Circular dependencies detected:\n" + "\n".join(violations)
                )
        
        # Build feature metadata
        self.feature_metadata = {}
        for col in X.columns:
            tier = self.validator.get_feature_tier(col)
            self.feature_metadata[col] = FeatureMetadata(
                name=col,
                tier=tier,
                category='unknown',  # Would be populated from property definitions
                unit='unknown',
                is_fundamental=(tier in [1, 2])
            )
        
        # Handle missing values
        X_processed = self._handle_missing_values(X)
        
        # Select features based on variance
        X_processed = self._select_features_by_variance(X_processed)
        
        # Fit scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        self.scaler.fit(X_processed)
        
        # Store feature statistics
        self.feature_means = dict(zip(X_processed.columns, X_processed.mean()))
        self.feature_stds = dict(zip(X_processed.columns, X_processed.std()))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Feature dataframe
        
        Returns:
            Transformed feature dataframe
        """
        if self.scaler is None:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Handle missing values
        X_processed = self._handle_missing_values(X)
        
        # Select same features as during fit
        X_processed = X_processed[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return pd.DataFrame(
            X_scaled,
            columns=self.selected_features,
            index=X_processed.index
        )
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y, target_name).transform(X)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategy."""
        if self.handle_missing == 'drop':
            # Drop rows with any missing values
            X_clean = X.dropna()
            if len(X_clean) < len(X):
                warnings.warn(
                    f"Dropped {len(X) - len(X_clean)} rows with missing values"
                )
            return X_clean
        
        elif self.handle_missing == 'median':
            # Impute with median
            X_imputed = X.copy()
            for col in X.columns:
                if X[col].isna().any():
                    median_val = X[col].median()
                    n_missing = X[col].isna().sum()
                    X_imputed[col] = X_imputed[col].fillna(median_val)
                    warnings.warn(
                        f"Imputed {n_missing} missing values in '{col}' "
                        f"with median {median_val:.4f}"
                    )
            return X_imputed
        
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")
    
    def _select_features_by_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features with sufficient variance."""
        variances = X.var()
        low_variance_features = variances[variances < self.min_feature_variance].index.tolist()
        
        if low_variance_features:
            warnings.warn(
                f"Removing {len(low_variance_features)} low-variance features: "
                f"{low_variance_features}"
            )
            X_selected = X.drop(columns=low_variance_features)
        else:
            X_selected = X
        
        self.selected_features = X_selected.columns.tolist()
        return X_selected
    
    def get_feature_importance_analysis(
        self,
        feature_importances: np.ndarray,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze feature importance from trained model.
        
        Args:
            feature_importances: Array of feature importances from model
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance analysis
        """
        if len(feature_importances) != len(self.selected_features):
            raise ValueError(
                f"Feature importance length {len(feature_importances)} does not match "
                f"number of features {len(self.selected_features)}"
            )
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': feature_importances,
            'tier': [self.feature_metadata[f].tier for f in self.selected_features],
            'is_fundamental': [self.feature_metadata[f].is_fundamental for f in self.selected_features]
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def validate_physical_bounds(
        self,
        X: pd.DataFrame,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that features are within physical bounds.
        
        Args:
            X: Feature dataframe
            bounds: Optional dict mapping feature names to (min, max) bounds
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        if bounds is None:
            # Default physical bounds for common properties
            bounds = {
                'density': (1.0, 20.0),  # g/cm³
                'hardness_vickers': (0.0, 50.0),  # GPa
                'fracture_toughness': (0.0, 15.0),  # MPa·m^0.5
                'youngs_modulus': (0.0, 1000.0),  # GPa
                'thermal_conductivity_300K': (0.0, 500.0),  # W/m·K
                'thermal_conductivity_1000K': (0.0, 500.0),  # W/m·K
                'band_gap': (0.0, 15.0),  # eV
                'melting_point': (0.0, 5000.0),  # K
            }
        
        violations = []
        
        for feature in X.columns:
            if feature in bounds:
                min_val, max_val = bounds[feature]
                out_of_bounds = (X[feature] < min_val) | (X[feature] > max_val)
                
                if out_of_bounds.any():
                    n_violations = out_of_bounds.sum()
                    violations.append(
                        f"Feature '{feature}' has {n_violations} values out of bounds "
                        f"[{min_val}, {max_val}]"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """Get statistics for all features."""
        if not self.feature_means:
            raise ValueError("Pipeline must be fitted first")
        
        stats_df = pd.DataFrame({
            'feature': list(self.feature_means.keys()),
            'mean': list(self.feature_means.values()),
            'std': list(self.feature_stds.values()),
            'tier': [self.feature_metadata[f].tier for f in self.feature_means.keys()],
        })
        
        return stats_df
