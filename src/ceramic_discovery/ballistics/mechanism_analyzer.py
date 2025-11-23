"""Mechanism analysis system for ballistic performance."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr


@dataclass
class ThermalConductivityAnalysis:
    """Analysis of thermal conductivity impact on ballistic performance."""
    
    correlation_with_v50: float
    p_value: float
    contribution_score: float
    mechanism_description: str
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if thermal conductivity impact is statistically significant."""
        return self.p_value < alpha


@dataclass
class PropertyInteraction:
    """Analysis of interaction between two properties."""
    
    property1: str
    property2: str
    interaction_strength: float  # Correlation between properties
    combined_effect_on_v50: float  # Combined contribution to V50
    synergy_score: float  # Positive = synergistic, negative = antagonistic
    
    def is_synergistic(self) -> bool:
        """Check if properties have synergistic effect."""
        return self.synergy_score > 0.1


@dataclass
class HypothesisTest:
    """Statistical hypothesis test result."""
    
    hypothesis: str
    test_statistic: float
    p_value: float
    conclusion: str
    confidence_level: float = 0.95
    
    def is_significant(self) -> bool:
        """Check if hypothesis test is significant."""
        alpha = 1 - self.confidence_level
        return self.p_value < alpha


class MechanismAnalyzer:
    """
    Advanced mechanism analysis system for ballistic performance.
    
    Analyzes the physical mechanisms driving ballistic performance through:
    - Thermal conductivity impact analysis
    - Property contribution decomposition
    - Correlation analysis between properties and performance
    - Hypothesis testing for performance mechanisms
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize mechanism analyzer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def analyze_thermal_conductivity_impact(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series
    ) -> ThermalConductivityAnalysis:
        """
        Analyze the specific impact of thermal conductivity on ballistic performance.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
        
        Returns:
            Thermal conductivity analysis results
        """
        if 'thermal_conductivity_1000C' not in materials.columns:
            return ThermalConductivityAnalysis(
                correlation_with_v50=0.0,
                p_value=1.0,
                contribution_score=0.0,
                mechanism_description="Thermal conductivity data not available",
            )
        
        # Get thermal conductivity values
        thermal_cond = materials['thermal_conductivity_1000C'].dropna()
        v50_aligned = v50_values[thermal_cond.index]
        
        if len(thermal_cond) < 3:
            return ThermalConductivityAnalysis(
                correlation_with_v50=0.0,
                p_value=1.0,
                contribution_score=0.0,
                mechanism_description="Insufficient data for analysis",
            )
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(thermal_cond, v50_aligned)
        
        # Calculate contribution using partial correlation
        # (controlling for other properties)
        other_properties = [col for col in materials.columns 
                          if col != 'thermal_conductivity_1000C' and materials[col].notna().sum() > 2]
        
        if other_properties:
            # Multiple regression to get contribution
            from sklearn.linear_model import LinearRegression
            
            X = materials[other_properties + ['thermal_conductivity_1000C']].dropna()
            y = v50_values[X.index]
            
            if len(X) > len(X.columns):
                model = LinearRegression()
                model.fit(X, y)
                
                # Get coefficient for thermal conductivity
                thermal_idx = list(X.columns).index('thermal_conductivity_1000C')
                thermal_coef = model.coef_[thermal_idx]
                
                # Normalize contribution
                total_contribution = np.sum(np.abs(model.coef_))
                contribution_score = abs(thermal_coef) / total_contribution if total_contribution > 0 else 0
            else:
                contribution_score = abs(correlation)
        else:
            contribution_score = abs(correlation)
        
        # Generate mechanism description
        if correlation > 0.3:
            mechanism = (
                "High thermal conductivity enhances ballistic performance by "
                "rapidly dissipating impact energy through phonon transport, "
                "reducing localized heating and preventing thermal softening."
            )
        elif correlation < -0.3:
            mechanism = (
                "Lower thermal conductivity may improve performance by "
                "localizing energy dissipation and promoting controlled fracture."
            )
        else:
            mechanism = (
                "Thermal conductivity shows weak correlation with ballistic performance, "
                "suggesting other mechanisms dominate."
            )
        
        return ThermalConductivityAnalysis(
            correlation_with_v50=correlation,
            p_value=p_value,
            contribution_score=contribution_score,
            mechanism_description=mechanism,
        )
    
    def analyze_property_contributions(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series,
        feature_importances: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze individual property contributions to ballistic performance.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
            feature_importances: Optional ML feature importances
            feature_names: Optional feature names corresponding to importances
        
        Returns:
            DataFrame with property contribution analysis
        """
        contributions = []
        
        for property_name in materials.columns:
            if materials[property_name].notna().sum() < 3:
                continue
            
            # Get aligned data
            prop_values = materials[property_name].dropna()
            v50_aligned = v50_values[prop_values.index]
            
            # Calculate correlation
            pearson_corr, pearson_p = stats.pearsonr(prop_values, v50_aligned)
            spearman_corr, spearman_p = spearmanr(prop_values, v50_aligned)
            
            # Get feature importance if available
            importance = 0.0
            if feature_importances is not None and feature_names is not None:
                if property_name in feature_names:
                    idx = feature_names.index(property_name)
                    importance = feature_importances[idx]
            
            # Calculate effect size (standardized coefficient)
            from sklearn.linear_model import LinearRegression
            X = prop_values.values.reshape(-1, 1)
            y = v50_aligned.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Standardized coefficient
            std_x = np.std(X)
            std_y = np.std(y)
            standardized_coef = model.coef_[0] * (std_x / std_y) if std_y > 0 else 0
            
            contributions.append({
                'property': property_name,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'feature_importance': importance,
                'standardized_coefficient': standardized_coef,
                'effect_size': abs(standardized_coef),
                'significant': pearson_p < 0.05,
            })
        
        df = pd.DataFrame(contributions)
        df = df.sort_values('effect_size', ascending=False)
        
        return df
    
    def analyze_property_interactions(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series,
        top_n: int = 5
    ) -> List[PropertyInteraction]:
        """
        Analyze interactions between properties and their combined effect on V50.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
            top_n: Number of top property pairs to analyze
        
        Returns:
            List of property interaction analyses
        """
        from sklearn.linear_model import LinearRegression
        from itertools import combinations
        
        # Get properties with sufficient data
        valid_properties = [
            col for col in materials.columns
            if materials[col].notna().sum() > 5
        ]
        
        if len(valid_properties) < 2:
            return []
        
        interactions = []
        
        # Analyze all pairs of properties
        for prop1, prop2 in combinations(valid_properties, 2):
            # Get complete cases
            data = materials[[prop1, prop2]].dropna()
            v50_aligned = v50_values[data.index]
            
            if len(data) < 5:
                continue
            
            # Calculate correlation between properties
            interaction_strength, _ = stats.pearsonr(data[prop1], data[prop2])
            
            # Fit model with both properties
            X = data[[prop1, prop2]].values
            y = v50_aligned.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate combined effect
            combined_effect = np.sum(np.abs(model.coef_))
            
            # Calculate synergy score
            # Fit individual models
            model1 = LinearRegression()
            model1.fit(data[[prop1]].values, y)
            effect1 = abs(model1.coef_[0])
            
            model2 = LinearRegression()
            model2.fit(data[[prop2]].values, y)
            effect2 = abs(model2.coef_[0])
            
            # Synergy: combined effect vs sum of individual effects
            expected_effect = effect1 + effect2
            synergy_score = (combined_effect - expected_effect) / expected_effect if expected_effect > 0 else 0
            
            interactions.append(PropertyInteraction(
                property1=prop1,
                property2=prop2,
                interaction_strength=interaction_strength,
                combined_effect_on_v50=combined_effect,
                synergy_score=synergy_score,
            ))
        
        # Sort by synergy score (most synergistic first)
        interactions.sort(key=lambda x: abs(x.synergy_score), reverse=True)
        
        return interactions[:top_n]
    
    def test_performance_hypotheses(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series
    ) -> List[HypothesisTest]:
        """
        Test statistical hypotheses about performance mechanisms.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
        
        Returns:
            List of hypothesis test results
        """
        hypotheses = []
        
        # Hypothesis 1: Hardness positively correlates with V50
        if 'hardness' in materials.columns:
            hardness = materials['hardness'].dropna()
            v50_aligned = v50_values[hardness.index]
            
            if len(hardness) >= 3:
                corr, p_value = stats.pearsonr(hardness, v50_aligned)
                
                hypotheses.append(HypothesisTest(
                    hypothesis="Hardness positively correlates with V50",
                    test_statistic=corr,
                    p_value=p_value,
                    conclusion=(
                        f"Hardness {'significantly' if p_value < 0.05 else 'does not significantly'} "
                        f"correlate with V50 (r={corr:.3f}, p={p_value:.3f})"
                    ),
                ))
        
        # Hypothesis 2: Fracture toughness positively correlates with V50
        if 'fracture_toughness' in materials.columns:
            toughness = materials['fracture_toughness'].dropna()
            v50_aligned = v50_values[toughness.index]
            
            if len(toughness) >= 3:
                corr, p_value = stats.pearsonr(toughness, v50_aligned)
                
                hypotheses.append(HypothesisTest(
                    hypothesis="Fracture toughness positively correlates with V50",
                    test_statistic=corr,
                    p_value=p_value,
                    conclusion=(
                        f"Fracture toughness {'significantly' if p_value < 0.05 else 'does not significantly'} "
                        f"correlate with V50 (r={corr:.3f}, p={p_value:.3f})"
                    ),
                ))
        
        # Hypothesis 3: Thermal conductivity affects V50
        if 'thermal_conductivity_1000C' in materials.columns:
            thermal = materials['thermal_conductivity_1000C'].dropna()
            v50_aligned = v50_values[thermal.index]
            
            if len(thermal) >= 3:
                corr, p_value = stats.pearsonr(thermal, v50_aligned)
                
                hypotheses.append(HypothesisTest(
                    hypothesis="Thermal conductivity affects V50",
                    test_statistic=corr,
                    p_value=p_value,
                    conclusion=(
                        f"Thermal conductivity {'significantly' if p_value < 0.05 else 'does not significantly'} "
                        f"affect V50 (r={corr:.3f}, p={p_value:.3f})"
                    ),
                ))
        
        # Hypothesis 4: Density affects V50
        if 'density' in materials.columns:
            density = materials['density'].dropna()
            v50_aligned = v50_values[density.index]
            
            if len(density) >= 3:
                corr, p_value = stats.pearsonr(density, v50_aligned)
                
                hypotheses.append(HypothesisTest(
                    hypothesis="Density affects V50",
                    test_statistic=corr,
                    p_value=p_value,
                    conclusion=(
                        f"Density {'significantly' if p_value < 0.05 else 'does not significantly'} "
                        f"affect V50 (r={corr:.3f}, p={p_value:.3f})"
                    ),
                ))
        
        # Hypothesis 5: High hardness + high toughness materials perform better
        if 'hardness' in materials.columns and 'fracture_toughness' in materials.columns:
            data = materials[['hardness', 'fracture_toughness']].dropna()
            v50_aligned = v50_values[data.index]
            
            if len(data) >= 5:
                # Split into high/low groups based on median
                median_hardness = data['hardness'].median()
                median_toughness = data['fracture_toughness'].median()
                
                high_both = (data['hardness'] >= median_hardness) & (data['fracture_toughness'] >= median_toughness)
                low_both = (data['hardness'] < median_hardness) & (data['fracture_toughness'] < median_toughness)
                
                if high_both.sum() >= 2 and low_both.sum() >= 2:
                    v50_high = v50_aligned[high_both]
                    v50_low = v50_aligned[low_both]
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(v50_high, v50_low)
                    
                    hypotheses.append(HypothesisTest(
                        hypothesis="Materials with high hardness AND high toughness have higher V50",
                        test_statistic=t_stat,
                        p_value=p_value,
                        conclusion=(
                            f"High hardness+toughness materials {'significantly' if p_value < 0.05 else 'do not significantly'} "
                            f"outperform low hardness+toughness materials (t={t_stat:.3f}, p={p_value:.3f})"
                        ),
                    ))
        
        return hypotheses
    
    def generate_mechanism_report(
        self,
        materials: pd.DataFrame,
        v50_values: pd.Series,
        feature_importances: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive mechanism analysis report.
        
        Args:
            materials: DataFrame with material properties
            v50_values: Series with V50 values
            feature_importances: Optional ML feature importances
            feature_names: Optional feature names
        
        Returns:
            Dictionary with complete mechanism analysis
        """
        report = {}
        
        # Thermal conductivity analysis
        report['thermal_conductivity_analysis'] = self.analyze_thermal_conductivity_impact(
            materials, v50_values
        )
        
        # Property contributions
        report['property_contributions'] = self.analyze_property_contributions(
            materials, v50_values, feature_importances, feature_names
        )
        
        # Property interactions
        report['property_interactions'] = self.analyze_property_interactions(
            materials, v50_values
        )
        
        # Hypothesis tests
        report['hypothesis_tests'] = self.test_performance_hypotheses(
            materials, v50_values
        )
        
        # Summary statistics
        report['summary'] = {
            'n_materials': len(materials),
            'n_properties': len(materials.columns),
            'v50_mean': v50_values.mean(),
            'v50_std': v50_values.std(),
            'v50_range': (v50_values.min(), v50_values.max()),
        }
        
        return report
