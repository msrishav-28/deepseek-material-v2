"""Composite materials property estimation with explicit limitations."""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
import math

from .ceramic_system import CeramicSystem, ValueWithUncertainty


@dataclass
class CompositeSystem:
    """Composite ceramic system with two components."""
    
    component1: CeramicSystem
    component2: CeramicSystem
    volume_fraction1: float  # Volume fraction of component 1
    volume_fraction2: float  # Volume fraction of component 2
    
    def __post_init__(self):
        """Validate composite system."""
        # Check volume fractions sum to 1
        total = self.volume_fraction1 + self.volume_fraction2
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Volume fractions must sum to 1.0, got {total}"
            )
        
        # Check volume fractions are positive
        if self.volume_fraction1 <= 0 or self.volume_fraction2 <= 0:
            raise ValueError("Volume fractions must be positive")
    
    def get_composition_string(self) -> str:
        """Get human-readable composition string."""
        pct1 = self.volume_fraction1 * 100
        pct2 = self.volume_fraction2 * 100
        comp1 = self.component1.get_composition_string()
        comp2 = self.component2.get_composition_string()
        return f"{comp1}({pct1:.0f}%)-{comp2}({pct2:.0f}%)"


class CompositeCalculator:
    """
    Calculate composite material properties using mixing rules.
    
    IMPORTANT LIMITATIONS:
    - Rule-of-mixtures provides ESTIMATES only
    - Assumes ideal mixing (no interfacial effects)
    - Does not account for microstructure
    - Does not account for processing effects
    - Uncertainty increases significantly for composites
    - Experimental validation strongly recommended
    """
    
    # Known composite systems for validation
    KNOWN_COMPOSITES = {
        "SiC-B₄C": {
            "ratios": [(0.7, 0.3)],
            "properties": {
                "hardness": {"expected": 29.0, "tolerance": 3.0},  # GPa
                "density": {"expected": 3.0, "tolerance": 0.2},  # g/cm³
            }
        },
        "SiC-TiC": {
            "ratios": [(0.8, 0.2)],
            "properties": {
                "hardness": {"expected": 28.5, "tolerance": 3.0},  # GPa
                "density": {"expected": 3.5, "tolerance": 0.3},  # g/cm³
            }
        },
        "Al₂O₃-SiC": {
            "ratios": [(0.5, 0.5)],
            "properties": {
                "hardness": {"expected": 24.0, "tolerance": 3.0},  # GPa
                "density": {"expected": 3.6, "tolerance": 0.2},  # g/cm³
            }
        }
    }
    
    def __init__(self):
        """Initialize composite calculator."""
        self._warnings_issued = []
    
    def _issue_limitation_warning(self, message: str) -> None:
        """Issue warning about composite modeling limitations."""
        if message not in self._warnings_issued:
            warnings.warn(
                f"COMPOSITE LIMITATION: {message}\n"
                "Rule-of-mixtures provides estimates only. "
                "Experimental validation strongly recommended.",
                UserWarning,
                stacklevel=3
            )
            self._warnings_issued.append(message)
    
    def calculate_rule_of_mixtures(
        self,
        value1: float,
        value2: float,
        fraction1: float,
        fraction2: float
    ) -> float:
        """
        Calculate property using simple rule of mixtures (linear).
        
        property = value1 * fraction1 + value2 * fraction2
        
        Args:
            value1: Property value for component 1
            value2: Property value for component 2
            fraction1: Volume fraction of component 1
            fraction2: Volume fraction of component 2
        
        Returns:
            Estimated composite property value
        """
        return value1 * fraction1 + value2 * fraction2
    
    def calculate_inverse_rule_of_mixtures(
        self,
        value1: float,
        value2: float,
        fraction1: float,
        fraction2: float
    ) -> float:
        """
        Calculate property using inverse rule of mixtures (harmonic mean).
        
        Used for properties like thermal conductivity in series arrangement.
        
        1/property = fraction1/value1 + fraction2/value2
        
        Args:
            value1: Property value for component 1
            value2: Property value for component 2
            fraction1: Volume fraction of component 1
            fraction2: Volume fraction of component 2
        
        Returns:
            Estimated composite property value
        """
        if value1 <= 0 or value2 <= 0:
            raise ValueError("Values must be positive for inverse rule of mixtures")
        
        inverse_property = fraction1 / value1 + fraction2 / value2
        return 1.0 / inverse_property
    
    def calculate_geometric_mean(
        self,
        value1: float,
        value2: float,
        fraction1: float,
        fraction2: float
    ) -> float:
        """
        Calculate property using geometric mean.
        
        property = value1^fraction1 * value2^fraction2
        
        Args:
            value1: Property value for component 1
            value2: Property value for component 2
            fraction1: Volume fraction of component 1
            fraction2: Volume fraction of component 2
        
        Returns:
            Estimated composite property value
        """
        if value1 <= 0 or value2 <= 0:
            raise ValueError("Values must be positive for geometric mean")
        
        return (value1 ** fraction1) * (value2 ** fraction2)
    
    def propagate_uncertainty(
        self,
        uncertainty1: float,
        uncertainty2: float,
        fraction1: float,
        fraction2: float,
        method: str = "linear"
    ) -> float:
        """
        Propagate uncertainties through mixing rule.
        
        Args:
            uncertainty1: Uncertainty for component 1
            uncertainty2: Uncertainty for component 2
            fraction1: Volume fraction of component 1
            fraction2: Volume fraction of component 2
            method: Mixing method ("linear", "inverse", "geometric")
        
        Returns:
            Propagated uncertainty
        """
        if method == "linear":
            # Linear propagation: σ² = (f1*σ1)² + (f2*σ2)²
            variance = (fraction1 * uncertainty1) ** 2 + (fraction2 * uncertainty2) ** 2
            base_uncertainty = math.sqrt(variance)
        elif method == "inverse":
            # More complex for inverse rule
            # Approximate with linear propagation
            variance = (fraction1 * uncertainty1) ** 2 + (fraction2 * uncertainty2) ** 2
            base_uncertainty = math.sqrt(variance)
        elif method == "geometric":
            # Approximate with linear propagation
            variance = (fraction1 * uncertainty1) ** 2 + (fraction2 * uncertainty2) ** 2
            base_uncertainty = math.sqrt(variance)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Increase uncertainty for composite (modeling uncertainty)
        # Factor of 2 accounts for interfacial effects, microstructure, etc.
        return base_uncertainty * 2.0
    
    def calculate_composite_properties(
        self,
        composite: CompositeSystem
    ) -> Dict[str, ValueWithUncertainty]:
        """
        Calculate all composite properties using appropriate mixing rules.
        
        Args:
            composite: Composite system
        
        Returns:
            Dictionary of calculated properties with uncertainties
        """
        self._issue_limitation_warning(
            "Calculating composite properties using rule-of-mixtures estimates"
        )
        
        comp1 = composite.component1
        comp2 = composite.component2
        f1 = composite.volume_fraction1
        f2 = composite.volume_fraction2
        
        properties = {}
        
        # Mechanical properties (linear rule of mixtures)
        properties["hardness"] = self._calculate_property(
            comp1.mechanical.hardness,
            comp2.mechanical.hardness,
            f1, f2,
            method="linear"
        )
        
        properties["fracture_toughness"] = self._calculate_property(
            comp1.mechanical.fracture_toughness,
            comp2.mechanical.fracture_toughness,
            f1, f2,
            method="linear"
        )
        
        properties["youngs_modulus"] = self._calculate_property(
            comp1.mechanical.youngs_modulus,
            comp2.mechanical.youngs_modulus,
            f1, f2,
            method="linear"
        )
        
        properties["bulk_modulus"] = self._calculate_property(
            comp1.mechanical.bulk_modulus,
            comp2.mechanical.bulk_modulus,
            f1, f2,
            method="linear"
        )
        
        properties["shear_modulus"] = self._calculate_property(
            comp1.mechanical.shear_modulus,
            comp2.mechanical.shear_modulus,
            f1, f2,
            method="linear"
        )
        
        properties["poisson_ratio"] = self._calculate_property(
            comp1.mechanical.poisson_ratio,
            comp2.mechanical.poisson_ratio,
            f1, f2,
            method="linear"
        )
        
        # Thermal properties (inverse rule for conductivity, linear for others)
        properties["thermal_conductivity_25C"] = self._calculate_property(
            comp1.thermal.thermal_conductivity_25C,
            comp2.thermal.thermal_conductivity_25C,
            f1, f2,
            method="inverse"
        )
        
        properties["thermal_conductivity_1000C"] = self._calculate_property(
            comp1.thermal.thermal_conductivity_1000C,
            comp2.thermal.thermal_conductivity_1000C,
            f1, f2,
            method="inverse"
        )
        
        properties["thermal_expansion"] = self._calculate_property(
            comp1.thermal.thermal_expansion,
            comp2.thermal.thermal_expansion,
            f1, f2,
            method="linear"
        )
        
        properties["specific_heat"] = self._calculate_property(
            comp1.thermal.specific_heat,
            comp2.thermal.specific_heat,
            f1, f2,
            method="linear"
        )
        
        # Melting point (use minimum - conservative estimate)
        mp1 = comp1.thermal.melting_point
        mp2 = comp2.thermal.melting_point
        min_mp = min(mp1.value, mp2.value)
        max_uncertainty = max(mp1.uncertainty, mp2.uncertainty)
        properties["melting_point"] = ValueWithUncertainty(
            value=min_mp,
            uncertainty=max_uncertainty * 1.5,
            unit=mp1.unit,
            source="composite_estimate"
        )
        
        # Structural properties
        properties["density"] = self._calculate_property(
            comp1.structural.density,
            comp2.structural.density,
            f1, f2,
            method="linear"
        )
        
        # Lattice parameter (geometric mean)
        properties["lattice_parameter"] = self._calculate_property(
            comp1.structural.lattice_parameter,
            comp2.structural.lattice_parameter,
            f1, f2,
            method="geometric"
        )
        
        return properties
    
    def _calculate_property(
        self,
        prop1: ValueWithUncertainty,
        prop2: ValueWithUncertainty,
        fraction1: float,
        fraction2: float,
        method: str = "linear"
    ) -> ValueWithUncertainty:
        """
        Calculate single composite property.
        
        Args:
            prop1: Property for component 1
            prop2: Property for component 2
            fraction1: Volume fraction of component 1
            fraction2: Volume fraction of component 2
            method: Mixing method
        
        Returns:
            Calculated property with uncertainty
        """
        if method == "linear":
            value = self.calculate_rule_of_mixtures(
                prop1.value, prop2.value, fraction1, fraction2
            )
        elif method == "inverse":
            value = self.calculate_inverse_rule_of_mixtures(
                prop1.value, prop2.value, fraction1, fraction2
            )
        elif method == "geometric":
            value = self.calculate_geometric_mean(
                prop1.value, prop2.value, fraction1, fraction2
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        uncertainty = self.propagate_uncertainty(
            prop1.uncertainty, prop2.uncertainty, fraction1, fraction2, method
        )
        
        return ValueWithUncertainty(
            value=value,
            uncertainty=uncertainty,
            unit=prop1.unit,
            source="composite_estimate"
        )
    
    def validate_against_known_composites(
        self,
        composite: CompositeSystem,
        calculated_properties: Dict[str, ValueWithUncertainty]
    ) -> Dict[str, bool]:
        """
        Validate calculated properties against known composite data.
        
        Args:
            composite: Composite system
            calculated_properties: Calculated properties
        
        Returns:
            Dictionary mapping property names to validation status
        """
        comp1_type = composite.component1.ceramic_type.value
        comp2_type = composite.component2.ceramic_type.value
        
        # Try both orderings
        composite_key = f"{comp1_type}-{comp2_type}"
        if composite_key not in self.KNOWN_COMPOSITES:
            composite_key = f"{comp2_type}-{comp1_type}"
        
        if composite_key not in self.KNOWN_COMPOSITES:
            return {}  # No known data for this composite
        
        known_data = self.KNOWN_COMPOSITES[composite_key]
        
        # Check if ratio matches known ratios
        ratio = (composite.volume_fraction1, composite.volume_fraction2)
        ratio_matches = any(
            math.isclose(ratio[0], kr[0], rel_tol=0.05) and
            math.isclose(ratio[1], kr[1], rel_tol=0.05)
            for kr in known_data["ratios"]
        )
        
        if not ratio_matches:
            return {}  # Ratio doesn't match known data
        
        # Validate properties
        validation_results = {}
        for prop_name, expected_data in known_data["properties"].items():
            if prop_name in calculated_properties:
                calc_value = calculated_properties[prop_name].value
                expected_value = expected_data["expected"]
                tolerance = expected_data["tolerance"]
                
                is_valid = abs(calc_value - expected_value) <= tolerance
                validation_results[prop_name] = is_valid
                
                if not is_valid:
                    warnings.warn(
                        f"Composite {prop_name} validation failed: "
                        f"calculated {calc_value:.2f}, expected {expected_value:.2f} "
                        f"± {tolerance:.2f}",
                        UserWarning
                    )
        
        return validation_results
