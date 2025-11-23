"""Physical plausibility validator for material properties."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import warnings

import numpy as np


class ValidationSeverity(Enum):
    """Severity level for validation violations."""
    
    INFO = "info"  # Informational, not a problem
    WARNING = "warning"  # Potentially problematic
    ERROR = "error"  # Definitely problematic
    CRITICAL = "critical"  # Physically impossible


@dataclass
class PropertyBounds:
    """Physical bounds for a material property."""
    
    min_value: float
    max_value: float
    unit: str
    typical_range: Optional[tuple[float, float]] = None  # Expected range for ceramics
    
    def is_within_bounds(self, value: float) -> bool:
        """Check if value is within physical bounds."""
        return self.min_value <= value <= self.max_value
    
    def is_within_typical(self, value: float) -> bool:
        """Check if value is within typical range."""
        if self.typical_range is None:
            return True
        return self.typical_range[0] <= value <= self.typical_range[1]


@dataclass
class ValidationViolation:
    """A validation violation with details."""
    
    property_name: str
    value: float
    severity: ValidationSeverity
    message: str
    expected_range: Optional[tuple[float, float]] = None
    unit: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a material."""
    
    material_id: str
    violations: List[ValidationViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    is_valid: bool = True
    confidence_score: float = 1.0
    
    def add_violation(self, violation: ValidationViolation) -> None:
        """Add a validation violation."""
        self.violations.append(violation)
        
        # Update validity based on severity
        if violation.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
            # Reduce confidence for errors
            self.confidence_score *= 0.5
        elif violation.severity == ValidationSeverity.WARNING:
            # Slightly reduce confidence for warnings
            self.confidence_score *= 0.9
    
    def get_violations_by_severity(self, severity: ValidationSeverity) -> List[ValidationViolation]:
        """Get violations of a specific severity."""
        return [v for v in self.violations if v.severity == severity]
    
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        return any(v.severity == ValidationSeverity.CRITICAL for v in self.violations)
    
    def summary(self) -> str:
        """Generate a summary of the validation report."""
        critical = len(self.get_violations_by_severity(ValidationSeverity.CRITICAL))
        errors = len(self.get_violations_by_severity(ValidationSeverity.ERROR))
        warnings = len(self.get_violations_by_severity(ValidationSeverity.WARNING))
        
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Validation Report for {self.material_id}: {status}\n"
            f"  Critical: {critical}, Errors: {errors}, Warnings: {warnings}\n"
            f"  Confidence Score: {self.confidence_score:.2f}"
        )


class PhysicalPlausibilityValidator:
    """
    Validator for physical plausibility of material properties.
    
    Checks material properties against:
    - Physical bounds (hard limits)
    - Typical ranges for ceramic materials
    - Cross-property consistency
    - Literature data when available
    """
    
    def __init__(self):
        """Initialize validator with property bounds."""
        self.property_bounds = self._initialize_property_bounds()
        self.literature_data = {}  # Will be populated with experimental data
    
    def _initialize_property_bounds(self) -> Dict[str, PropertyBounds]:
        """
        Initialize physical bounds for all material properties.
        
        Returns:
            Dictionary mapping property names to PropertyBounds
        """
        return {
            # Mechanical properties
            "hardness": PropertyBounds(
                min_value=0.1,
                max_value=50.0,
                unit="GPa",
                typical_range=(15.0, 35.0)  # Typical for ceramics
            ),
            "fracture_toughness": PropertyBounds(
                min_value=0.5,
                max_value=15.0,
                unit="MPa·m^0.5",
                typical_range=(2.0, 8.0)  # Typical for ceramics
            ),
            "youngs_modulus": PropertyBounds(
                min_value=50.0,
                max_value=1000.0,
                unit="GPa",
                typical_range=(300.0, 700.0)  # Typical for ceramics
            ),
            "bulk_modulus": PropertyBounds(
                min_value=50.0,
                max_value=500.0,
                unit="GPa",
                typical_range=(150.0, 450.0)
            ),
            "shear_modulus": PropertyBounds(
                min_value=50.0,
                max_value=400.0,
                unit="GPa",
                typical_range=(100.0, 300.0)
            ),
            "poisson_ratio": PropertyBounds(
                min_value=-1.0,
                max_value=0.5,
                unit="dimensionless",
                typical_range=(0.10, 0.30)  # Typical for ceramics
            ),
            
            # Thermal properties
            "thermal_conductivity_25C": PropertyBounds(
                min_value=0.1,
                max_value=500.0,
                unit="W/m·K",
                typical_range=(5.0, 150.0)  # Typical for ceramics
            ),
            "thermal_conductivity_1000C": PropertyBounds(
                min_value=0.1,
                max_value=500.0,
                unit="W/m·K",
                typical_range=(5.0, 100.0)
            ),
            "thermal_expansion": PropertyBounds(
                min_value=0.0,
                max_value=20.0,
                unit="10⁻⁶/K",
                typical_range=(3.0, 10.0)  # Typical for ceramics
            ),
            "specific_heat": PropertyBounds(
                min_value=100.0,
                max_value=2000.0,
                unit="J/kg·K",
                typical_range=(500.0, 1200.0)
            ),
            "melting_point": PropertyBounds(
                min_value=1000.0,
                max_value=5000.0,
                unit="K",
                typical_range=(2000.0, 3500.0)  # Typical for ceramics
            ),
            
            # Structural properties
            "density": PropertyBounds(
                min_value=1.0,
                max_value=20.0,
                unit="g/cm³",
                typical_range=(2.0, 16.0)  # Typical for ceramics
            ),
            "lattice_parameter": PropertyBounds(
                min_value=2.0,
                max_value=10.0,
                unit="Å",
                typical_range=(3.0, 6.0)
            ),
            
            # DFT-derived properties
            "energy_above_hull": PropertyBounds(
                min_value=0.0,
                max_value=2.0,
                unit="eV/atom",
                typical_range=(0.0, 0.1)  # Viable materials
            ),
            "formation_energy_per_atom": PropertyBounds(
                min_value=-10.0,
                max_value=5.0,
                unit="eV/atom",
                typical_range=(-5.0, 0.0)
            ),
            "band_gap": PropertyBounds(
                min_value=0.0,
                max_value=15.0,
                unit="eV",
                typical_range=(0.0, 6.0)
            ),
        }
    
    def validate_property(
        self,
        property_name: str,
        value: float,
        uncertainty: Optional[float] = None
    ) -> Optional[ValidationViolation]:
        """
        Validate a single property value.
        
        Args:
            property_name: Name of the property
            value: Property value
            uncertainty: Optional uncertainty in the value
        
        Returns:
            ValidationViolation if invalid, None if valid
        """
        if property_name not in self.property_bounds:
            return ValidationViolation(
                property_name=property_name,
                value=value,
                severity=ValidationSeverity.INFO,
                message=f"No validation bounds defined for {property_name}"
            )
        
        bounds = self.property_bounds[property_name]
        
        # Check physical bounds (hard limits)
        if not bounds.is_within_bounds(value):
            return ValidationViolation(
                property_name=property_name,
                value=value,
                severity=ValidationSeverity.CRITICAL,
                message=f"{property_name} = {value} {bounds.unit} is outside physical bounds",
                expected_range=(bounds.min_value, bounds.max_value),
                unit=bounds.unit
            )
        
        # Check typical range (soft limits)
        if not bounds.is_within_typical(value):
            return ValidationViolation(
                property_name=property_name,
                value=value,
                severity=ValidationSeverity.WARNING,
                message=f"{property_name} = {value} {bounds.unit} is outside typical range for ceramics",
                expected_range=bounds.typical_range,
                unit=bounds.unit
            )
        
        return None

    
    def validate_material(
        self,
        material_id: str,
        properties: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate all properties of a material.
        
        Args:
            material_id: Material identifier
            properties: Dictionary of property values
        
        Returns:
            ValidationReport with all violations
        """
        report = ValidationReport(material_id=material_id)
        
        # Validate individual properties
        for prop_name, prop_value in properties.items():
            # Handle ValueWithUncertainty objects
            if hasattr(prop_value, 'value'):
                value = prop_value.value
                uncertainty = prop_value.uncertainty
            else:
                value = prop_value
                uncertainty = None
            
            violation = self.validate_property(prop_name, value, uncertainty)
            if violation:
                report.add_violation(violation)
        
        # Check cross-property consistency
        consistency_violations = self._check_cross_property_consistency(properties)
        for violation in consistency_violations:
            report.add_violation(violation)
        
        return report
    
    def _check_cross_property_consistency(
        self,
        properties: Dict[str, Any]
    ) -> List[ValidationViolation]:
        """
        Check consistency between related properties.
        
        Args:
            properties: Dictionary of property values
        
        Returns:
            List of validation violations
        """
        violations = []
        
        # Extract values (handle ValueWithUncertainty objects)
        def get_value(prop_name: str) -> Optional[float]:
            if prop_name not in properties:
                return None
            prop = properties[prop_name]
            return prop.value if hasattr(prop, 'value') else prop
        
        # Check elastic moduli relationships
        E = get_value("youngs_modulus")
        G = get_value("shear_modulus")
        K = get_value("bulk_modulus")
        nu = get_value("poisson_ratio")
        
        if E and G and nu:
            # E = 2G(1 + ν)
            expected_E = 2 * G * (1 + nu)
            relative_error = abs(E - expected_E) / E
            if relative_error > 0.3:  # 30% tolerance
                violations.append(ValidationViolation(
                    property_name="elastic_moduli_consistency",
                    value=relative_error,
                    severity=ValidationSeverity.WARNING,
                    message=f"Elastic moduli inconsistent: E={E:.1f} GPa, but 2G(1+ν)={expected_E:.1f} GPa"
                ))
        
        if E and K and nu:
            # E = 3K(1 - 2ν)
            expected_E = 3 * K * (1 - 2 * nu)
            relative_error = abs(E - expected_E) / E
            if relative_error > 0.3:  # 30% tolerance
                violations.append(ValidationViolation(
                    property_name="elastic_moduli_consistency",
                    value=relative_error,
                    severity=ValidationSeverity.WARNING,
                    message=f"Elastic moduli inconsistent: E={E:.1f} GPa, but 3K(1-2ν)={expected_E:.1f} GPa"
                ))
        
        # Check thermal conductivity temperature dependence
        k_25 = get_value("thermal_conductivity_25C")
        k_1000 = get_value("thermal_conductivity_1000C")
        
        if k_25 and k_1000:
            # For ceramics, thermal conductivity typically decreases with temperature
            if k_1000 > k_25:
                violations.append(ValidationViolation(
                    property_name="thermal_conductivity_temperature",
                    value=k_1000 / k_25,
                    severity=ValidationSeverity.WARNING,
                    message=f"Thermal conductivity increases with temperature (unusual for ceramics): "
                            f"k(25°C)={k_25:.1f}, k(1000°C)={k_1000:.1f} W/m·K"
                ))
        
        # Check hardness vs fracture toughness trade-off
        hardness = get_value("hardness")
        toughness = get_value("fracture_toughness")
        
        if hardness and toughness:
            # Very hard materials typically have lower toughness
            if hardness > 30 and toughness > 6:
                violations.append(ValidationViolation(
                    property_name="hardness_toughness_tradeoff",
                    value=hardness * toughness,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unusually high hardness ({hardness:.1f} GPa) and toughness "
                            f"({toughness:.1f} MPa·m^0.5) combination"
                ))
        
        return violations
    
    def validate_batch(
        self,
        materials: List[Dict[str, Any]]
    ) -> List[ValidationReport]:
        """
        Validate multiple materials.
        
        Args:
            materials: List of material dictionaries with 'id' and properties
        
        Returns:
            List of ValidationReport objects
        """
        reports = []
        
        for material in materials:
            material_id = material.get('id', material.get('material_id', 'unknown'))
            # Remove id from properties dict
            properties = {k: v for k, v in material.items() if k not in ['id', 'material_id']}
            
            report = self.validate_material(material_id, properties)
            reports.append(report)
        
        return reports
    
    def add_literature_reference(
        self,
        material_id: str,
        property_name: str,
        value: float,
        uncertainty: float,
        source: str
    ) -> None:
        """
        Add experimental literature data for cross-reference.
        
        Args:
            material_id: Material identifier
            property_name: Property name
            value: Experimental value
            uncertainty: Experimental uncertainty
            source: Literature source citation
        """
        if material_id not in self.literature_data:
            self.literature_data[material_id] = {}
        
        self.literature_data[material_id][property_name] = {
            'value': value,
            'uncertainty': uncertainty,
            'source': source
        }
    
    def cross_reference_literature(
        self,
        material_id: str,
        property_name: str,
        predicted_value: float,
        predicted_uncertainty: Optional[float] = None
    ) -> Optional[ValidationViolation]:
        """
        Cross-reference predicted value with experimental literature data.
        
        Args:
            material_id: Material identifier
            property_name: Property name
            predicted_value: Predicted property value
            predicted_uncertainty: Optional uncertainty in prediction
        
        Returns:
            ValidationViolation if significant discrepancy, None otherwise
        """
        if material_id not in self.literature_data:
            return None
        
        if property_name not in self.literature_data[material_id]:
            return None
        
        lit_data = self.literature_data[material_id][property_name]
        lit_value = lit_data['value']
        lit_uncertainty = lit_data['uncertainty']
        
        # Calculate discrepancy
        discrepancy = abs(predicted_value - lit_value)
        relative_discrepancy = discrepancy / lit_value if lit_value != 0 else float('inf')
        
        # Determine if discrepancy is significant
        # Consider both experimental and predicted uncertainties
        total_uncertainty = lit_uncertainty
        if predicted_uncertainty:
            total_uncertainty = np.sqrt(lit_uncertainty**2 + predicted_uncertainty**2)
        
        # Discrepancy is significant if > 3σ
        if discrepancy > 3 * total_uncertainty:
            severity = ValidationSeverity.ERROR if relative_discrepancy > 0.5 else ValidationSeverity.WARNING
            
            return ValidationViolation(
                property_name=property_name,
                value=predicted_value,
                severity=severity,
                message=f"Predicted {property_name}={predicted_value:.2f} differs significantly from "
                        f"literature value {lit_value:.2f} ± {lit_uncertainty:.2f} "
                        f"(source: {lit_data['source']})",
                expected_range=(lit_value - lit_uncertainty, lit_value + lit_uncertainty)
            )
        
        return None
    
    def get_validation_statistics(
        self,
        reports: List[ValidationReport]
    ) -> Dict[str, Any]:
        """
        Calculate statistics across multiple validation reports.
        
        Args:
            reports: List of ValidationReport objects
        
        Returns:
            Dictionary with validation statistics
        """
        if not reports:
            return {
                'total_materials': 0,
                'valid_materials': 0,
                'invalid_materials': 0,
                'avg_confidence': 0.0,
                'total_violations': 0,
                'violations_by_severity': {}
            }
        
        valid_count = sum(1 for r in reports if r.is_valid)
        invalid_count = len(reports) - valid_count
        avg_confidence = np.mean([r.confidence_score for r in reports])
        
        # Count violations by severity
        all_violations = [v for r in reports for v in r.violations]
        violations_by_severity = {
            'critical': len([v for v in all_violations if v.severity == ValidationSeverity.CRITICAL]),
            'error': len([v for v in all_violations if v.severity == ValidationSeverity.ERROR]),
            'warning': len([v for v in all_violations if v.severity == ValidationSeverity.WARNING]),
            'info': len([v for v in all_violations if v.severity == ValidationSeverity.INFO])
        }
        
        # Most common violations
        violation_counts = {}
        for v in all_violations:
            violation_counts[v.property_name] = violation_counts.get(v.property_name, 0) + 1
        
        most_common = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_materials': len(reports),
            'valid_materials': valid_count,
            'invalid_materials': invalid_count,
            'valid_fraction': valid_count / len(reports),
            'avg_confidence': avg_confidence,
            'total_violations': len(all_violations),
            'violations_by_severity': violations_by_severity,
            'most_common_violations': most_common
        }


class SensitivityAnalyzer:
    """
    Analyzer for sensitivity of key performance drivers.
    
    Identifies which material properties have the strongest influence
    on ballistic performance and other target metrics.
    """
    
    def __init__(self):
        """Initialize sensitivity analyzer."""
        self.sensitivity_cache = {}
    
    def analyze_property_sensitivity(
        self,
        base_properties: Dict[str, float],
        target_metric_calculator: callable,
        perturbation_fraction: float = 0.1
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of target metric to property changes.
        
        Uses one-at-a-time (OAT) sensitivity analysis:
        - Perturb each property by a small amount
        - Calculate change in target metric
        - Compute sensitivity coefficient
        
        Args:
            base_properties: Baseline property values
            target_metric_calculator: Function that calculates target metric from properties
            perturbation_fraction: Fractional perturbation (default 10%)
        
        Returns:
            Dictionary mapping property names to sensitivity coefficients
        """
        # Calculate baseline metric
        baseline_metric = target_metric_calculator(base_properties)
        
        sensitivities = {}
        
        for prop_name, base_value in base_properties.items():
            # Skip non-numeric properties
            if not isinstance(base_value, (int, float)):
                continue
            
            # Perturb property
            perturbation = base_value * perturbation_fraction
            perturbed_properties = base_properties.copy()
            perturbed_properties[prop_name] = base_value + perturbation
            
            # Calculate perturbed metric
            perturbed_metric = target_metric_calculator(perturbed_properties)
            
            # Calculate sensitivity: (ΔMetric / Metric) / (ΔProperty / Property)
            if baseline_metric != 0 and perturbation != 0:
                metric_change = (perturbed_metric - baseline_metric) / baseline_metric
                property_change = perturbation / base_value
                sensitivity = metric_change / property_change
            else:
                sensitivity = 0.0
            
            sensitivities[prop_name] = sensitivity
        
        return sensitivities
    
    def rank_performance_drivers(
        self,
        sensitivities: Dict[str, float],
        top_n: int = 10
    ) -> List[tuple[str, float]]:
        """
        Rank properties by their influence on performance.
        
        Args:
            sensitivities: Dictionary of sensitivity coefficients
            top_n: Number of top drivers to return
        
        Returns:
            List of (property_name, sensitivity) tuples, sorted by absolute sensitivity
        """
        # Sort by absolute sensitivity (magnitude of influence)
        ranked = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return ranked[:top_n]
    
    def generate_sensitivity_report(
        self,
        sensitivities: Dict[str, float],
        property_units: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate human-readable sensitivity report.
        
        Args:
            sensitivities: Dictionary of sensitivity coefficients
            property_units: Optional dictionary of property units
        
        Returns:
            Formatted sensitivity report string
        """
        ranked = self.rank_performance_drivers(sensitivities)
        
        report = "Sensitivity Analysis Report\n"
        report += "=" * 50 + "\n\n"
        report += "Top Performance Drivers:\n"
        report += "-" * 50 + "\n"
        
        for i, (prop_name, sensitivity) in enumerate(ranked, 1):
            unit = property_units.get(prop_name, "") if property_units else ""
            direction = "increases" if sensitivity > 0 else "decreases"
            
            report += f"{i}. {prop_name}"
            if unit:
                report += f" ({unit})"
            report += f"\n   Sensitivity: {sensitivity:.3f}\n"
            report += f"   A 10% increase in {prop_name} {direction} performance by "
            report += f"{abs(sensitivity * 10):.1f}%\n\n"
        
        return report
