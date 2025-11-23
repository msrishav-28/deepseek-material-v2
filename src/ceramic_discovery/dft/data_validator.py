"""Data validation and quality checking for materials data."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float  # 0-1 scale


@dataclass
class PropertyBounds:
    """Physical bounds for material properties."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""


class DataValidator:
    """Validator for materials data quality and physical plausibility."""

    # Physical bounds for common properties
    PROPERTY_BOUNDS = {
        "density": PropertyBounds(min_value=0.1, max_value=25.0, unit="g/cm³"),
        "formation_energy_per_atom": PropertyBounds(
            min_value=-20.0, max_value=5.0, unit="eV/atom"
        ),
        "energy_above_hull": PropertyBounds(min_value=0.0, max_value=5.0, unit="eV/atom"),
        "band_gap": PropertyBounds(min_value=0.0, max_value=15.0, unit="eV"),
        "bulk_modulus": PropertyBounds(min_value=1.0, max_value=500.0, unit="GPa"),
        "shear_modulus": PropertyBounds(min_value=1.0, max_value=300.0, unit="GPa"),
        "hardness": PropertyBounds(min_value=0.1, max_value=50.0, unit="GPa"),
        "fracture_toughness": PropertyBounds(
            min_value=0.5, max_value=15.0, unit="MPa·m^0.5"
        ),
        "youngs_modulus": PropertyBounds(min_value=1.0, max_value=1000.0, unit="GPa"),
        "thermal_conductivity": PropertyBounds(min_value=0.1, max_value=500.0, unit="W/m·K"),
    }

    # Required properties for ceramic materials
    REQUIRED_PROPERTIES = {
        "material_id",
        "formula",
        "formation_energy_per_atom",
        "energy_above_hull",
    }

    def __init__(self, outlier_threshold_sigma: float = 3.0):
        """
        Initialize data validator.

        Args:
            outlier_threshold_sigma: Number of standard deviations for outlier detection
        """
        self.outlier_threshold = outlier_threshold_sigma

    def validate_material(self, material_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single material's data.

        Args:
            material_data: Dictionary containing material properties

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        quality_score = 1.0

        # Check required properties
        missing_required = self.REQUIRED_PROPERTIES - set(material_data.keys())
        if missing_required:
            errors.append(f"Missing required properties: {missing_required}")
            quality_score -= 0.3

        # Validate property bounds
        for prop_name, value in material_data.items():
            if prop_name in self.PROPERTY_BOUNDS and value is not None:
                bounds = self.PROPERTY_BOUNDS[prop_name]
                validation = self._check_bounds(prop_name, value, bounds)

                if not validation["valid"]:
                    errors.append(validation["message"])
                    quality_score -= 0.1
                elif validation.get("warning"):
                    warnings.append(validation["message"])
                    quality_score -= 0.05

        # Check for physically implausible combinations
        implausibility_check = self._check_physical_consistency(material_data)
        if implausibility_check["errors"]:
            errors.extend(implausibility_check["errors"])
            quality_score -= 0.2
        if implausibility_check["warnings"]:
            warnings.extend(implausibility_check["warnings"])
            quality_score -= 0.05

        # Ensure quality score is in [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, quality_score=quality_score
        )

    def _check_bounds(
        self, prop_name: str, value: float, bounds: PropertyBounds
    ) -> Dict[str, Any]:
        """
        Check if property value is within physical bounds.

        Args:
            prop_name: Property name
            value: Property value
            bounds: PropertyBounds object

        Returns:
            Dictionary with validation result
        """
        result = {"valid": True, "warning": False, "message": ""}

        if bounds.min_value is not None and value < bounds.min_value:
            result["valid"] = False
            result["message"] = (
                f"{prop_name} = {value} {bounds.unit or ''} is below "
                f"minimum bound {bounds.min_value}"
            )
        elif bounds.max_value is not None and value > bounds.max_value:
            result["valid"] = False
            result["message"] = (
                f"{prop_name} = {value} {bounds.unit or ''} exceeds "
                f"maximum bound {bounds.max_value}"
            )

        # Warn if close to bounds (within 10%)
        if result["valid"] and bounds.min_value is not None:
            range_size = (bounds.max_value or value * 2) - bounds.min_value
            if value < bounds.min_value + 0.1 * range_size:
                result["warning"] = True
                result["message"] = f"{prop_name} = {value} is close to lower bound"

        if result["valid"] and bounds.max_value is not None:
            range_size = bounds.max_value - (bounds.min_value or 0)
            if value > bounds.max_value - 0.1 * range_size:
                result["warning"] = True
                result["message"] = f"{prop_name} = {value} is close to upper bound"

        return result

    def _check_physical_consistency(self, material_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Check for physically inconsistent property combinations.

        Args:
            material_data: Dictionary containing material properties

        Returns:
            Dictionary with lists of errors and warnings
        """
        errors = []
        warnings = []

        # Check: Metals should have zero band gap
        if material_data.get("is_metal") and material_data.get("band_gap", 0) > 0.1:
            warnings.append(
                f"Material marked as metal but has band gap = {material_data['band_gap']} eV"
            )

        # Check: Stable materials should have energy_above_hull = 0
        if material_data.get("is_stable") and material_data.get("energy_above_hull", 0) > 0.001:
            warnings.append(
                f"Material marked as stable but has energy_above_hull = "
                f"{material_data['energy_above_hull']} eV/atom"
            )

        # Check: Bulk modulus should be greater than shear modulus
        bulk_mod = material_data.get("bulk_modulus")
        shear_mod = material_data.get("shear_modulus")
        if bulk_mod is not None and shear_mod is not None:
            if shear_mod > bulk_mod:
                errors.append(
                    f"Shear modulus ({shear_mod} GPa) exceeds bulk modulus ({bulk_mod} GPa)"
                )

        # Check: Density should be reasonable for given formula
        density = material_data.get("density")
        if density is not None:
            # Ceramics typically have density 2-6 g/cm³
            if density < 1.5 or density > 10.0:
                warnings.append(
                    f"Density {density} g/cm³ is unusual for ceramic materials"
                )

        return {"errors": errors, "warnings": warnings}

    def detect_outliers(
        self, materials: List[Dict[str, Any]], property_name: str
    ) -> List[int]:
        """
        Detect outliers in a property across multiple materials.

        Args:
            materials: List of material data dictionaries
            property_name: Name of property to check for outliers

        Returns:
            List of indices of materials with outlier values
        """
        values = []
        indices = []

        for i, material in enumerate(materials):
            if property_name in material and material[property_name] is not None:
                values.append(material[property_name])
                indices.append(i)

        if len(values) < 3:
            return []  # Need at least 3 values for outlier detection

        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)

        if std == 0:
            return []  # All values are the same

        outlier_indices = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > self.outlier_threshold:
                outlier_indices.append(indices[i])

        return outlier_indices

    def calculate_data_quality_score(self, material_data: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score for a material.

        Args:
            material_data: Dictionary containing material properties

        Returns:
            Quality score between 0 and 1
        """
        validation = self.validate_material(material_data)
        return validation.quality_score

    def validate_batch(
        self, materials: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate a batch of materials and provide summary statistics.

        Args:
            materials: List of material data dictionaries

        Returns:
            Dictionary with validation summary
        """
        results = []
        for material in materials:
            results.append(self.validate_material(material))

        valid_count = sum(1 for r in results if r.is_valid)
        avg_quality = np.mean([r.quality_score for r in results])

        all_errors = []
        all_warnings = []
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)

        return {
            "total_materials": len(materials),
            "valid_materials": valid_count,
            "invalid_materials": len(materials) - valid_count,
            "average_quality_score": avg_quality,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "unique_errors": list(set(all_errors)),
            "unique_warnings": list(set(all_warnings)),
            "individual_results": results,
        }
