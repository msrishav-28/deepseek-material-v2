"""Unit conversion and standardization system."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class UnitConversion:
    """Unit conversion definition."""

    from_unit: str
    to_unit: str
    conversion_factor: float
    offset: float = 0.0  # For temperature conversions


class UnitConverter:
    """Convert and standardize material property units."""

    # Standard units for each property type
    STANDARD_UNITS = {
        # Mechanical properties
        "hardness": "GPa",
        "fracture_toughness": "MPa·m^0.5",  # CRITICAL: Correct unit
        "youngs_modulus": "GPa",
        "bulk_modulus": "GPa",
        "shear_modulus": "GPa",
        "poisson_ratio": "dimensionless",
        "elastic_anisotropy": "dimensionless",
        # Thermal properties
        "thermal_conductivity": "W/m·K",
        "thermal_expansion": "10⁻⁶/K",
        "specific_heat": "J/kg·K",
        "debye_temperature": "K",
        "melting_point": "K",
        "thermal_diffusivity": "mm²/s",
        # Structural properties
        "density": "g/cm³",
        "lattice_parameter": "Å",
        "volume": "Å³",
        # Thermodynamic properties
        "formation_energy": "eV/atom",
        "energy_above_hull": "eV/atom",
        # Electronic properties
        "band_gap": "eV",
        "efermi": "eV",
        # Ballistic properties
        "v50": "m/s",
        "areal_density": "kg/m²",
        "specific_energy_absorption": "J·m²/kg",
    }

    # Conversion factors (to standard units)
    CONVERSIONS = {
        # Fracture toughness conversions
        ("MPa·m^1/2", "MPa·m^0.5"): UnitConversion("MPa·m^1/2", "MPa·m^0.5", 1.0),
        ("MPa√m", "MPa·m^0.5"): UnitConversion("MPa√m", "MPa·m^0.5", 1.0),
        ("ksi·in^0.5", "MPa·m^0.5"): UnitConversion("ksi·in^0.5", "MPa·m^0.5", 1.099),
        # Pressure/stress conversions
        ("Pa", "GPa"): UnitConversion("Pa", "GPa", 1e-9),
        ("MPa", "GPa"): UnitConversion("MPa", "GPa", 1e-3),
        ("kPa", "GPa"): UnitConversion("kPa", "GPa", 1e-6),
        ("psi", "GPa"): UnitConversion("psi", "GPa", 6.89476e-6),
        # Density conversions
        ("kg/m³", "g/cm³"): UnitConversion("kg/m³", "g/cm³", 1e-3),
        ("lb/in³", "g/cm³"): UnitConversion("lb/in³", "g/cm³", 27.68),
        # Length conversions
        ("nm", "Å"): UnitConversion("nm", "Å", 10.0),
        ("pm", "Å"): UnitConversion("pm", "Å", 0.01),
        ("m", "Å"): UnitConversion("m", "Å", 1e10),
        # Energy conversions
        ("J", "eV"): UnitConversion("J", "eV", 6.242e18),
        ("kJ/mol", "eV/atom"): UnitConversion("kJ/mol", "eV/atom", 0.01036),
        ("kcal/mol", "eV/atom"): UnitConversion("kcal/mol", "eV/atom", 0.04336),
        # Temperature conversions
        ("°C", "K"): UnitConversion("°C", "K", 1.0, 273.15),
        ("°F", "K"): UnitConversion("°F", "K", 5 / 9, 255.372),
        # Thermal conductivity conversions
        ("W/cm·K", "W/m·K"): UnitConversion("W/cm·K", "W/m·K", 100.0),
        ("cal/s·cm·K", "W/m·K"): UnitConversion("cal/s·cm·K", "W/m·K", 418.4),
        # Velocity conversions
        ("km/s", "m/s"): UnitConversion("km/s", "m/s", 1000.0),
        ("ft/s", "m/s"): UnitConversion("ft/s", "m/s", 0.3048),
    }

    def __init__(self):
        """Initialize unit converter."""
        # Build reverse conversions
        self._build_reverse_conversions()

    def _build_reverse_conversions(self) -> None:
        """Build reverse conversion mappings."""
        reverse_conversions = {}

        for (from_unit, to_unit), conversion in self.CONVERSIONS.items():
            # Add reverse conversion
            if conversion.offset == 0.0:
                # Simple multiplicative conversion
                reverse_conversions[(to_unit, from_unit)] = UnitConversion(
                    to_unit, from_unit, 1.0 / conversion.conversion_factor
                )
            else:
                # Temperature conversion with offset
                reverse_conversions[(to_unit, from_unit)] = UnitConversion(
                    to_unit,
                    from_unit,
                    1.0 / conversion.conversion_factor,
                    -conversion.offset / conversion.conversion_factor,
                )

        self.CONVERSIONS.update(reverse_conversions)

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value from one unit to another.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If conversion is not supported
        """
        if from_unit == to_unit:
            return value

        # Normalize unit strings
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)

        # Look up conversion
        conversion_key = (from_unit, to_unit)
        if conversion_key not in self.CONVERSIONS:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

        conversion = self.CONVERSIONS[conversion_key]

        # Apply conversion
        converted_value = value * conversion.conversion_factor + conversion.offset

        return converted_value

    def standardize(self, value: float, current_unit: str, property_name: str) -> Tuple[float, str]:
        """
        Convert value to standard unit for property.

        Args:
            value: Value to standardize
            current_unit: Current unit
            property_name: Property name

        Returns:
            Tuple of (standardized_value, standard_unit)

        Raises:
            ValueError: If property or conversion not supported
        """
        if property_name not in self.STANDARD_UNITS:
            raise ValueError(f"Unknown property: {property_name}")

        standard_unit = self.STANDARD_UNITS[property_name]

        # Convert to standard unit
        try:
            standardized_value = self.convert(value, current_unit, standard_unit)
        except ValueError:
            # If direct conversion not available, return original with warning
            print(
                f"Warning: Cannot convert {current_unit} to {standard_unit} for {property_name}"
            )
            return value, current_unit

        return standardized_value, standard_unit

    def _normalize_unit(self, unit: str) -> str:
        """
        Normalize unit string for consistent matching.

        Args:
            unit: Unit string

        Returns:
            Normalized unit string
        """
        # Remove extra whitespace
        unit = unit.strip()

        # Common normalizations
        normalizations = {
            "mpa*m^0.5": "MPa·m^0.5",
            "mpa*m^1/2": "MPa·m^0.5",
            "mpa*sqrt(m)": "MPa·m^0.5",
            "mpa√m": "MPa·m^0.5",
            "gpa": "GPa",
            "g/cc": "g/cm³",
            "g/cm3": "g/cm³",
            "angstrom": "Å",
            "ev": "eV",
            "w/mk": "W/m·K",
            "w/(m*k)": "W/m·K",
        }

        unit_lower = unit.lower()
        if unit_lower in normalizations:
            return normalizations[unit_lower]

        return unit

    def validate_unit(self, unit: str, property_name: str) -> bool:
        """
        Validate that unit is appropriate for property.

        Args:
            unit: Unit string
            property_name: Property name

        Returns:
            True if unit is valid for property
        """
        if property_name not in self.STANDARD_UNITS:
            return False

        standard_unit = self.STANDARD_UNITS[property_name]
        normalized_unit = self._normalize_unit(unit)

        # Check if it's the standard unit
        if normalized_unit == standard_unit:
            return True

        # Check if conversion exists
        try:
            self.convert(1.0, normalized_unit, standard_unit)
            return True
        except ValueError:
            return False

    def get_standard_unit(self, property_name: str) -> Optional[str]:
        """
        Get standard unit for a property.

        Args:
            property_name: Property name

        Returns:
            Standard unit string or None if property unknown
        """
        return self.STANDARD_UNITS.get(property_name)

    def batch_standardize(
        self, properties: Dict[str, Tuple[float, str]]
    ) -> Dict[str, Tuple[float, str]]:
        """
        Standardize multiple properties at once.

        Args:
            properties: Dictionary mapping property names to (value, unit) tuples

        Returns:
            Dictionary mapping property names to (standardized_value, standard_unit) tuples
        """
        standardized = {}

        for prop_name, (value, unit) in properties.items():
            try:
                std_value, std_unit = self.standardize(value, unit, prop_name)
                standardized[prop_name] = (std_value, std_unit)
            except ValueError as e:
                print(f"Warning: Could not standardize {prop_name}: {e}")
                standardized[prop_name] = (value, unit)

        return standardized


# Global converter instance
converter = UnitConverter()
