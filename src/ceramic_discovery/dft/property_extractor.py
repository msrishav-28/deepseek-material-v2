"""Standardized property extraction for 58 material properties."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PropertyValue:
    """Container for property value with metadata."""

    value: Optional[float]
    unit: str
    source: str  # 'dft', 'experimental', 'predicted', 'calculated'
    uncertainty: Optional[float] = None
    quality_score: float = 1.0
    notes: str = ""


class PropertyExtractor:
    """Extract and standardize 58 material properties."""

    # Property definitions with units and categories
    PROPERTY_DEFINITIONS = {
        # Tier 1: Fundamental Structural Properties (1-10)
        "lattice_a": {"unit": "Å", "category": "structural", "tier": 1},
        "lattice_b": {"unit": "Å", "category": "structural", "tier": 1},
        "lattice_c": {"unit": "Å", "category": "structural", "tier": 1},
        "lattice_alpha": {"unit": "degrees", "category": "structural", "tier": 1},
        "lattice_beta": {"unit": "degrees", "category": "structural", "tier": 1},
        "lattice_gamma": {"unit": "degrees", "category": "structural", "tier": 1},
        "volume": {"unit": "Å³", "category": "structural", "tier": 1},
        "density": {"unit": "g/cm³", "category": "structural", "tier": 1},
        "space_group_number": {"unit": "dimensionless", "category": "structural", "tier": 1},
        "nsites": {"unit": "dimensionless", "category": "structural", "tier": 1},
        # Tier 1: Fundamental Thermodynamic Properties (11-15)
        "formation_energy_per_atom": {"unit": "eV/atom", "category": "thermodynamic", "tier": 1},
        "energy_above_hull": {"unit": "eV/atom", "category": "thermodynamic", "tier": 1},
        "decomposition_energy": {"unit": "eV/atom", "category": "thermodynamic", "tier": 1},
        "is_stable": {"unit": "boolean", "category": "thermodynamic", "tier": 1},
        "decomposes_to": {"unit": "list", "category": "thermodynamic", "tier": 1},
        # Tier 1: Fundamental Electronic Properties (16-20)
        "band_gap": {"unit": "eV", "category": "electronic", "tier": 1},
        "is_metal": {"unit": "boolean", "category": "electronic", "tier": 1},
        "is_magnetic": {"unit": "boolean", "category": "electronic", "tier": 1},
        "total_magnetization": {"unit": "μB", "category": "electronic", "tier": 1},
        "efermi": {"unit": "eV", "category": "electronic", "tier": 1},
        # Tier 2: Elastic and Mechanical Properties (21-30)
        "bulk_modulus_vrh": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "shear_modulus_vrh": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "youngs_modulus": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "poisson_ratio": {"unit": "dimensionless", "category": "mechanical", "tier": 2},
        "elastic_anisotropy": {"unit": "dimensionless", "category": "mechanical", "tier": 2},
        "hardness_vickers": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "fracture_toughness": {"unit": "MPa·m^0.5", "category": "mechanical", "tier": 2},
        "c11": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "c12": {"unit": "GPa", "category": "mechanical", "tier": 2},
        "c44": {"unit": "GPa", "category": "mechanical", "tier": 2},
        # Tier 2: Thermal Properties (31-38)
        "thermal_conductivity_300K": {"unit": "W/m·K", "category": "thermal", "tier": 2},
        "thermal_conductivity_1000K": {"unit": "W/m·K", "category": "thermal", "tier": 2},
        "thermal_expansion_300K": {"unit": "10⁻⁶/K", "category": "thermal", "tier": 2},
        "specific_heat_300K": {"unit": "J/kg·K", "category": "thermal", "tier": 2},
        "debye_temperature": {"unit": "K", "category": "thermal", "tier": 2},
        "melting_point": {"unit": "K", "category": "thermal", "tier": 2},
        "thermal_diffusivity": {"unit": "mm²/s", "category": "thermal", "tier": 2},
        "phonon_band_gap": {"unit": "THz", "category": "thermal", "tier": 2},
        # Tier 2: Optical Properties (39-43)
        "refractive_index": {"unit": "dimensionless", "category": "optical", "tier": 2},
        "dielectric_constant": {"unit": "dimensionless", "category": "optical", "tier": 2},
        "absorption_coefficient": {"unit": "cm⁻¹", "category": "optical", "tier": 2},
        "reflectivity": {"unit": "dimensionless", "category": "optical", "tier": 2},
        "transparency_range": {"unit": "μm", "category": "optical", "tier": 2},
        # Tier 3: Derived Ballistic Properties (44-50)
        "v50_predicted": {"unit": "m/s", "category": "ballistic", "tier": 3},
        "areal_density": {"unit": "kg/m²", "category": "ballistic", "tier": 3},
        "specific_energy_absorption": {"unit": "J·m²/kg", "category": "ballistic", "tier": 3},
        "penetration_resistance": {"unit": "dimensionless", "category": "ballistic", "tier": 3},
        "multi_hit_capability": {"unit": "dimensionless", "category": "ballistic", "tier": 3},
        "spall_resistance": {"unit": "dimensionless", "category": "ballistic", "tier": 3},
        "shock_impedance": {"unit": "kg/m²·s", "category": "ballistic", "tier": 3},
        # Additional Properties (51-58)
        "cost_per_kg": {"unit": "USD/kg", "category": "economic", "tier": 3},
        "availability_score": {"unit": "dimensionless", "category": "economic", "tier": 3},
        "manufacturability_score": {"unit": "dimensionless", "category": "manufacturing", "tier": 3},
        "environmental_impact": {"unit": "dimensionless", "category": "environmental", "tier": 3},
        "oxidation_resistance": {"unit": "dimensionless", "category": "chemical", "tier": 2},
        "corrosion_resistance": {"unit": "dimensionless", "category": "chemical", "tier": 2},
        "grain_size": {"unit": "μm", "category": "microstructural", "tier": 2},
        "porosity": {"unit": "%", "category": "microstructural", "tier": 2},
    }

    def __init__(self):
        """Initialize property extractor."""
        self.tier_1_properties = self._get_properties_by_tier(1)
        self.tier_2_properties = self._get_properties_by_tier(2)
        self.tier_3_properties = self._get_properties_by_tier(3)

    def _get_properties_by_tier(self, tier: int) -> List[str]:
        """Get list of property names for a specific tier."""
        return [
            name
            for name, info in self.PROPERTY_DEFINITIONS.items()
            if info.get("tier") == tier
        ]

    def extract_all_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """
        Extract all 58 properties from material data.

        Args:
            material_data: Raw material data dictionary

        Returns:
            Dictionary mapping property names to PropertyValue objects
        """
        extracted = {}

        # Extract structural properties
        extracted.update(self._extract_structural_properties(material_data))

        # Extract thermodynamic properties
        extracted.update(self._extract_thermodynamic_properties(material_data))

        # Extract electronic properties
        extracted.update(self._extract_electronic_properties(material_data))

        # Extract mechanical properties
        extracted.update(self._extract_mechanical_properties(material_data))

        # Extract thermal properties
        extracted.update(self._extract_thermal_properties(material_data))

        # Extract optical properties
        extracted.update(self._extract_optical_properties(material_data))

        # Initialize placeholders for properties not yet available
        for prop_name in self.PROPERTY_DEFINITIONS:
            if prop_name not in extracted:
                prop_def = self.PROPERTY_DEFINITIONS[prop_name]
                extracted[prop_name] = PropertyValue(
                    value=None,
                    unit=prop_def["unit"],
                    source="not_available",
                    quality_score=0.0,
                    notes="Property not available in source data",
                )

        return extracted

    def _extract_structural_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract structural properties from material data."""
        properties = {}

        # Lattice parameters
        if "crystal_structure" in material_data:
            structure = material_data["crystal_structure"]
            lattice = structure.get("lattice", {})

            if "a" in lattice:
                properties["lattice_a"] = PropertyValue(
                    value=lattice["a"], unit="Å", source="dft"
                )
            if "b" in lattice:
                properties["lattice_b"] = PropertyValue(
                    value=lattice["b"], unit="Å", source="dft"
                )
            if "c" in lattice:
                properties["lattice_c"] = PropertyValue(
                    value=lattice["c"], unit="Å", source="dft"
                )
            if "alpha" in lattice:
                properties["lattice_alpha"] = PropertyValue(
                    value=lattice["alpha"], unit="degrees", source="dft"
                )
            if "beta" in lattice:
                properties["lattice_beta"] = PropertyValue(
                    value=lattice["beta"], unit="degrees", source="dft"
                )
            if "gamma" in lattice:
                properties["lattice_gamma"] = PropertyValue(
                    value=lattice["gamma"], unit="degrees", source="dft"
                )
            if "volume" in lattice:
                properties["volume"] = PropertyValue(
                    value=lattice["volume"], unit="Å³", source="dft"
                )

            # Space group
            if "space_group" in structure:
                sg = structure["space_group"]
                if isinstance(sg, dict) and "number" in sg:
                    properties["space_group_number"] = PropertyValue(
                        value=float(sg["number"]), unit="dimensionless", source="dft"
                    )

            # Number of sites
            if "nsites" in structure:
                properties["nsites"] = PropertyValue(
                    value=float(structure["nsites"]), unit="dimensionless", source="dft"
                )

        # Density
        if "density" in material_data:
            properties["density"] = PropertyValue(
                value=material_data["density"], unit="g/cm³", source="dft"
            )

        return properties

    def _extract_thermodynamic_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract thermodynamic properties from material data."""
        properties = {}

        if "formation_energy" in material_data:
            properties["formation_energy_per_atom"] = PropertyValue(
                value=material_data["formation_energy"], unit="eV/atom", source="dft"
            )

        if "energy_above_hull" in material_data:
            properties["energy_above_hull"] = PropertyValue(
                value=material_data["energy_above_hull"], unit="eV/atom", source="dft"
            )

        if "is_stable" in material_data:
            properties["is_stable"] = PropertyValue(
                value=float(material_data["is_stable"]), unit="boolean", source="dft"
            )

        return properties

    def _extract_electronic_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract electronic properties from material data."""
        properties = {}

        if "band_gap" in material_data:
            properties["band_gap"] = PropertyValue(
                value=material_data["band_gap"], unit="eV", source="dft"
            )

        if "is_metal" in material_data:
            properties["is_metal"] = PropertyValue(
                value=float(material_data["is_metal"]), unit="boolean", source="dft"
            )

        if "is_magnetic" in material_data:
            properties["is_magnetic"] = PropertyValue(
                value=float(material_data["is_magnetic"]), unit="boolean", source="dft"
            )

        if "total_magnetization" in material_data:
            properties["total_magnetization"] = PropertyValue(
                value=material_data["total_magnetization"], unit="μB", source="dft"
            )

        return properties

    def _extract_mechanical_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract mechanical properties from material data."""
        properties = {}

        if "bulk_modulus" in material_data:
            properties["bulk_modulus_vrh"] = PropertyValue(
                value=material_data["bulk_modulus"], unit="GPa", source="dft"
            )

        if "shear_modulus" in material_data:
            properties["shear_modulus_vrh"] = PropertyValue(
                value=material_data["shear_modulus"], unit="GPa", source="dft"
            )

        # Calculate Young's modulus if bulk and shear moduli are available
        if "bulk_modulus" in material_data and "shear_modulus" in material_data:
            K = material_data["bulk_modulus"]
            G = material_data["shear_modulus"]
            E = (9 * K * G) / (3 * K + G)
            properties["youngs_modulus"] = PropertyValue(
                value=E, unit="GPa", source="calculated"
            )

            # Calculate Poisson's ratio
            nu = (3 * K - 2 * G) / (2 * (3 * K + G))
            properties["poisson_ratio"] = PropertyValue(
                value=nu, unit="dimensionless", source="calculated"
            )

        return properties

    def _extract_thermal_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract thermal properties from material data."""
        properties = {}

        # Thermal properties are typically not in basic MP data
        # These would come from additional calculations or experimental data

        return properties

    def _extract_optical_properties(
        self, material_data: Dict[str, Any]
    ) -> Dict[str, PropertyValue]:
        """Extract optical properties from material data."""
        properties = {}

        if "dielectric_constant" in material_data:
            properties["dielectric_constant"] = PropertyValue(
                value=material_data["dielectric_constant"],
                unit="dimensionless",
                source="dft",
            )

        return properties

    def get_tier_1_properties(
        self, extracted_properties: Dict[str, PropertyValue]
    ) -> Dict[str, PropertyValue]:
        """Get only Tier 1 fundamental properties."""
        return {
            name: prop
            for name, prop in extracted_properties.items()
            if name in self.tier_1_properties
        }

    def get_tier_2_properties(
        self, extracted_properties: Dict[str, PropertyValue]
    ) -> Dict[str, PropertyValue]:
        """Get only Tier 2 properties."""
        return {
            name: prop
            for name, prop in extracted_properties.items()
            if name in self.tier_2_properties
        }

    def filter_available_properties(
        self, extracted_properties: Dict[str, PropertyValue]
    ) -> Dict[str, PropertyValue]:
        """Filter to only properties with available values."""
        return {
            name: prop
            for name, prop in extracted_properties.items()
            if prop.value is not None
        }
