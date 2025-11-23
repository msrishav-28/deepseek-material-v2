"""Base ceramic system classes with baseline properties and dopant handling."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import warnings
from enum import Enum


class CeramicType(Enum):
    """Supported ceramic system types."""
    
    SIC = "SiC"
    B4C = "B₄C"
    WC = "WC"
    TIC = "TiC"
    AL2O3 = "Al₂O₃"


@dataclass
class ValueWithUncertainty:
    """Property value with uncertainty quantification."""
    
    value: float
    uncertainty: float
    unit: str
    source: str = "baseline"  # 'baseline', 'dft', 'experimental', 'predicted'
    
    def __post_init__(self):
        """Validate value and uncertainty."""
        if self.uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative")


@dataclass
class MechanicalProperties:
    """Mechanical properties of ceramic materials."""
    
    hardness: ValueWithUncertainty  # GPa
    fracture_toughness: ValueWithUncertainty  # MPa·m^0.5
    youngs_modulus: ValueWithUncertainty  # GPa
    bulk_modulus: ValueWithUncertainty  # GPa
    shear_modulus: ValueWithUncertainty  # GPa
    poisson_ratio: ValueWithUncertainty  # dimensionless
    
    def validate_bounds(self) -> List[str]:
        """Validate physical bounds for mechanical properties."""
        violations = []
        
        # Hardness bounds (GPa)
        if not (0 < self.hardness.value < 50):
            violations.append(f"Hardness {self.hardness.value} GPa out of physical bounds (0-50 GPa)")
        
        # Fracture toughness bounds (MPa·m^0.5)
        if not (0 < self.fracture_toughness.value < 15):
            violations.append(
                f"Fracture toughness {self.fracture_toughness.value} MPa·m^0.5 "
                f"out of physical bounds (0-15 MPa·m^0.5)"
            )
        
        # Young's modulus bounds (GPa)
        if not (0 < self.youngs_modulus.value < 1000):
            violations.append(
                f"Young's modulus {self.youngs_modulus.value} GPa out of physical bounds (0-1000 GPa)"
            )
        
        # Poisson ratio bounds (dimensionless)
        if not (-1 < self.poisson_ratio.value < 0.5):
            violations.append(
                f"Poisson ratio {self.poisson_ratio.value} out of physical bounds (-1 to 0.5)"
            )
        
        return violations


@dataclass
class ThermalProperties:
    """Thermal properties of ceramic materials."""
    
    thermal_conductivity_25C: ValueWithUncertainty  # W/m·K at 25°C
    thermal_conductivity_1000C: ValueWithUncertainty  # W/m·K at 1000°C
    thermal_expansion: ValueWithUncertainty  # 10⁻⁶/K
    specific_heat: ValueWithUncertainty  # J/kg·K
    melting_point: ValueWithUncertainty  # K
    
    def validate_bounds(self) -> List[str]:
        """Validate physical bounds for thermal properties."""
        violations = []
        
        # Thermal conductivity bounds (W/m·K)
        if not (0 < self.thermal_conductivity_25C.value < 500):
            violations.append(
                f"Thermal conductivity at 25°C {self.thermal_conductivity_25C.value} W/m·K "
                f"out of physical bounds (0-500 W/m·K)"
            )
        
        if not (0 < self.thermal_conductivity_1000C.value < 500):
            violations.append(
                f"Thermal conductivity at 1000°C {self.thermal_conductivity_1000C.value} W/m·K "
                f"out of physical bounds (0-500 W/m·K)"
            )
        
        # Thermal expansion bounds (10⁻⁶/K)
        if not (0 < self.thermal_expansion.value < 20):
            violations.append(
                f"Thermal expansion {self.thermal_expansion.value} 10⁻⁶/K "
                f"out of physical bounds (0-20 10⁻⁶/K)"
            )
        
        # Melting point bounds (K)
        if not (1000 < self.melting_point.value < 5000):
            violations.append(
                f"Melting point {self.melting_point.value} K out of physical bounds (1000-5000 K)"
            )
        
        return violations


@dataclass
class StructuralProperties:
    """Structural properties of ceramic materials."""
    
    density: ValueWithUncertainty  # g/cm³
    lattice_parameter: ValueWithUncertainty  # Å
    crystal_structure: str
    
    def validate_bounds(self) -> List[str]:
        """Validate physical bounds for structural properties."""
        violations = []
        
        # Density bounds (g/cm³)
        if not (1 < self.density.value < 20):
            violations.append(
                f"Density {self.density.value} g/cm³ out of physical bounds (1-20 g/cm³)"
            )
        
        # Lattice parameter bounds (Å)
        if not (2 < self.lattice_parameter.value < 10):
            violations.append(
                f"Lattice parameter {self.lattice_parameter.value} Å out of physical bounds (2-10 Å)"
            )
        
        return violations


@dataclass
class CeramicSystem:
    """Base ceramic system with properties and dopant handling."""
    
    ceramic_type: CeramicType
    mechanical: MechanicalProperties
    thermal: ThermalProperties
    structural: StructuralProperties
    dopant_element: Optional[str] = None
    dopant_concentration: Optional[float] = None  # atomic fraction (0.01 = 1%)
    
    # Supported dopant concentrations
    SUPPORTED_CONCENTRATIONS = [0.01, 0.02, 0.03, 0.05, 0.10]
    
    def __post_init__(self):
        """Validate ceramic system after initialization."""
        if self.dopant_concentration is not None:
            if self.dopant_concentration not in self.SUPPORTED_CONCENTRATIONS:
                warnings.warn(
                    f"Dopant concentration {self.dopant_concentration} not in standard set "
                    f"{self.SUPPORTED_CONCENTRATIONS}. Results may be less reliable."
                )
            
            if not (0 < self.dopant_concentration <= 0.10):
                raise ValueError(
                    f"Dopant concentration {self.dopant_concentration} out of valid range (0-0.10)"
                )
    
    def validate_all_properties(self) -> Dict[str, List[str]]:
        """
        Validate all property bounds.
        
        Returns:
            Dictionary mapping property category to list of violations
        """
        violations = {
            "mechanical": self.mechanical.validate_bounds(),
            "thermal": self.thermal.validate_bounds(),
            "structural": self.structural.validate_bounds(),
        }
        
        return {k: v for k, v in violations.items() if v}
    
    def is_doped(self) -> bool:
        """Check if system is doped."""
        return self.dopant_element is not None and self.dopant_concentration is not None
    
    def get_composition_string(self) -> str:
        """Get human-readable composition string."""
        base = self.ceramic_type.value
        
        if self.is_doped():
            concentration_pct = self.dopant_concentration * 100
            return f"{base}:{self.dopant_element}({concentration_pct:.1f}%)"
        
        return base
    
    def calculate_doped_property(
        self,
        base_value: float,
        dopant_effect: float,
        concentration: float
    ) -> float:
        """
        Calculate concentration-dependent property for doped system.
        
        Uses linear interpolation for dopant effects:
        property = base_value + dopant_effect * concentration
        
        Args:
            base_value: Property value for undoped system
            dopant_effect: Change in property per unit concentration
            concentration: Dopant concentration (atomic fraction)
        
        Returns:
            Calculated property value
        """
        return base_value + dopant_effect * concentration


class CeramicSystemFactory:
    """Factory for creating ceramic systems with baseline properties."""
    
    @staticmethod
    def create_sic() -> CeramicSystem:
        """
        Create SiC (Silicon Carbide) system with baseline properties.
        
        Properties from literature (Munro, 1997; Shaffer, 1964).
        """
        mechanical = MechanicalProperties(
            hardness=ValueWithUncertainty(28.0, 2.0, "GPa", "literature"),
            fracture_toughness=ValueWithUncertainty(4.6, 0.5, "MPa·m^0.5", "literature"),
            youngs_modulus=ValueWithUncertainty(410.0, 20.0, "GPa", "literature"),
            bulk_modulus=ValueWithUncertainty(220.0, 10.0, "GPa", "literature"),
            shear_modulus=ValueWithUncertainty(190.0, 10.0, "GPa", "literature"),
            poisson_ratio=ValueWithUncertainty(0.14, 0.02, "dimensionless", "literature"),
        )
        
        thermal = ThermalProperties(
            thermal_conductivity_25C=ValueWithUncertainty(120.0, 10.0, "W/m·K", "literature"),
            thermal_conductivity_1000C=ValueWithUncertainty(45.0, 5.0, "W/m·K", "literature"),
            thermal_expansion=ValueWithUncertainty(4.5, 0.3, "10⁻⁶/K", "literature"),
            specific_heat=ValueWithUncertainty(750.0, 50.0, "J/kg·K", "literature"),
            melting_point=ValueWithUncertainty(3003.0, 50.0, "K", "literature"),
        )
        
        structural = StructuralProperties(
            density=ValueWithUncertainty(3.21, 0.05, "g/cm³", "literature"),
            lattice_parameter=ValueWithUncertainty(4.36, 0.02, "Å", "literature"),
            crystal_structure="hexagonal (6H)",
        )
        
        return CeramicSystem(
            ceramic_type=CeramicType.SIC,
            mechanical=mechanical,
            thermal=thermal,
            structural=structural,
        )
    
    @staticmethod
    def create_b4c() -> CeramicSystem:
        """
        Create B₄C (Boron Carbide) system with baseline properties.
        
        Properties from literature (Thevenot, 1990; Domnich et al., 2011).
        """
        mechanical = MechanicalProperties(
            hardness=ValueWithUncertainty(30.0, 2.5, "GPa", "literature"),
            fracture_toughness=ValueWithUncertainty(3.5, 0.4, "MPa·m^0.5", "literature"),
            youngs_modulus=ValueWithUncertainty(450.0, 25.0, "GPa", "literature"),
            bulk_modulus=ValueWithUncertainty(247.0, 15.0, "GPa", "literature"),
            shear_modulus=ValueWithUncertainty(197.0, 12.0, "GPa", "literature"),
            poisson_ratio=ValueWithUncertainty(0.17, 0.02, "dimensionless", "literature"),
        )
        
        thermal = ThermalProperties(
            thermal_conductivity_25C=ValueWithUncertainty(30.0, 5.0, "W/m·K", "literature"),
            thermal_conductivity_1000C=ValueWithUncertainty(20.0, 3.0, "W/m·K", "literature"),
            thermal_expansion=ValueWithUncertainty(5.6, 0.4, "10⁻⁶/K", "literature"),
            specific_heat=ValueWithUncertainty(950.0, 60.0, "J/kg·K", "literature"),
            melting_point=ValueWithUncertainty(2723.0, 50.0, "K", "literature"),
        )
        
        structural = StructuralProperties(
            density=ValueWithUncertainty(2.52, 0.04, "g/cm³", "literature"),
            lattice_parameter=ValueWithUncertainty(5.60, 0.03, "Å", "literature"),
            crystal_structure="rhombohedral",
        )
        
        return CeramicSystem(
            ceramic_type=CeramicType.B4C,
            mechanical=mechanical,
            thermal=thermal,
            structural=structural,
        )

    @staticmethod
    def create_wc() -> CeramicSystem:
        """
        Create WC (Tungsten Carbide) system with baseline properties.
        
        Properties from literature (Upadhyaya, 2001; Exner, 1979).
        """
        mechanical = MechanicalProperties(
            hardness=ValueWithUncertainty(22.0, 2.0, "GPa", "literature"),
            fracture_toughness=ValueWithUncertainty(5.5, 0.6, "MPa·m^0.5", "literature"),
            youngs_modulus=ValueWithUncertainty(700.0, 30.0, "GPa", "literature"),
            bulk_modulus=ValueWithUncertainty(439.0, 20.0, "GPa", "literature"),
            shear_modulus=ValueWithUncertainty(282.0, 15.0, "GPa", "literature"),
            poisson_ratio=ValueWithUncertainty(0.24, 0.02, "dimensionless", "literature"),
        )
        
        thermal = ThermalProperties(
            thermal_conductivity_25C=ValueWithUncertainty(110.0, 10.0, "W/m·K", "literature"),
            thermal_conductivity_1000C=ValueWithUncertainty(65.0, 8.0, "W/m·K", "literature"),
            thermal_expansion=ValueWithUncertainty(5.2, 0.3, "10⁻⁶/K", "literature"),
            specific_heat=ValueWithUncertainty(203.0, 15.0, "J/kg·K", "literature"),
            melting_point=ValueWithUncertainty(3058.0, 50.0, "K", "literature"),
        )
        
        structural = StructuralProperties(
            density=ValueWithUncertainty(15.63, 0.10, "g/cm³", "literature"),
            lattice_parameter=ValueWithUncertainty(2.91, 0.02, "Å", "literature"),
            crystal_structure="hexagonal",
        )
        
        return CeramicSystem(
            ceramic_type=CeramicType.WC,
            mechanical=mechanical,
            thermal=thermal,
            structural=structural,
        )
    
    @staticmethod
    def create_tic() -> CeramicSystem:
        """
        Create TiC (Titanium Carbide) system with baseline properties.
        
        Properties from literature (Pierson, 1996; Toth, 1971).
        """
        mechanical = MechanicalProperties(
            hardness=ValueWithUncertainty(28.0, 2.5, "GPa", "literature"),
            fracture_toughness=ValueWithUncertainty(4.0, 0.5, "MPa·m^0.5", "literature"),
            youngs_modulus=ValueWithUncertainty(460.0, 25.0, "GPa", "literature"),
            bulk_modulus=ValueWithUncertainty(242.0, 15.0, "GPa", "literature"),
            shear_modulus=ValueWithUncertainty(188.0, 12.0, "GPa", "literature"),
            poisson_ratio=ValueWithUncertainty(0.19, 0.02, "dimensionless", "literature"),
        )
        
        thermal = ThermalProperties(
            thermal_conductivity_25C=ValueWithUncertainty(27.0, 4.0, "W/m·K", "literature"),
            thermal_conductivity_1000C=ValueWithUncertainty(22.0, 3.0, "W/m·K", "literature"),
            thermal_expansion=ValueWithUncertainty(7.4, 0.4, "10⁻⁶/K", "literature"),
            specific_heat=ValueWithUncertainty(544.0, 40.0, "J/kg·K", "literature"),
            melting_point=ValueWithUncertainty(3433.0, 50.0, "K", "literature"),
        )
        
        structural = StructuralProperties(
            density=ValueWithUncertainty(4.93, 0.05, "g/cm³", "literature"),
            lattice_parameter=ValueWithUncertainty(4.33, 0.02, "Å", "literature"),
            crystal_structure="cubic (NaCl-type)",
        )
        
        return CeramicSystem(
            ceramic_type=CeramicType.TIC,
            mechanical=mechanical,
            thermal=thermal,
            structural=structural,
        )
    
    @staticmethod
    def create_al2o3() -> CeramicSystem:
        """
        Create Al₂O₃ (Alumina) system with baseline properties.
        
        Properties from literature (Munro, 1997; Kingery et al., 1976).
        """
        mechanical = MechanicalProperties(
            hardness=ValueWithUncertainty(20.0, 2.0, "GPa", "literature"),
            fracture_toughness=ValueWithUncertainty(4.0, 0.5, "MPa·m^0.5", "literature"),
            youngs_modulus=ValueWithUncertainty(380.0, 20.0, "GPa", "literature"),
            bulk_modulus=ValueWithUncertainty(252.0, 15.0, "GPa", "literature"),
            shear_modulus=ValueWithUncertainty(152.0, 10.0, "GPa", "literature"),
            poisson_ratio=ValueWithUncertainty(0.25, 0.02, "dimensionless", "literature"),
        )
        
        thermal = ThermalProperties(
            thermal_conductivity_25C=ValueWithUncertainty(35.0, 5.0, "W/m·K", "literature"),
            thermal_conductivity_1000C=ValueWithUncertainty(6.0, 1.0, "W/m·K", "literature"),
            thermal_expansion=ValueWithUncertainty(8.1, 0.4, "10⁻⁶/K", "literature"),
            specific_heat=ValueWithUncertainty(880.0, 50.0, "J/kg·K", "literature"),
            melting_point=ValueWithUncertainty(2327.0, 50.0, "K", "literature"),
        )
        
        structural = StructuralProperties(
            density=ValueWithUncertainty(3.98, 0.05, "g/cm³", "literature"),
            lattice_parameter=ValueWithUncertainty(4.76, 0.02, "Å", "literature"),
            crystal_structure="hexagonal (corundum)",
        )
        
        return CeramicSystem(
            ceramic_type=CeramicType.AL2O3,
            mechanical=mechanical,
            thermal=thermal,
            structural=structural,
        )
    
    @staticmethod
    def create_doped_system(
        base_ceramic: CeramicSystem,
        dopant_element: str,
        dopant_concentration: float,
        property_modifications: Optional[Dict[str, float]] = None
    ) -> CeramicSystem:
        """
        Create a doped ceramic system with modified properties.
        
        Args:
            base_ceramic: Base ceramic system
            dopant_element: Dopant element symbol
            dopant_concentration: Dopant concentration (atomic fraction)
            property_modifications: Optional dict of property modifications
                                  (property_name -> change per unit concentration)
        
        Returns:
            Doped ceramic system
        """
        # Validate concentration range
        if not (0 < dopant_concentration <= 0.10):
            raise ValueError(
                f"Dopant concentration {dopant_concentration} out of valid range (0-0.10)"
            )
        
        if dopant_concentration not in CeramicSystem.SUPPORTED_CONCENTRATIONS:
            warnings.warn(
                f"Dopant concentration {dopant_concentration} not in standard set. "
                f"Property estimates may be less reliable."
            )
        
        # Create copy of base system
        import copy
        doped_system = copy.deepcopy(base_ceramic)
        
        # Set dopant information
        doped_system.dopant_element = dopant_element
        doped_system.dopant_concentration = dopant_concentration
        
        # Apply property modifications if provided
        if property_modifications:
            for prop_name, effect_per_concentration in property_modifications.items():
                # Calculate modified value
                if hasattr(doped_system.mechanical, prop_name):
                    prop = getattr(doped_system.mechanical, prop_name)
                    modified_value = doped_system.calculate_doped_property(
                        prop.value, effect_per_concentration, dopant_concentration
                    )
                    prop.value = modified_value
                    prop.source = "calculated_doped"
                    # Increase uncertainty for doped systems
                    prop.uncertainty *= 1.5
                
                elif hasattr(doped_system.thermal, prop_name):
                    prop = getattr(doped_system.thermal, prop_name)
                    modified_value = doped_system.calculate_doped_property(
                        prop.value, effect_per_concentration, dopant_concentration
                    )
                    prop.value = modified_value
                    prop.source = "calculated_doped"
                    prop.uncertainty *= 1.5
                
                elif hasattr(doped_system.structural, prop_name):
                    prop = getattr(doped_system.structural, prop_name)
                    modified_value = doped_system.calculate_doped_property(
                        prop.value, effect_per_concentration, dopant_concentration
                    )
                    prop.value = modified_value
                    prop.source = "calculated_doped"
                    prop.uncertainty *= 1.5
        
        return doped_system
    
    @classmethod
    def create_system(cls, ceramic_type: CeramicType) -> CeramicSystem:
        """
        Create ceramic system by type.
        
        Args:
            ceramic_type: Type of ceramic system
        
        Returns:
            Ceramic system with baseline properties
        """
        factory_methods = {
            CeramicType.SIC: cls.create_sic,
            CeramicType.B4C: cls.create_b4c,
            CeramicType.WC: cls.create_wc,
            CeramicType.TIC: cls.create_tic,
            CeramicType.AL2O3: cls.create_al2o3,
        }
        
        if ceramic_type not in factory_methods:
            raise ValueError(f"Unsupported ceramic type: {ceramic_type}")
        
        return factory_methods[ceramic_type]()
