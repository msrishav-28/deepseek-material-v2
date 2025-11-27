"""
Structure-derived descriptor calculator for materials.

This module calculates structure-derived features from material properties
such as bulk modulus, shear modulus, formation energy, and density.
These descriptors are used as ML features for predicting material properties.
"""

from typing import Dict, Optional
import warnings
import math

import numpy as np


class StructureDescriptorCalculator:
    """
    Calculate structure-derived material descriptors for ML features.
    
    Computes features including:
    - Pugh ratio (B/G) for ductility assessment
    - Young's modulus and Poisson ratio from elastic moduli
    - Energy density from formation energy and density
    - Ductility indicators
    """
    
    def __init__(self):
        """Initialize structure descriptor calculator."""
        pass
    
    def calculate_descriptors(self, properties: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate all structure descriptors from material properties.
        
        Args:
            properties: Dictionary of material properties including:
                - bulk_modulus: Bulk modulus in GPa
                - shear_modulus: Shear modulus in GPa
                - formation_energy: Formation energy in eV/atom
                - density: Density in g/cm³
                
        Returns:
            Dictionary mapping descriptor names to values
            
        Examples:
            >>> calc = StructureDescriptorCalculator()
            >>> props = {'bulk_modulus': 220, 'shear_modulus': 190}
            >>> descriptors = calc.calculate_descriptors(props)
            >>> 'pugh_ratio' in descriptors
            True
        """
        descriptors = {}
        
        # Extract properties with safe defaults
        bulk_modulus = properties.get('bulk_modulus')
        shear_modulus = properties.get('shear_modulus')
        formation_energy = properties.get('formation_energy')
        density = properties.get('density')
        
        # Calculate Pugh ratio if both moduli available
        if bulk_modulus is not None and shear_modulus is not None:
            pugh_ratio = self.calculate_pugh_ratio(bulk_modulus, shear_modulus)
            if pugh_ratio is not None:
                descriptors['pugh_ratio'] = pugh_ratio
                
                # Ductility indicator based on Pugh ratio
                # B/G > 1.75 indicates ductile behavior
                descriptors['is_ductile'] = 1.0 if pugh_ratio > 1.75 else 0.0
        
        # Calculate elastic properties if both moduli available
        if bulk_modulus is not None and shear_modulus is not None:
            elastic_props = self.calculate_elastic_properties(bulk_modulus, shear_modulus)
            descriptors.update(elastic_props)
        
        # Calculate energy density if both formation energy and density available
        if formation_energy is not None and density is not None:
            energy_density = self.calculate_energy_density(formation_energy, density)
            if energy_density is not None:
                descriptors['energy_density'] = energy_density
        
        return descriptors
    
    def calculate_pugh_ratio(self, bulk_modulus: float, shear_modulus: float) -> Optional[float]:
        """
        Calculate Pugh ratio (B/G) for ductility assessment.
        
        The Pugh ratio is the ratio of bulk modulus to shear modulus.
        Values > 1.75 typically indicate ductile behavior, while values < 1.75
        indicate brittle behavior.
        
        Args:
            bulk_modulus: Bulk modulus in GPa
            shear_modulus: Shear modulus in GPa
            
        Returns:
            Pugh ratio (B/G), or None if calculation fails or inputs invalid
            
        Examples:
            >>> calc = StructureDescriptorCalculator()
            >>> ratio = calc.calculate_pugh_ratio(220, 190)
            >>> 1.0 < ratio < 2.0
            True
        """
        try:
            # Validate inputs
            if bulk_modulus <= 0 or shear_modulus <= 0:
                warnings.warn(f"Invalid moduli values: B={bulk_modulus}, G={shear_modulus}")
                return None
            
            if not math.isfinite(bulk_modulus) or not math.isfinite(shear_modulus):
                warnings.warn(f"Non-finite moduli values: B={bulk_modulus}, G={shear_modulus}")
                return None
            
            pugh_ratio = bulk_modulus / shear_modulus
            
            # Validate result
            if not math.isfinite(pugh_ratio):
                warnings.warn(f"Non-finite Pugh ratio calculated: {pugh_ratio}")
                return None
            
            return float(pugh_ratio)
        
        except Exception as e:
            warnings.warn(f"Failed to calculate Pugh ratio: {e}")
            return None
    
    def calculate_elastic_properties(
        self, 
        bulk_modulus: float, 
        shear_modulus: float
    ) -> Dict[str, float]:
        """
        Calculate Young's modulus and Poisson ratio from elastic moduli.
        
        Formulas:
        - Young's modulus: E = 9BG / (3B + G)
        - Poisson ratio: ν = (3B - 2G) / (2(3B + G))
        
        Args:
            bulk_modulus: Bulk modulus in GPa
            shear_modulus: Shear modulus in GPa
            
        Returns:
            Dictionary with 'youngs_modulus' and 'poisson_ratio'
            
        Examples:
            >>> calc = StructureDescriptorCalculator()
            >>> props = calc.calculate_elastic_properties(220, 190)
            >>> 'youngs_modulus' in props
            True
            >>> -1 <= props['poisson_ratio'] <= 0.5
            True
        """
        properties = {}
        
        try:
            # Validate inputs
            if bulk_modulus <= 0 or shear_modulus <= 0:
                warnings.warn(f"Invalid moduli values: B={bulk_modulus}, G={shear_modulus}")
                return properties
            
            if not math.isfinite(bulk_modulus) or not math.isfinite(shear_modulus):
                warnings.warn(f"Non-finite moduli values: B={bulk_modulus}, G={shear_modulus}")
                return properties
            
            # Calculate Young's modulus: E = 9BG / (3B + G)
            denominator = 3 * bulk_modulus + shear_modulus
            if denominator > 0:
                youngs_modulus = (9 * bulk_modulus * shear_modulus) / denominator
                
                # Validate Young's modulus
                if math.isfinite(youngs_modulus) and youngs_modulus > 0:
                    properties['youngs_modulus'] = float(youngs_modulus)
                else:
                    warnings.warn(f"Invalid Young's modulus calculated: {youngs_modulus}")
            
            # Calculate Poisson ratio: ν = (3B - 2G) / (2(3B + G))
            numerator = 3 * bulk_modulus - 2 * shear_modulus
            denominator = 2 * (3 * bulk_modulus + shear_modulus)
            
            if denominator > 0:
                poisson_ratio = numerator / denominator
                
                # Validate Poisson ratio (physical bounds: -1 to 0.5)
                if math.isfinite(poisson_ratio):
                    if -1.0 <= poisson_ratio <= 0.5:
                        properties['poisson_ratio'] = float(poisson_ratio)
                    else:
                        warnings.warn(
                            f"Poisson ratio {poisson_ratio} outside physical bounds [-1, 0.5]"
                        )
                else:
                    warnings.warn(f"Non-finite Poisson ratio calculated: {poisson_ratio}")
        
        except Exception as e:
            warnings.warn(f"Failed to calculate elastic properties: {e}")
        
        return properties
    
    def calculate_energy_density(
        self, 
        formation_energy: float, 
        density: float
    ) -> Optional[float]:
        """
        Calculate energy density from formation energy and density.
        
        Energy density = |formation_energy| / density
        
        This represents the energy per unit volume, useful for comparing
        materials with different densities.
        
        Args:
            formation_energy: Formation energy in eV/atom
            density: Density in g/cm³
            
        Returns:
            Energy density in eV·cm³/atom·g, or None if calculation fails
            
        Examples:
            >>> calc = StructureDescriptorCalculator()
            >>> ed = calc.calculate_energy_density(-2.5, 3.2)
            >>> ed > 0
            True
        """
        try:
            # Validate inputs
            if density <= 0:
                warnings.warn(f"Invalid density value: {density}")
                return None
            
            if not math.isfinite(formation_energy) or not math.isfinite(density):
                warnings.warn(
                    f"Non-finite values: formation_energy={formation_energy}, density={density}"
                )
                return None
            
            # Calculate energy density using absolute value of formation energy
            energy_density = abs(formation_energy) / density
            
            # Validate result
            if not math.isfinite(energy_density) or energy_density < 0:
                warnings.warn(f"Invalid energy density calculated: {energy_density}")
                return None
            
            return float(energy_density)
        
        except Exception as e:
            warnings.warn(f"Failed to calculate energy density: {e}")
            return None
