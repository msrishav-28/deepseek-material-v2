"""
Composition-based descriptor calculator for materials.

This module calculates composition-based features from chemical formulas
using pymatgen Composition objects. These descriptors are used as ML features
for predicting material properties.
"""

from typing import Dict, Optional
import warnings
import math

import numpy as np
from pymatgen.core import Composition, Element


class CompositionDescriptorCalculator:
    """
    Calculate composition-based material descriptors for ML features.
    
    Computes features including:
    - Number of elements and atoms
    - Composition entropy (Shannon entropy)
    - Atomic property statistics (radius, mass, number, electronegativity)
    - Weighted atomic properties
    """
    
    def __init__(self):
        """Initialize composition descriptor calculator."""
        pass
    
    def calculate_descriptors(self, formula: str) -> Dict[str, float]:
        """
        Calculate all composition descriptors for a chemical formula.
        
        Args:
            formula: Chemical formula string (e.g., "SiC", "TiC", "B4C")
            
        Returns:
            Dictionary mapping descriptor names to values
            
        Examples:
            >>> calc = CompositionDescriptorCalculator()
            >>> descriptors = calc.calculate_descriptors("SiC")
            >>> descriptors['num_elements']
            2.0
            >>> descriptors['composition_entropy'] > 0
            True
        """
        try:
            composition = Composition(formula)
        except Exception as e:
            warnings.warn(f"Failed to parse formula '{formula}': {e}")
            return {}
        
        descriptors = {}
        
        # Basic composition features
        descriptors['num_elements'] = float(len(composition.elements))
        descriptors['num_atoms'] = float(composition.num_atoms)
        
        # Composition entropy
        entropy = self.calculate_composition_entropy(composition)
        if entropy is not None:
            descriptors['composition_entropy'] = entropy
        
        # Electronegativity features
        en_features = self.calculate_electronegativity_features(composition)
        descriptors.update(en_features)
        
        # Atomic property features
        atomic_features = self.calculate_atomic_property_features(composition)
        descriptors.update(atomic_features)
        
        return descriptors
    
    def calculate_composition_entropy(self, composition: Composition) -> Optional[float]:
        """
        Calculate Shannon entropy of composition.
        
        Shannon entropy: -Σ(f_i * log(f_i)) where f_i is atomic fraction
        
        Args:
            composition: Pymatgen Composition object
            
        Returns:
            Composition entropy value, or None if calculation fails
            
        Examples:
            >>> calc = CompositionDescriptorCalculator()
            >>> comp = Composition("SiC")
            >>> entropy = calc.calculate_composition_entropy(comp)
            >>> 0 < entropy < 1
            True
        """
        try:
            # Get atomic fractions
            fractions = composition.fractional_composition.as_dict()
            
            # Calculate Shannon entropy: -Σ(f_i * log(f_i))
            entropy = 0.0
            for element, fraction in fractions.items():
                if fraction > 0:
                    entropy -= fraction * math.log(fraction)
            
            return float(entropy)
        except Exception as e:
            warnings.warn(f"Failed to calculate composition entropy: {e}")
            return None
    
    def calculate_electronegativity_features(self, composition: Composition) -> Dict[str, float]:
        """
        Calculate electronegativity-based features.
        
        Computes mean, standard deviation, and delta (max - min) of
        Pauling electronegativities weighted by atomic fractions.
        
        Args:
            composition: Pymatgen Composition object
            
        Returns:
            Dictionary with electronegativity features
            
        Examples:
            >>> calc = CompositionDescriptorCalculator()
            >>> comp = Composition("SiC")
            >>> features = calc.calculate_electronegativity_features(comp)
            >>> 'mean_electronegativity' in features
            True
        """
        features = {}
        
        try:
            # Get elements and their fractions
            elements = composition.elements
            fractions = [composition.get_atomic_fraction(el) for el in elements]
            
            # Get Pauling electronegativities
            electronegativities = []
            for el in elements:
                try:
                    en = el.X  # Pauling electronegativity
                    if en is not None:
                        electronegativities.append(en)
                    else:
                        # Use a default value if not available
                        electronegativities.append(1.8)
                except (AttributeError, ValueError):
                    electronegativities.append(1.8)
            
            if electronegativities:
                # Weighted mean
                mean_en = np.average(electronegativities, weights=fractions)
                features['mean_electronegativity'] = float(mean_en)
                
                # Weighted standard deviation
                variance = np.average((np.array(electronegativities) - mean_en)**2, weights=fractions)
                features['std_electronegativity'] = float(np.sqrt(variance))
                
                # Delta (max - min)
                features['delta_electronegativity'] = float(max(electronegativities) - min(electronegativities))
        
        except Exception as e:
            warnings.warn(f"Failed to calculate electronegativity features: {e}")
        
        return features
    
    def calculate_atomic_property_features(self, composition: Composition) -> Dict[str, float]:
        """
        Calculate features from atomic properties.
        
        Computes mean and weighted statistics for:
        - Atomic radius
        - Atomic mass
        - Atomic number
        
        Args:
            composition: Pymatgen Composition object
            
        Returns:
            Dictionary with atomic property features
            
        Examples:
            >>> calc = CompositionDescriptorCalculator()
            >>> comp = Composition("SiC")
            >>> features = calc.calculate_atomic_property_features(comp)
            >>> 'mean_atomic_radius' in features
            True
        """
        features = {}
        
        try:
            # Get elements and their fractions
            elements = composition.elements
            fractions = [composition.get_atomic_fraction(el) for el in elements]
            
            # Atomic radii (in Angstroms)
            radii = []
            for el in elements:
                try:
                    # Use atomic radius (metallic, covalent, or van der Waals)
                    radius = el.atomic_radius
                    if radius is not None:
                        radii.append(radius)
                    else:
                        # Fallback to average radius
                        radii.append(1.5)
                except (AttributeError, ValueError):
                    radii.append(1.5)
            
            if radii:
                mean_radius = np.average(radii, weights=fractions)
                features['mean_atomic_radius'] = float(mean_radius)
                
                variance = np.average((np.array(radii) - mean_radius)**2, weights=fractions)
                features['std_atomic_radius'] = float(np.sqrt(variance))
                
                features['delta_atomic_radius'] = float(max(radii) - min(radii))
            
            # Atomic masses
            masses = [el.atomic_mass for el in elements]
            if masses:
                mean_mass = np.average(masses, weights=fractions)
                features['mean_atomic_mass'] = float(mean_mass)
                
                variance = np.average((np.array(masses) - mean_mass)**2, weights=fractions)
                features['std_atomic_mass'] = float(np.sqrt(variance))
            
            # Atomic numbers
            atomic_numbers = [el.Z for el in elements]
            if atomic_numbers:
                mean_z = np.average(atomic_numbers, weights=fractions)
                features['mean_atomic_number'] = float(mean_z)
                
                # Weighted atomic number (sum of Z * fraction)
                weighted_z = sum(z * f for z, f in zip(atomic_numbers, fractions))
                features['weighted_atomic_number'] = float(weighted_z)
        
        except Exception as e:
            warnings.warn(f"Failed to calculate atomic property features: {e}")
        
        return features
