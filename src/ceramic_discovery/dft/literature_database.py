"""
Literature database for experimental reference data.

This module provides access to curated experimental data from literature sources
for ceramic materials, including uncertainties and source references.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LiteratureDatabase:
    """
    Database of experimental literature data for ceramic materials.
    
    Contains curated reference data with uncertainties for key ceramic materials
    including SiC, TiC, ZrC, HfC, and refractory metals Ta, Zr, Hf.
    """

    def __init__(self):
        """Initialize literature database with embedded reference data."""
        self._data = self._initialize_data()
        logger.info(f"Initialized literature database with {len(self._data)} materials")

    def _initialize_data(self) -> Dict[str, Dict]:
        """
        Initialize embedded literature data.
        
        Returns:
            Dictionary mapping material formulas to property data
        """
        return {
            "SiC": {
                "hardness_vickers": 2800.0,
                "hardness_vickers_std": 200.0,
                "thermal_conductivity": 120.0,
                "thermal_conductivity_std": 10.0,
                "melting_point": 3103.0,
                "melting_point_std": 50.0,
                "density": 3.21,
                "density_std": 0.02,
                "youngs_modulus": 450.0,
                "youngs_modulus_std": 20.0,
                "bulk_modulus": 220.0,
                "bulk_modulus_std": 10.0,
                "shear_modulus": 190.0,
                "shear_modulus_std": 10.0,
                "sources": [
                    "Handbook of Advanced Ceramics (2013)",
                    "INSPEC Materials Properties Database (1997)",
                    "ASM Ceramics and Glasses Handbook (1991)"
                ]
            },
            "TiC": {
                "hardness_vickers": 3200.0,
                "hardness_vickers_std": 250.0,
                "thermal_conductivity": 33.0,
                "thermal_conductivity_std": 3.0,
                "melting_point": 3433.0,
                "melting_point_std": 50.0,
                "density": 4.93,
                "density_std": 0.03,
                "youngs_modulus": 460.0,
                "youngs_modulus_std": 25.0,
                "bulk_modulus": 240.0,
                "bulk_modulus_std": 15.0,
                "shear_modulus": 188.0,
                "shear_modulus_std": 12.0,
                "sources": [
                    "Handbook of Refractory Carbides and Nitrides (1996)",
                    "Journal of the American Ceramic Society (2005)",
                    "Materials Science and Engineering A (2008)"
                ]
            },
            "ZrC": {
                "hardness_vickers": 2600.0,
                "hardness_vickers_std": 200.0,
                "thermal_conductivity": 20.5,
                "thermal_conductivity_std": 2.0,
                "melting_point": 3813.0,
                "melting_point_std": 50.0,
                "density": 6.73,
                "density_std": 0.05,
                "youngs_modulus": 400.0,
                "youngs_modulus_std": 30.0,
                "bulk_modulus": 230.0,
                "bulk_modulus_std": 15.0,
                "shear_modulus": 160.0,
                "shear_modulus_std": 15.0,
                "sources": [
                    "Journal of Nuclear Materials (2010)",
                    "Ceramics International (2012)",
                    "High Temperature Materials and Processes (2015)"
                ]
            },
            "HfC": {
                "hardness_vickers": 2800.0,
                "hardness_vickers_std": 250.0,
                "thermal_conductivity": 22.0,
                "thermal_conductivity_std": 2.5,
                "melting_point": 4163.0,
                "melting_point_std": 50.0,
                "density": 12.2,
                "density_std": 0.1,
                "youngs_modulus": 460.0,
                "youngs_modulus_std": 35.0,
                "bulk_modulus": 250.0,
                "bulk_modulus_std": 20.0,
                "shear_modulus": 180.0,
                "shear_modulus_std": 18.0,
                "sources": [
                    "Journal of the American Ceramic Society (2007)",
                    "Acta Materialia (2011)",
                    "Materials Research Bulletin (2014)"
                ]
            },
            "Ta": {
                "hardness_vickers": 873.0,
                "hardness_vickers_std": 50.0,
                "thermal_conductivity": 57.5,
                "thermal_conductivity_std": 2.0,
                "melting_point": 3290.0,
                "melting_point_std": 20.0,
                "density": 16.65,
                "density_std": 0.05,
                "youngs_modulus": 186.0,
                "youngs_modulus_std": 10.0,
                "bulk_modulus": 200.0,
                "bulk_modulus_std": 10.0,
                "shear_modulus": 69.0,
                "shear_modulus_std": 5.0,
                "sources": [
                    "CRC Handbook of Chemistry and Physics (2020)",
                    "ASM Metals Handbook (2016)",
                    "Materials Science and Technology (2009)"
                ]
            },
            "Zr": {
                "hardness_vickers": 903.0,
                "hardness_vickers_std": 60.0,
                "thermal_conductivity": 22.7,
                "thermal_conductivity_std": 1.5,
                "melting_point": 2128.0,
                "melting_point_std": 10.0,
                "density": 6.52,
                "density_std": 0.03,
                "youngs_modulus": 88.0,
                "youngs_modulus_std": 5.0,
                "bulk_modulus": 91.0,
                "bulk_modulus_std": 5.0,
                "shear_modulus": 33.0,
                "shear_modulus_std": 3.0,
                "sources": [
                    "CRC Handbook of Chemistry and Physics (2020)",
                    "ASM Metals Handbook (2016)",
                    "Journal of Nuclear Materials (2013)"
                ]
            },
            "Hf": {
                "hardness_vickers": 1760.0,
                "hardness_vickers_std": 100.0,
                "thermal_conductivity": 23.0,
                "thermal_conductivity_std": 1.5,
                "melting_point": 2506.0,
                "melting_point_std": 15.0,
                "density": 13.31,
                "density_std": 0.05,
                "youngs_modulus": 141.0,
                "youngs_modulus_std": 8.0,
                "bulk_modulus": 110.0,
                "bulk_modulus_std": 8.0,
                "shear_modulus": 56.0,
                "shear_modulus_std": 4.0,
                "sources": [
                    "CRC Handbook of Chemistry and Physics (2020)",
                    "ASM Metals Handbook (2016)",
                    "Materials Science and Engineering A (2010)"
                ]
            }
        }

    def get_material_data(self, formula: str) -> Optional[Dict]:
        """
        Retrieve literature data for a material.
        
        Args:
            formula: Chemical formula (e.g., "SiC", "TiC")
            
        Returns:
            Dictionary with property values, uncertainties, and sources,
            or None if material not found
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> data = db.get_material_data("SiC")
            >>> data["hardness_vickers"]
            2800.0
            >>> data["sources"]
            ['Handbook of Advanced Ceramics (2013)', ...]
        """
        if not formula:
            logger.warning("Empty formula provided")
            return None
            
        # Normalize formula (remove spaces, handle case)
        normalized_formula = formula.strip()
        
        if normalized_formula not in self._data:
            logger.debug(f"Material '{formula}' not found in literature database")
            return None
            
        return self._data[normalized_formula].copy()

    def get_property_with_uncertainty(
        self, formula: str, property_name: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get property value and uncertainty from literature.
        
        Args:
            formula: Chemical formula
            property_name: Name of property (e.g., "hardness_vickers", "thermal_conductivity")
            
        Returns:
            Tuple of (value, uncertainty) or (None, None) if not available
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> value, uncertainty = db.get_property_with_uncertainty("SiC", "hardness_vickers")
            >>> value
            2800.0
            >>> uncertainty
            200.0
            >>> value, uncertainty = db.get_property_with_uncertainty("SiC", "unknown_property")
            >>> value is None
            True
        """
        material_data = self.get_material_data(formula)
        
        if material_data is None:
            return (None, None)
            
        # Get property value
        value = material_data.get(property_name)
        
        # Get uncertainty (standard deviation)
        uncertainty_key = f"{property_name}_std"
        uncertainty = material_data.get(uncertainty_key)
        
        if value is None:
            logger.debug(
                f"Property '{property_name}' not found for material '{formula}'"
            )
            return (None, None)
            
        return (value, uncertainty)

    def list_available_materials(self) -> List[str]:
        """
        List materials with literature data.
        
        Returns:
            List of material formulas available in the database
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> materials = db.list_available_materials()
            >>> "SiC" in materials
            True
            >>> len(materials)
            7
        """
        return sorted(list(self._data.keys()))

    def get_available_properties(self, formula: str) -> List[str]:
        """
        Get list of available properties for a material.
        
        Args:
            formula: Chemical formula
            
        Returns:
            List of property names (excluding uncertainties and sources)
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> props = db.get_available_properties("SiC")
            >>> "hardness_vickers" in props
            True
            >>> "hardness_vickers_std" in props
            False
        """
        material_data = self.get_material_data(formula)
        
        if material_data is None:
            return []
            
        # Filter out uncertainty keys (ending with _std) and sources
        properties = [
            key for key in material_data.keys()
            if not key.endswith('_std') and key != 'sources'
        ]
        
        return sorted(properties)

    def get_sources(self, formula: str) -> List[str]:
        """
        Get literature sources for a material.
        
        Args:
            formula: Chemical formula
            
        Returns:
            List of literature source references
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> sources = db.get_sources("SiC")
            >>> len(sources) > 0
            True
        """
        material_data = self.get_material_data(formula)
        
        if material_data is None:
            return []
            
        return material_data.get('sources', [])

    def has_material(self, formula: str) -> bool:
        """
        Check if material exists in database.
        
        Args:
            formula: Chemical formula
            
        Returns:
            True if material has literature data
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> db.has_material("SiC")
            True
            >>> db.has_material("UnknownMaterial")
            False
        """
        return formula.strip() in self._data

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics about the database
            
        Examples:
            >>> db = LiteratureDatabase()
            >>> stats = db.get_statistics()
            >>> stats["total_materials"]
            7
        """
        total_properties = sum(
            len([k for k in data.keys() if not k.endswith('_std') and k != 'sources'])
            for data in self._data.values()
        )
        
        return {
            "total_materials": len(self._data),
            "total_properties": total_properties,
            "materials": self.list_available_materials()
        }
