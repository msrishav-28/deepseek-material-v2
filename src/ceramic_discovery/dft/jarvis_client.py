"""
JARVIS-DFT data client for loading and parsing carbide materials.

This module provides functionality to load JARVIS-DFT database files,
extract carbide materials, and parse material properties.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ceramic_discovery.dft.materials_project_client import MaterialData
from ceramic_discovery.utils.data_conversion import safe_float, safe_int

logger = logging.getLogger(__name__)


class JarvisClient:
    """Client for loading and parsing JARVIS-DFT database files."""

    def __init__(self, jarvis_file_path: str):
        """
        Initialize JARVIS client with path to database file.

        Args:
            jarvis_file_path: Path to JARVIS JSON database file

        Raises:
            FileNotFoundError: If JARVIS file does not exist
            ValueError: If file path is empty
        """
        if not jarvis_file_path:
            raise ValueError("JARVIS file path cannot be empty")

        self.jarvis_file_path = Path(jarvis_file_path)

        if not self.jarvis_file_path.exists():
            raise FileNotFoundError(f"JARVIS file not found: {jarvis_file_path}")

        self.data: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load JARVIS data from JSON file."""
        try:
            with open(self.jarvis_file_path, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} materials from JARVIS database")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in JARVIS file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading JARVIS file: {e}")

    def load_carbides(self, metal_elements: Optional[Set[str]] = None) -> List[MaterialData]:
        """
        Extract carbide materials containing specified metals.

        Filters JARVIS database for materials that:
        1. Contain carbon (C)
        2. Contain at least one specified metal element (if provided)
        3. Have valid property data

        Args:
            metal_elements: Set of metal element symbols to filter for.
                          If None, returns all carbides.

        Returns:
            List of MaterialData objects for carbide materials

        Examples:
            >>> client = JarvisClient("jarvis.json")
            >>> carbides = client.load_carbides({"Si", "Ti", "Zr"})
            >>> len(carbides)
            400
        """
        carbides = []

        for entry in self.data:
            # Extract formula from entry
            formula = self.parse_formula_from_jid(entry.get("jid", ""))
            if not formula:
                # Try direct formula field
                formula = entry.get("formula", "")

            if not formula:
                continue

            # Check if material is a carbide
            if not self._is_carbide(formula, metal_elements):
                continue

            # Extract properties
            try:
                material_data = self._create_material_data(entry, formula)
                if material_data:
                    carbides.append(material_data)
            except Exception as e:
                logger.warning(f"Error processing JARVIS entry {entry.get('jid')}: {e}")
                continue

        logger.info(f"Extracted {len(carbides)} carbide materials from JARVIS")
        return carbides

    def _is_carbide(self, formula: str, metal_elements: Optional[Set[str]] = None) -> bool:
        """
        Check if formula represents a carbide material.

        Args:
            formula: Chemical formula
            metal_elements: Optional set of metal elements to require

        Returns:
            True if formula is a carbide with required metals
        """
        # Parse formula to extract elements
        # Simple element extraction: look for uppercase letters followed by optional lowercase
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        # Must contain carbon as an element
        if 'C' not in elements:
            return False

        # If no metal filter specified, any carbide is valid
        if metal_elements is None:
            return True

        # Check if any specified metal is present as an element
        for metal in metal_elements:
            if metal in elements:
                return True

        return False

    def parse_formula_from_jid(self, jid: str) -> Optional[str]:
        """
        Parse chemical formula from JARVIS identifier.

        JARVIS jid format examples:
        - "JVASP-12345"
        - "JVASP-12345-SiC"
        - "JVASP-SiC-12345"

        Args:
            jid: JARVIS identifier string

        Returns:
            Chemical formula if found, None otherwise

        Examples:
            >>> client = JarvisClient("jarvis.json")
            >>> client.parse_formula_from_jid("JVASP-12345-SiC")
            'SiC'
            >>> client.parse_formula_from_jid("JVASP-12345")
            None
        """
        if not jid or not isinstance(jid, str):
            return None

        # Split by hyphen
        parts = jid.split('-')

        # Look for formula part (contains letters and numbers, not just numbers)
        for part in parts:
            if part and not part.isdigit() and part != "JVASP":
                # Check if it looks like a chemical formula
                if any(c.isupper() for c in part):
                    return part

        return None

    def extract_properties(self, jarvis_entry: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract properties from JARVIS entry.

        Extracts:
        - formation_energy_per_atom (eV/atom)
        - band_gap (eV)
        - bulk_modulus (GPa)
        - shear_modulus (GPa)
        - energy_above_hull (eV/atom)
        - density (g/cm³)

        Args:
            jarvis_entry: Raw JARVIS database entry

        Returns:
            Dictionary of extracted properties with float values

        Examples:
            >>> properties = client.extract_properties(entry)
            >>> properties['formation_energy_per_atom']
            -0.65
        """
        properties = {}

        # Formation energy per atom
        formation_energy = safe_float(jarvis_entry.get("formation_energy_peratom"))
        if formation_energy is not None:
            properties["formation_energy_per_atom"] = formation_energy

        # Band gap
        band_gap = safe_float(jarvis_entry.get("optb88vdw_bandgap"))
        if band_gap is None:
            band_gap = safe_float(jarvis_entry.get("mbj_bandgap"))
        if band_gap is None:
            band_gap = safe_float(jarvis_entry.get("bandgap"))
        if band_gap is not None:
            properties["band_gap"] = band_gap

        # Bulk modulus
        bulk_modulus = safe_float(jarvis_entry.get("bulk_modulus_kv"))
        if bulk_modulus is not None:
            properties["bulk_modulus"] = bulk_modulus

        # Shear modulus
        shear_modulus = safe_float(jarvis_entry.get("shear_modulus_gv"))
        if shear_modulus is not None:
            properties["shear_modulus"] = shear_modulus

        # Energy above hull
        energy_above_hull = safe_float(jarvis_entry.get("ehull"))
        if energy_above_hull is not None:
            properties["energy_above_hull"] = energy_above_hull

        # Density
        density = safe_float(jarvis_entry.get("density"))
        if density is not None:
            properties["density"] = density

        # Number of atoms
        natoms = safe_int(jarvis_entry.get("natoms"))
        if natoms is not None:
            properties["natoms"] = float(natoms)

        return properties

    def _create_material_data(self, entry: Dict[str, Any], formula: str) -> Optional[MaterialData]:
        """
        Create MaterialData object from JARVIS entry.

        Args:
            entry: JARVIS database entry
            formula: Chemical formula

        Returns:
            MaterialData object or None if validation fails
        """
        # Extract properties
        properties = self.extract_properties(entry)

        # Validate that we have at least formation energy
        if "formation_energy_per_atom" not in properties:
            return None

        # Get material ID
        material_id = entry.get("jid", f"jarvis-{formula}")

        # Get energy above hull (default to 0 if not available)
        energy_above_hull = properties.get("energy_above_hull", 0.0)

        # Create MaterialData object
        material_data = MaterialData(
            material_id=material_id,
            formula=formula,
            crystal_structure=entry.get("atoms", {}),
            formation_energy=properties["formation_energy_per_atom"],
            energy_above_hull=energy_above_hull,
            band_gap=properties.get("band_gap"),
            density=properties.get("density"),
            properties=properties,
            metadata={
                "source": "JARVIS-DFT",
                "jid": entry.get("jid"),
                "spacegroup": entry.get("spacegroup"),
            }
        )

        return material_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded JARVIS data.

        Returns:
            Dictionary with statistics about the database
        """
        return {
            "total_materials": len(self.data),
            "file_path": str(self.jarvis_file_path),
        }
