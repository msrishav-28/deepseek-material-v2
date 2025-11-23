"""
Data combiner for merging materials data from multiple sources.

This module provides functionality to combine data from Materials Project,
JARVIS-DFT, NIST-JANAF, and literature databases with deduplication and
conflict resolution.
"""

import logging
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from ceramic_discovery.dft.materials_project_client import MaterialData

logger = logging.getLogger(__name__)


class DataCombiner:
    """Combine materials data from multiple sources with conflict resolution."""

    # Source priority for conflict resolution (higher = higher priority)
    SOURCE_PRIORITY = {
        "Literature": 4,  # Experimental data has highest priority
        "Materials Project": 3,
        "JARVIS-DFT": 2,
        "NIST-JANAF": 1,
    }

    def __init__(self):
        """Initialize data combiner."""
        self.combined_data: List[MaterialData] = []
        self.source_statistics: Dict[str, int] = {}

    def combine_sources(
        self,
        mp_data: Optional[List[MaterialData]] = None,
        jarvis_data: Optional[List[MaterialData]] = None,
        nist_data: Optional[Dict[str, pd.DataFrame]] = None,
        literature_data: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Combine all data sources into unified dataset.

        Merges materials from multiple sources with:
        1. Deduplication by chemical formula
        2. Conflict resolution using source priority
        3. Metadata tracking for data provenance

        Args:
            mp_data: List of MaterialData from Materials Project
            jarvis_data: List of MaterialData from JARVIS-DFT
            nist_data: Dictionary mapping formula to NIST DataFrames
            literature_data: Dictionary mapping formula to property dictionaries

        Returns:
            DataFrame with combined material data

        Examples:
            >>> combiner = DataCombiner()
            >>> df = combiner.combine_sources(mp_data=mp_materials, jarvis_data=jarvis_materials)
            >>> len(df)
            500
        """
        # Collect all materials
        all_materials: List[MaterialData] = []

        if mp_data:
            all_materials.extend(mp_data)
            self.source_statistics["Materials Project"] = len(mp_data)
            logger.info(f"Added {len(mp_data)} materials from Materials Project")

        if jarvis_data:
            all_materials.extend(jarvis_data)
            self.source_statistics["JARVIS-DFT"] = len(jarvis_data)
            logger.info(f"Added {len(jarvis_data)} materials from JARVIS-DFT")

        # Deduplicate materials by formula
        deduplicated = self.deduplicate_materials(all_materials)
        logger.info(
            f"Deduplicated {len(all_materials)} materials to {len(deduplicated)} unique formulas"
        )

        # Enhance with NIST data if available
        if nist_data:
            deduplicated = self._enhance_with_nist(deduplicated, nist_data)
            self.source_statistics["NIST-JANAF"] = len(nist_data)
            logger.info(f"Enhanced with NIST data for {len(nist_data)} materials")

        # Enhance with literature data if available
        if literature_data:
            deduplicated = self._enhance_with_literature(deduplicated, literature_data)
            self.source_statistics["Literature"] = len(literature_data)
            logger.info(f"Enhanced with literature data for {len(literature_data)} materials")

        self.combined_data = deduplicated

        # Convert to DataFrame
        return self._to_dataframe(deduplicated)

    def deduplicate_materials(
        self, materials: List[MaterialData]
    ) -> List[MaterialData]:
        """
        Remove duplicate materials based on chemical formula.

        When duplicates are found, resolves conflicts using source priority
        and merges properties from all sources.

        Args:
            materials: List of MaterialData objects potentially with duplicates

        Returns:
            List of unique MaterialData objects with merged properties

        Examples:
            >>> materials = [mat1, mat2, mat3]  # mat1 and mat2 have same formula
            >>> unique = combiner.deduplicate_materials(materials)
            >>> len(unique)
            2
        """
        # Group materials by formula
        formula_groups: Dict[str, List[MaterialData]] = {}

        for material in materials:
            formula = self._normalize_formula(material.formula)
            if formula not in formula_groups:
                formula_groups[formula] = []
            formula_groups[formula].append(material)

        # Resolve conflicts for each formula
        deduplicated = []
        for formula, group in formula_groups.items():
            if len(group) == 1:
                # No conflict, use as-is
                deduplicated.append(group[0])
            else:
                # Resolve conflicts
                merged = self._merge_materials(formula, group)
                deduplicated.append(merged)

        return deduplicated

    def resolve_conflicts(
        self, material_id: str, properties: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve conflicting property values from different sources.

        Uses source priority to select values:
        1. Literature (experimental) - highest priority
        2. Materials Project
        3. JARVIS-DFT
        4. NIST-JANAF - lowest priority

        Args:
            material_id: Material identifier for logging
            properties: List of property dictionaries with 'source' key

        Returns:
            Dictionary with resolved property values

        Examples:
            >>> props = [
            ...     {"source": "JARVIS-DFT", "hardness": 25.0},
            ...     {"source": "Literature", "hardness": 28.0}
            ... ]
            >>> resolved = combiner.resolve_conflicts("SiC", props)
            >>> resolved["hardness"]
            28.0
        """
        if not properties:
            return {}

        # Collect all property keys
        all_keys: Set[str] = set()
        for prop_dict in properties:
            all_keys.update(prop_dict.keys())

        # Remove metadata keys
        all_keys.discard("source")
        all_keys.discard("metadata")

        # Resolve each property
        resolved = {}
        source_tracking = {}

        for key in all_keys:
            # Collect values from all sources
            candidates = []
            for prop_dict in properties:
                if key in prop_dict and prop_dict[key] is not None:
                    source = prop_dict.get("source", "Unknown")
                    priority = self.SOURCE_PRIORITY.get(source, 0)
                    candidates.append((priority, source, prop_dict[key]))

            if candidates:
                # Sort by priority (highest first)
                candidates.sort(reverse=True, key=lambda x: x[0])
                # Use highest priority value
                _, source, value = candidates[0]
                resolved[key] = value
                source_tracking[key] = source

        # Add source tracking metadata
        resolved["_source_tracking"] = source_tracking

        return resolved

    def _normalize_formula(self, formula: str) -> str:
        """
        Normalize chemical formula for comparison.

        Removes whitespace and standardizes format.

        Args:
            formula: Chemical formula

        Returns:
            Normalized formula string
        """
        return formula.strip().replace(" ", "")

    def _merge_materials(
        self, formula: str, materials: List[MaterialData]
    ) -> MaterialData:
        """
        Merge multiple MaterialData objects for same formula.

        Args:
            formula: Chemical formula
            materials: List of MaterialData with same formula

        Returns:
            Merged MaterialData object
        """
        # Collect properties from all materials
        all_properties = []
        for material in materials:
            props = material.properties.copy()
            props["source"] = material.metadata.get("source", "Unknown")
            all_properties.append(props)

        # Resolve conflicts
        merged_properties = self.resolve_conflicts(formula, all_properties)

        # Extract source tracking
        source_tracking = merged_properties.pop("_source_tracking", {})

        # Use highest priority material as base
        materials_by_priority = sorted(
            materials,
            key=lambda m: self.SOURCE_PRIORITY.get(
                m.metadata.get("source", "Unknown"), 0
            ),
            reverse=True,
        )
        base_material = materials_by_priority[0]

        # Collect all sources
        sources = [m.metadata.get("source", "Unknown") for m in materials]

        # Create merged material
        merged = MaterialData(
            material_id=base_material.material_id,
            formula=formula,
            crystal_structure=base_material.crystal_structure,
            formation_energy=merged_properties.get(
                "formation_energy_per_atom", base_material.formation_energy
            ),
            energy_above_hull=merged_properties.get(
                "energy_above_hull", base_material.energy_above_hull
            ),
            band_gap=merged_properties.get("band_gap", base_material.band_gap),
            density=merged_properties.get("density", base_material.density),
            properties=merged_properties,
            metadata={
                "sources": sources,
                "source_tracking": source_tracking,
                "primary_source": base_material.metadata.get("source", "Unknown"),
                "merged_from": len(materials),
            },
        )

        return merged

    def _enhance_with_nist(
        self, materials: List[MaterialData], nist_data: Dict[str, pd.DataFrame]
    ) -> List[MaterialData]:
        """
        Enhance materials with NIST thermochemical data.

        Args:
            materials: List of MaterialData objects
            nist_data: Dictionary mapping formula to NIST DataFrames

        Returns:
            Enhanced list of MaterialData objects
        """
        enhanced = []

        for material in materials:
            formula = self._normalize_formula(material.formula)

            if formula in nist_data:
                # Add NIST data to properties
                nist_df = nist_data[formula]

                # Store temperature-dependent data
                material.properties["nist_thermochemical_data"] = nist_df.to_dict()

                # Add room temperature values if available
                if 298.15 in nist_df.index:
                    room_temp_data = nist_df.loc[298.15]
                    for col in room_temp_data.index:
                        prop_key = f"nist_{col.lower().replace(' ', '_')}"
                        material.properties[prop_key] = room_temp_data[col]

                # Update metadata
                if "sources" in material.metadata:
                    if "NIST-JANAF" not in material.metadata["sources"]:
                        material.metadata["sources"].append("NIST-JANAF")
                else:
                    material.metadata["sources"] = [
                        material.metadata.get("source", "Unknown"),
                        "NIST-JANAF",
                    ]

            enhanced.append(material)

        return enhanced

    def _enhance_with_literature(
        self, materials: List[MaterialData], literature_data: Dict[str, Dict]
    ) -> List[MaterialData]:
        """
        Enhance materials with literature data.

        Args:
            materials: List of MaterialData objects
            literature_data: Dictionary mapping formula to property dictionaries

        Returns:
            Enhanced list of MaterialData objects
        """
        enhanced = []

        for material in materials:
            formula = self._normalize_formula(material.formula)

            if formula in literature_data:
                lit_data = literature_data[formula]

                # Add literature properties with higher priority
                for key, value in lit_data.items():
                    if key not in ["sources", "references"]:
                        # Use literature value (highest priority)
                        lit_key = f"literature_{key}"
                        material.properties[lit_key] = value

                        # Update source tracking
                        if "source_tracking" not in material.metadata:
                            material.metadata["source_tracking"] = {}
                        material.metadata["source_tracking"][key] = "Literature"

                # Update metadata
                if "sources" in material.metadata:
                    if "Literature" not in material.metadata["sources"]:
                        material.metadata["sources"].append("Literature")
                else:
                    material.metadata["sources"] = [
                        material.metadata.get("source", "Unknown"),
                        "Literature",
                    ]

                # Add literature references
                if "references" in lit_data:
                    material.metadata["literature_references"] = lit_data["references"]

            enhanced.append(material)

        return enhanced

    def _to_dataframe(self, materials: List[MaterialData]) -> pd.DataFrame:
        """
        Convert list of MaterialData to DataFrame.

        Args:
            materials: List of MaterialData objects

        Returns:
            DataFrame with material properties
        """
        records = []

        for material in materials:
            record = {
                "material_id": material.material_id,
                "formula": material.formula,
                "formation_energy": material.formation_energy,
                "energy_above_hull": material.energy_above_hull,
                "band_gap": material.band_gap,
                "density": material.density,
            }

            # Add all properties
            for key, value in material.properties.items():
                # Skip nested structures
                if not isinstance(value, (dict, list, pd.DataFrame)):
                    record[key] = value

            # Add metadata
            record["sources"] = ",".join(material.metadata.get("sources", []))
            record["primary_source"] = material.metadata.get("primary_source", "")

            records.append(record)

        df = pd.DataFrame(records)
        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about combined data.

        Returns:
            Dictionary with statistics about data sources and combination
        """
        stats = {
            "total_materials": len(self.combined_data),
            "source_counts": self.source_statistics.copy(),
        }

        if self.combined_data:
            # Count materials by number of sources
            source_counts = {}
            for material in self.combined_data:
                num_sources = len(material.metadata.get("sources", []))
                source_counts[num_sources] = source_counts.get(num_sources, 0) + 1

            stats["materials_by_source_count"] = source_counts

        return stats
