"""DFT stability analysis with correct thermodynamic criteria."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class StabilityClassification(Enum):
    """Thermodynamic stability classification."""

    STABLE = "stable"  # ΔE_hull = 0
    METASTABLE = "metastable"  # 0 < ΔE_hull ≤ 0.1 eV/atom
    UNSTABLE = "unstable"  # ΔE_hull > 0.1 eV/atom


@dataclass
class StabilityResult:
    """Result of stability analysis."""

    material_id: str
    formula: str
    energy_above_hull: float  # eV/atom
    formation_energy_per_atom: float  # eV/atom
    classification: StabilityClassification
    is_viable: bool  # True if stable or metastable
    confidence_score: float  # 0-1 scale
    notes: str = ""


class StabilityAnalyzer:
    """
    Analyzer for thermodynamic stability using DFT data.

    Implements CORRECT stability criteria:
    - Stable: ΔE_hull = 0 eV/atom
    - Metastable (viable): 0 < ΔE_hull ≤ 0.1 eV/atom
    - Unstable (not viable): ΔE_hull > 0.1 eV/atom
    """

    # CRITICAL: Correct stability threshold
    METASTABLE_THRESHOLD = 0.1  # eV/atom

    # Tolerance for numerical precision
    STABLE_TOLERANCE = 1e-6  # eV/atom

    def __init__(self, metastable_threshold: float = 0.1):
        """
        Initialize stability analyzer.

        Args:
            metastable_threshold: Maximum energy above hull for metastable compounds (eV/atom)
        """
        if metastable_threshold <= 0:
            raise ValueError("Metastable threshold must be positive")

        self.metastable_threshold = metastable_threshold

    def analyze_stability(
        self,
        material_id: str,
        formula: str,
        energy_above_hull: float,
        formation_energy_per_atom: float,
    ) -> StabilityResult:
        """
        Analyze thermodynamic stability of a material.

        Args:
            material_id: Material identifier
            formula: Chemical formula
            energy_above_hull: Energy above convex hull (eV/atom)
            formation_energy_per_atom: Formation energy per atom (eV/atom)

        Returns:
            StabilityResult with classification and viability
        """
        # Validate inputs
        if energy_above_hull < 0:
            raise ValueError(
                f"Energy above hull cannot be negative: {energy_above_hull} eV/atom"
            )

        # Classify stability
        classification = self._classify_stability(energy_above_hull)

        # Determine viability (stable or metastable)
        is_viable = classification in [
            StabilityClassification.STABLE,
            StabilityClassification.METASTABLE,
        ]

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            energy_above_hull, formation_energy_per_atom
        )

        # Generate notes
        notes = self._generate_notes(classification, energy_above_hull)

        return StabilityResult(
            material_id=material_id,
            formula=formula,
            energy_above_hull=energy_above_hull,
            formation_energy_per_atom=formation_energy_per_atom,
            classification=classification,
            is_viable=is_viable,
            confidence_score=confidence_score,
            notes=notes,
        )

    def _classify_stability(self, energy_above_hull: float) -> StabilityClassification:
        """
        Classify stability based on energy above hull.

        Args:
            energy_above_hull: Energy above convex hull (eV/atom)

        Returns:
            StabilityClassification enum value
        """
        if energy_above_hull <= self.STABLE_TOLERANCE:
            return StabilityClassification.STABLE
        elif energy_above_hull <= self.metastable_threshold:
            return StabilityClassification.METASTABLE
        else:
            return StabilityClassification.UNSTABLE

    def _calculate_confidence(
        self, energy_above_hull: float, formation_energy_per_atom: float
    ) -> float:
        """
        Calculate confidence score for stability prediction.

        Confidence is higher for:
        - Materials closer to the convex hull
        - Materials with more negative formation energies

        Args:
            energy_above_hull: Energy above convex hull (eV/atom)
            formation_energy_per_atom: Formation energy per atom (eV/atom)

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from energy above hull
        if energy_above_hull <= self.STABLE_TOLERANCE:
            hull_confidence = 1.0
        elif energy_above_hull <= self.metastable_threshold:
            # Linear decrease from 1.0 to 0.7 as we approach threshold
            hull_confidence = 1.0 - 0.3 * (energy_above_hull / self.metastable_threshold)
        else:
            # Exponential decrease for unstable materials
            hull_confidence = 0.7 * np.exp(-2 * (energy_above_hull - self.metastable_threshold))

        # Bonus confidence for strongly negative formation energy
        if formation_energy_per_atom < -1.0:
            formation_bonus = 0.1
        elif formation_energy_per_atom < -0.5:
            formation_bonus = 0.05
        else:
            formation_bonus = 0.0

        confidence = min(1.0, hull_confidence + formation_bonus)

        return confidence

    def _generate_notes(
        self, classification: StabilityClassification, energy_above_hull: float
    ) -> str:
        """
        Generate explanatory notes for stability classification.

        Args:
            classification: Stability classification
            energy_above_hull: Energy above convex hull (eV/atom)

        Returns:
            Explanatory notes string
        """
        if classification == StabilityClassification.STABLE:
            return "Material is thermodynamically stable (on convex hull)"
        elif classification == StabilityClassification.METASTABLE:
            return (
                f"Material is metastable (ΔE_hull = {energy_above_hull:.4f} eV/atom ≤ 0.1). "
                "May be synthesizable under appropriate conditions."
            )
        else:
            return (
                f"Material is thermodynamically unstable (ΔE_hull = {energy_above_hull:.4f} eV/atom > 0.1). "
                "Synthesis is unlikely without kinetic stabilization."
            )

    def is_viable(self, energy_above_hull: float) -> bool:
        """
        Quick check if material is viable (stable or metastable).

        Args:
            energy_above_hull: Energy above convex hull (eV/atom)

        Returns:
            True if material is viable for synthesis
        """
        return energy_above_hull <= self.metastable_threshold

    def batch_analyze(
        self, materials: List[Dict[str, any]]
    ) -> List[StabilityResult]:
        """
        Analyze stability for multiple materials.

        Args:
            materials: List of material dictionaries with required fields

        Returns:
            List of StabilityResult objects
        """
        results = []

        for material in materials:
            try:
                result = self.analyze_stability(
                    material_id=material.get("material_id", "unknown"),
                    formula=material.get("formula", "unknown"),
                    energy_above_hull=material["energy_above_hull"],
                    formation_energy_per_atom=material["formation_energy_per_atom"],
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                # Skip materials with missing or invalid data
                print(f"Warning: Skipping material due to error: {e}")
                continue

        return results

    def filter_viable_materials(
        self, materials: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Filter materials to only viable candidates (stable or metastable).

        Args:
            materials: List of material dictionaries

        Returns:
            List of viable materials
        """
        viable = []

        for material in materials:
            try:
                energy_above_hull = material["energy_above_hull"]
                if self.is_viable(energy_above_hull):
                    viable.append(material)
            except KeyError:
                continue

        return viable

    def get_stability_statistics(
        self, results: List[StabilityResult]
    ) -> Dict[str, any]:
        """
        Calculate statistics for a set of stability results.

        Args:
            results: List of StabilityResult objects

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                "total": 0,
                "stable": 0,
                "metastable": 0,
                "unstable": 0,
                "viable_fraction": 0.0,
                "avg_energy_above_hull": 0.0,
                "avg_confidence": 0.0,
            }

        classifications = [r.classification for r in results]
        energies = [r.energy_above_hull for r in results]
        confidences = [r.confidence_score for r in results]

        stable_count = sum(1 for c in classifications if c == StabilityClassification.STABLE)
        metastable_count = sum(
            1 for c in classifications if c == StabilityClassification.METASTABLE
        )
        unstable_count = sum(
            1 for c in classifications if c == StabilityClassification.UNSTABLE
        )

        viable_count = stable_count + metastable_count

        return {
            "total": len(results),
            "stable": stable_count,
            "metastable": metastable_count,
            "unstable": unstable_count,
            "viable_fraction": viable_count / len(results),
            "avg_energy_above_hull": np.mean(energies),
            "avg_confidence": np.mean(confidences),
            "min_energy_above_hull": np.min(energies),
            "max_energy_above_hull": np.max(energies),
        }


class ConvexHullCalculator:
    """
    Calculator for energy above convex hull using pymatgen.

    Note: This is a placeholder for integration with pymatgen's phase diagram tools.
    Full implementation would use pymatgen.analysis.phase_diagram.PhaseDiagram
    """

    def __init__(self):
        """Initialize convex hull calculator."""
        self.phase_diagrams = {}

    def calculate_energy_above_hull(
        self,
        composition: str,
        formation_energy: float,
        reference_energies: Dict[str, float],
    ) -> float:
        """
        Calculate energy above convex hull for a composition.

        This is a simplified placeholder. Full implementation would use:
        - pymatgen.analysis.phase_diagram.PhaseDiagram
        - pymatgen.core.composition.Composition

        Args:
            composition: Chemical composition string
            formation_energy: Formation energy of the material (eV/atom)
            reference_energies: Dictionary of reference energies for elements

        Returns:
            Energy above convex hull (eV/atom)
        """
        # Placeholder implementation
        # In production, this would:
        # 1. Parse composition using pymatgen.Composition
        # 2. Build phase diagram with all competing phases
        # 3. Calculate distance to convex hull
        # 4. Return energy above hull

        raise NotImplementedError(
            "Full convex hull calculation requires pymatgen integration. "
            "Use Materials Project API data for energy_above_hull values."
        )

    def build_phase_diagram(
        self, system: str, materials: List[Dict[str, any]]
    ) -> None:
        """
        Build phase diagram for a chemical system.

        Args:
            system: Chemical system (e.g., 'Si-C', 'B-C')
            materials: List of materials in the system
        """
        # Placeholder for pymatgen PhaseDiagram integration
        raise NotImplementedError(
            "Phase diagram construction requires pymatgen integration"
        )
