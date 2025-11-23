"""Tests for DFT stability analysis."""

import pytest
import numpy as np

from ceramic_discovery.dft import (
    StabilityAnalyzer,
    StabilityClassification,
    StabilityResult,
)


class TestStabilityAnalyzer:
    """Test stability analyzer with correct DFT criteria."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes with correct threshold."""
        analyzer = StabilityAnalyzer()
        assert analyzer.metastable_threshold == 0.1

    def test_analyzer_custom_threshold(self):
        """Test analyzer accepts custom threshold."""
        analyzer = StabilityAnalyzer(metastable_threshold=0.15)
        assert analyzer.metastable_threshold == 0.15

    def test_analyzer_rejects_invalid_threshold(self):
        """Test analyzer rejects non-positive threshold."""
        with pytest.raises(ValueError, match="must be positive"):
            StabilityAnalyzer(metastable_threshold=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            StabilityAnalyzer(metastable_threshold=-0.1)

    def test_classify_stable_material(self):
        """Test classification of stable material (ΔE_hull = 0)."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-149",
            formula="SiC",
            energy_above_hull=0.0,
            formation_energy_per_atom=-0.65,
        )

        assert isinstance(result, StabilityResult)
        assert result.classification == StabilityClassification.STABLE
        assert result.is_viable is True
        assert result.confidence_score >= 0.9
        assert "stable" in result.notes.lower()

    def test_classify_metastable_material_low(self):
        """Test classification of metastable material (ΔE_hull = 0.02)."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-test",
            formula="SiC-B",
            energy_above_hull=0.02,
            formation_energy_per_atom=-0.60,
        )

        assert result.classification == StabilityClassification.METASTABLE
        assert result.is_viable is True
        assert result.confidence_score > 0.7
        assert "metastable" in result.notes.lower()

    def test_classify_metastable_material_threshold(self):
        """Test classification at metastable threshold (ΔE_hull = 0.1)."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-test",
            formula="Test",
            energy_above_hull=0.1,
            formation_energy_per_atom=-0.50,
        )

        assert result.classification == StabilityClassification.METASTABLE
        assert result.is_viable is True
        assert "metastable" in result.notes.lower()

    def test_classify_unstable_material(self):
        """Test classification of unstable material (ΔE_hull > 0.1)."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-test",
            formula="Unstable",
            energy_above_hull=0.15,
            formation_energy_per_atom=-0.30,
        )

        assert result.classification == StabilityClassification.UNSTABLE
        assert result.is_viable is False
        assert result.confidence_score < 0.7
        assert "unstable" in result.notes.lower()

    def test_classify_highly_unstable_material(self):
        """Test classification of highly unstable material (ΔE_hull >> 0.1)."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-test",
            formula="VeryUnstable",
            energy_above_hull=0.5,
            formation_energy_per_atom=0.1,
        )

        assert result.classification == StabilityClassification.UNSTABLE
        assert result.is_viable is False
        assert result.confidence_score < 0.5

    def test_correct_stability_criteria_implementation(self):
        """CRITICAL: Test that correct stability criteria are implemented."""
        analyzer = StabilityAnalyzer()

        # Test boundary cases around the CORRECT 0.1 eV/atom threshold
        test_cases = [
            (0.0, True, StabilityClassification.STABLE),
            (0.001, True, StabilityClassification.METASTABLE),
            (0.05, True, StabilityClassification.METASTABLE),
            (0.09, True, StabilityClassification.METASTABLE),
            (0.1, True, StabilityClassification.METASTABLE),
            (0.101, False, StabilityClassification.UNSTABLE),
            (0.15, False, StabilityClassification.UNSTABLE),
            (0.2, False, StabilityClassification.UNSTABLE),
        ]

        for energy_hull, expected_viable, expected_class in test_cases:
            result = analyzer.analyze_stability(
                material_id=f"test-{energy_hull}",
                formula="Test",
                energy_above_hull=energy_hull,
                formation_energy_per_atom=-0.5,
            )

            assert result.is_viable == expected_viable, (
                f"Failed for ΔE_hull = {energy_hull}: "
                f"expected viable={expected_viable}, got {result.is_viable}"
            )
            assert result.classification == expected_class, (
                f"Failed for ΔE_hull = {energy_hull}: "
                f"expected {expected_class}, got {result.classification}"
            )

    def test_is_viable_quick_check(self):
        """Test quick viability check method."""
        analyzer = StabilityAnalyzer()

        assert analyzer.is_viable(0.0) is True
        assert analyzer.is_viable(0.05) is True
        assert analyzer.is_viable(0.1) is True
        assert analyzer.is_viable(0.11) is False
        assert analyzer.is_viable(0.5) is False

    def test_confidence_score_stable_material(self):
        """Test confidence score is high for stable materials."""
        analyzer = StabilityAnalyzer()

        result = analyzer.analyze_stability(
            material_id="mp-test",
            formula="Test",
            energy_above_hull=0.0,
            formation_energy_per_atom=-1.5,  # Very negative
        )

        assert result.confidence_score >= 0.95

    def test_confidence_score_decreases_with_energy(self):
        """Test confidence score decreases as energy above hull increases."""
        analyzer = StabilityAnalyzer()

        energies = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.3]
        confidences = []

        for energy in energies:
            result = analyzer.analyze_stability(
                material_id="test",
                formula="Test",
                energy_above_hull=energy,
                formation_energy_per_atom=-0.5,
            )
            confidences.append(result.confidence_score)

        # Confidence should generally decrease
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1] - 0.15  # Allow small variations

    def test_confidence_bonus_for_negative_formation_energy(self):
        """Test confidence bonus for strongly negative formation energy."""
        analyzer = StabilityAnalyzer()

        # Same energy above hull, different formation energies
        result_weak = analyzer.analyze_stability(
            material_id="test1",
            formula="Test",
            energy_above_hull=0.05,
            formation_energy_per_atom=-0.3,
        )

        result_strong = analyzer.analyze_stability(
            material_id="test2",
            formula="Test",
            energy_above_hull=0.05,
            formation_energy_per_atom=-1.5,
        )

        assert result_strong.confidence_score > result_weak.confidence_score

    def test_rejects_negative_energy_above_hull(self):
        """Test that negative energy above hull is rejected."""
        analyzer = StabilityAnalyzer()

        with pytest.raises(ValueError, match="cannot be negative"):
            analyzer.analyze_stability(
                material_id="test",
                formula="Test",
                energy_above_hull=-0.1,
                formation_energy_per_atom=-0.5,
            )

    def test_batch_analyze(self):
        """Test batch analysis of multiple materials."""
        analyzer = StabilityAnalyzer()

        materials = [
            {
                "material_id": "mp-1",
                "formula": "SiC",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -0.65,
            },
            {
                "material_id": "mp-2",
                "formula": "B4C",
                "energy_above_hull": 0.05,
                "formation_energy_per_atom": -0.50,
            },
            {
                "material_id": "mp-3",
                "formula": "Test",
                "energy_above_hull": 0.15,
                "formation_energy_per_atom": -0.30,
            },
        ]

        results = analyzer.batch_analyze(materials)

        assert len(results) == 3
        assert all(isinstance(r, StabilityResult) for r in results)
        assert results[0].classification == StabilityClassification.STABLE
        assert results[1].classification == StabilityClassification.METASTABLE
        assert results[2].classification == StabilityClassification.UNSTABLE

    def test_batch_analyze_skips_invalid_data(self):
        """Test batch analysis skips materials with missing data."""
        analyzer = StabilityAnalyzer()

        materials = [
            {
                "material_id": "mp-1",
                "formula": "SiC",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -0.65,
            },
            {
                "material_id": "mp-2",
                "formula": "Invalid",
                # Missing energy_above_hull
                "formation_energy_per_atom": -0.50,
            },
            {
                "material_id": "mp-3",
                "formula": "Test",
                "energy_above_hull": 0.05,
                "formation_energy_per_atom": -0.40,
            },
        ]

        results = analyzer.batch_analyze(materials)

        # Should skip the invalid material
        assert len(results) == 2
        assert results[0].material_id == "mp-1"
        assert results[1].material_id == "mp-3"

    def test_filter_viable_materials(self):
        """Test filtering to only viable materials."""
        analyzer = StabilityAnalyzer()

        materials = [
            {"material_id": "mp-1", "energy_above_hull": 0.0},
            {"material_id": "mp-2", "energy_above_hull": 0.05},
            {"material_id": "mp-3", "energy_above_hull": 0.1},
            {"material_id": "mp-4", "energy_above_hull": 0.15},
            {"material_id": "mp-5", "energy_above_hull": 0.3},
        ]

        viable = analyzer.filter_viable_materials(materials)

        assert len(viable) == 3
        assert all(m["energy_above_hull"] <= 0.1 for m in viable)

    def test_get_stability_statistics(self):
        """Test calculation of stability statistics."""
        analyzer = StabilityAnalyzer()

        materials = [
            {
                "material_id": f"mp-{i}",
                "formula": "Test",
                "energy_above_hull": energy,
                "formation_energy_per_atom": -0.5,
            }
            for i, energy in enumerate([0.0, 0.0, 0.05, 0.08, 0.15, 0.2])
        ]

        results = analyzer.batch_analyze(materials)
        stats = analyzer.get_stability_statistics(results)

        assert stats["total"] == 6
        assert stats["stable"] == 2
        assert stats["metastable"] == 2
        assert stats["unstable"] == 2
        assert stats["viable_fraction"] == 4 / 6
        assert 0.0 <= stats["avg_energy_above_hull"] <= 0.2
        assert 0.0 <= stats["avg_confidence"] <= 1.0
        assert stats["min_energy_above_hull"] == 0.0
        assert stats["max_energy_above_hull"] == 0.2

    def test_get_stability_statistics_empty(self):
        """Test statistics with empty results."""
        analyzer = StabilityAnalyzer()

        stats = analyzer.get_stability_statistics([])

        assert stats["total"] == 0
        assert stats["stable"] == 0
        assert stats["metastable"] == 0
        assert stats["unstable"] == 0
        assert stats["viable_fraction"] == 0.0

    def test_notes_generation(self):
        """Test that appropriate notes are generated."""
        analyzer = StabilityAnalyzer()

        # Stable material
        result_stable = analyzer.analyze_stability(
            "mp-1", "SiC", 0.0, -0.65
        )
        assert "stable" in result_stable.notes.lower()
        assert "convex hull" in result_stable.notes.lower()

        # Metastable material
        result_meta = analyzer.analyze_stability(
            "mp-2", "Test", 0.05, -0.50
        )
        assert "metastable" in result_meta.notes.lower()
        assert "0.05" in result_meta.notes or "0.0500" in result_meta.notes

        # Unstable material
        result_unstable = analyzer.analyze_stability(
            "mp-3", "Test", 0.2, -0.30
        )
        assert "unstable" in result_unstable.notes.lower()
        assert "0.2" in result_unstable.notes or "0.20" in result_unstable.notes

    def test_ceramic_dopant_screening_scenario(self):
        """Test realistic ceramic dopant screening scenario."""
        analyzer = StabilityAnalyzer()

        # Simulate screening of SiC with various dopants
        dopant_systems = [
            ("SiC", 0.0, -0.65),  # Pure SiC - stable
            ("Si0.98B0.02C", 0.02, -0.63),  # B-doped - metastable, viable
            ("Si0.95Al0.05C", 0.08, -0.60),  # Al-doped - metastable, viable
            ("Si0.90Ti0.10C", 0.12, -0.55),  # Ti-doped - unstable
            ("Si0.90Fe0.10C", 0.25, -0.40),  # Fe-doped - unstable
        ]

        results = []
        for formula, e_hull, e_form in dopant_systems:
            result = analyzer.analyze_stability(
                material_id=f"test-{formula}",
                formula=formula,
                energy_above_hull=e_hull,
                formation_energy_per_atom=e_form,
            )
            results.append(result)

        # Check that correct materials are identified as viable
        viable_results = [r for r in results if r.is_viable]
        assert len(viable_results) == 3  # Pure SiC, B-doped, Al-doped

        # Check that unstable materials are correctly identified
        unstable_results = [
            r for r in results if r.classification == StabilityClassification.UNSTABLE
        ]
        assert len(unstable_results) == 2  # Ti-doped, Fe-doped

        # Verify statistics
        stats = analyzer.get_stability_statistics(results)
        assert stats["viable_fraction"] == 0.6  # 3 out of 5
