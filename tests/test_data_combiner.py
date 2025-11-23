"""Tests for data combiner."""

import pytest
import pandas as pd
from hypothesis import given, settings, strategies as st

from ceramic_discovery.dft.data_combiner import DataCombiner
from ceramic_discovery.dft.materials_project_client import MaterialData


class TestDataCombiner:
    """Test data combiner functionality."""

    def test_combiner_initialization(self):
        """Test data combiner initializes correctly."""
        combiner = DataCombiner()
        assert combiner is not None
        assert combiner.combined_data == []
        assert combiner.source_statistics == {}

    def test_combine_sources_empty(self):
        """Test combining with no data sources."""
        combiner = DataCombiner()
        df = combiner.combine_sources()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_combine_sources_single_source(self):
        """Test combining with single data source."""
        combiner = DataCombiner()

        # Create test materials
        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5},
                metadata={"source": "Materials Project"},
            ),
            MaterialData(
                material_id="mp-2",
                formula="TiC",
                crystal_structure={},
                formation_energy=-0.7,
                energy_above_hull=0.0,
                band_gap=None,
                density=4.9,
                properties={"formation_energy_per_atom": -0.7},
                metadata={"source": "Materials Project"},
            ),
        ]

        df = combiner.combine_sources(mp_data=materials)
        assert len(df) == 2
        assert "SiC" in df["formula"].values
        assert "TiC" in df["formula"].values

    def test_deduplicate_materials_no_duplicates(self):
        """Test deduplication with no duplicates."""
        combiner = DataCombiner()

        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={},
                metadata={"source": "Materials Project"},
            ),
            MaterialData(
                material_id="mp-2",
                formula="TiC",
                crystal_structure={},
                formation_energy=-0.7,
                energy_above_hull=0.0,
                band_gap=None,
                density=4.9,
                properties={},
                metadata={"source": "Materials Project"},
            ),
        ]

        deduplicated = combiner.deduplicate_materials(materials)
        assert len(deduplicated) == 2

    def test_deduplicate_materials_with_duplicates(self):
        """Test deduplication with duplicate formulas."""
        combiner = DataCombiner()

        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5},
                metadata={"source": "Materials Project"},
            ),
            MaterialData(
                material_id="jarvis-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.52,
                energy_above_hull=0.0,
                band_gap=2.4,
                density=3.21,
                properties={"formation_energy_per_atom": -0.52},
                metadata={"source": "JARVIS-DFT"},
            ),
        ]

        deduplicated = combiner.deduplicate_materials(materials)
        assert len(deduplicated) == 1
        assert deduplicated[0].formula == "SiC"

    def test_resolve_conflicts_single_source(self):
        """Test conflict resolution with single source."""
        combiner = DataCombiner()

        properties = [{"source": "Materials Project", "hardness": 25.0, "density": 3.2}]

        resolved = combiner.resolve_conflicts("SiC", properties)
        assert resolved["hardness"] == 25.0
        assert resolved["density"] == 3.2

    def test_resolve_conflicts_multiple_sources(self):
        """Test conflict resolution with multiple sources."""
        combiner = DataCombiner()

        properties = [
            {"source": "JARVIS-DFT", "hardness": 25.0, "density": 3.2},
            {"source": "Literature", "hardness": 28.0},
        ]

        resolved = combiner.resolve_conflicts("SiC", properties)
        # Literature has higher priority
        assert resolved["hardness"] == 28.0
        # Density only from JARVIS
        assert resolved["density"] == 3.2

    def test_resolve_conflicts_priority_order(self):
        """Test that conflict resolution respects priority order."""
        combiner = DataCombiner()

        properties = [
            {"source": "NIST-JANAF", "value": 1.0},
            {"source": "JARVIS-DFT", "value": 2.0},
            {"source": "Materials Project", "value": 3.0},
            {"source": "Literature", "value": 4.0},
        ]

        resolved = combiner.resolve_conflicts("test", properties)
        # Literature should win (highest priority)
        assert resolved["value"] == 4.0

    def test_normalize_formula(self):
        """Test formula normalization."""
        combiner = DataCombiner()

        assert combiner._normalize_formula("SiC") == "SiC"
        assert combiner._normalize_formula(" SiC ") == "SiC"
        assert combiner._normalize_formula("Si C") == "SiC"

    def test_get_statistics_empty(self):
        """Test statistics with no data."""
        combiner = DataCombiner()
        stats = combiner.get_statistics()
        assert stats["total_materials"] == 0
        assert stats["source_counts"] == {}

    def test_get_statistics_with_data(self):
        """Test statistics with combined data."""
        combiner = DataCombiner()

        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={},
                metadata={"source": "Materials Project"},
            ),
        ]

        combiner.combine_sources(mp_data=materials)
        stats = combiner.get_statistics()
        assert stats["total_materials"] == 1
        assert stats["source_counts"]["Materials Project"] == 1


class TestDataCombinerPropertyBased:
    """Property-based tests for data combiner."""

    @given(
        st.lists(
            st.tuples(
                st.text(
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                    min_size=1,
                    max_size=10,
                ),
                st.floats(
                    min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False
                ),
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_data_source_deduplication_consistency(self, material_specs):
        """
        **Feature: sic-alloy-integration, Property 2: Data source deduplication consistency**

        For any set of materials from multiple sources, after deduplication by formula,
        no two materials in the result should have identical chemical formulas.

        **Validates: Requirements 1.3**
        """
        combiner = DataCombiner()

        # Create materials from specs
        materials = []
        for i, (formula, formation_energy, energy_above_hull) in enumerate(
            material_specs
        ):
            # Create material with unique ID but potentially duplicate formula
            material = MaterialData(
                material_id=f"test-{i}",
                formula=formula,
                crystal_structure={},
                formation_energy=formation_energy,
                energy_above_hull=energy_above_hull,
                band_gap=None,
                density=None,
                properties={"formation_energy_per_atom": formation_energy},
                metadata={"source": "Test"},
            )
            materials.append(material)

        # Deduplicate
        deduplicated = combiner.deduplicate_materials(materials)

        # Property: No two materials should have identical formulas
        formulas = [combiner._normalize_formula(m.formula) for m in deduplicated]
        unique_formulas = set(formulas)

        assert len(formulas) == len(
            unique_formulas
        ), f"Found duplicate formulas after deduplication: {formulas}"

    @given(
        st.lists(
            st.tuples(
                st.text(
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                    min_size=1,
                    max_size=10,
                ),
                st.sampled_from(
                    ["Materials Project", "JARVIS-DFT", "NIST-JANAF", "Literature"]
                ),
                st.floats(
                    min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False
                ),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_data_combination_preserves_sources(self, material_specs):
        """
        **Feature: sic-alloy-integration, Property 13: Data combination preserves sources**

        For any combined dataset, each material should retain information about which
        data sources contributed to its properties.

        **Validates: Requirements 9.1**
        """
        combiner = DataCombiner()

        # Create materials from specs
        materials = []
        for i, (formula, source, formation_energy) in enumerate(material_specs):
            material = MaterialData(
                material_id=f"{source.lower().replace(' ', '-')}-{i}",
                formula=formula,
                crystal_structure={},
                formation_energy=formation_energy,
                energy_above_hull=0.0,
                band_gap=None,
                density=None,
                properties={"formation_energy_per_atom": formation_energy},
                metadata={"source": source},
            )
            materials.append(material)

        # Deduplicate (which merges sources)
        deduplicated = combiner.deduplicate_materials(materials)

        # Property: Each material should have source information
        for material in deduplicated:
            # Check that metadata contains source information
            assert "sources" in material.metadata or "source" in material.metadata, (
                f"Material {material.formula} missing source information"
            )

            # If merged from multiple sources, should have sources list
            if material.metadata.get("merged_from", 1) > 1:
                assert "sources" in material.metadata, (
                    f"Merged material {material.formula} missing sources list"
                )
                assert len(material.metadata["sources"]) > 0, (
                    f"Material {material.formula} has empty sources list"
                )



class TestDataCombinerIntegration:
    """Integration tests for data combiner with multiple sources."""

    def test_combine_mp_and_jarvis_data(self):
        """Test combining Materials Project and JARVIS data."""
        combiner = DataCombiner()

        # Create MP materials
        mp_materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5, "bulk_modulus": 220.0},
                metadata={"source": "Materials Project"},
            ),
            MaterialData(
                material_id="mp-2",
                formula="TiC",
                crystal_structure={},
                formation_energy=-0.7,
                energy_above_hull=0.0,
                band_gap=None,
                density=4.9,
                properties={"formation_energy_per_atom": -0.7},
                metadata={"source": "Materials Project"},
            ),
        ]

        # Create JARVIS materials (one duplicate, one unique)
        jarvis_materials = [
            MaterialData(
                material_id="jarvis-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.52,
                energy_above_hull=0.0,
                band_gap=2.4,
                density=3.21,
                properties={
                    "formation_energy_per_atom": -0.52,
                    "shear_modulus": 190.0,
                },
                metadata={"source": "JARVIS-DFT"},
            ),
            MaterialData(
                material_id="jarvis-2",
                formula="ZrC",
                crystal_structure={},
                formation_energy=-0.8,
                energy_above_hull=0.0,
                band_gap=None,
                density=6.7,
                properties={"formation_energy_per_atom": -0.8},
                metadata={"source": "JARVIS-DFT"},
            ),
        ]

        # Combine sources
        df = combiner.combine_sources(mp_data=mp_materials, jarvis_data=jarvis_materials)

        # Should have 3 unique materials (SiC merged, TiC from MP, ZrC from JARVIS)
        assert len(df) == 3
        assert set(df["formula"].values) == {"SiC", "TiC", "ZrC"}

        # Check that SiC has data from both sources
        sic_row = df[df["formula"] == "SiC"].iloc[0]
        assert "Materials Project" in sic_row["sources"]
        assert "JARVIS-DFT" in sic_row["sources"]

    def test_combine_with_nist_data(self):
        """Test combining with NIST thermochemical data."""
        combiner = DataCombiner()

        # Create base materials
        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5},
                metadata={"source": "Materials Project"},
            ),
        ]

        # Create NIST data
        nist_data = {
            "SiC": pd.DataFrame(
                {
                    "Cp": [26.86, 31.23],
                    "S": [16.55, 24.89],
                    "H-H298": [0.0, 5.234],
                },
                index=[298.15, 500.0],
            )
        }

        # Combine with NIST data
        df = combiner.combine_sources(mp_data=materials, nist_data=nist_data)

        assert len(df) == 1
        sic_row = df[df["formula"] == "SiC"].iloc[0]
        assert "NIST-JANAF" in sic_row["sources"]

    def test_combine_with_literature_data(self):
        """Test combining with literature data."""
        combiner = DataCombiner()

        # Create base materials
        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5},
                metadata={"source": "Materials Project"},
            ),
        ]

        # Create literature data
        literature_data = {
            "SiC": {
                "hardness_vickers": 2800,
                "hardness_std": 200,
                "thermal_conductivity": 120,
                "references": ["Handbook of Advanced Ceramics"],
            }
        }

        # Combine with literature data
        df = combiner.combine_sources(mp_data=materials, literature_data=literature_data)

        assert len(df) == 1
        sic_row = df[df["formula"] == "SiC"].iloc[0]
        assert "Literature" in sic_row["sources"]

    def test_conflict_resolution_with_multiple_sources(self):
        """Test conflict resolution when same property from multiple sources."""
        combiner = DataCombiner()

        # Create materials with conflicting properties
        mp_material = MaterialData(
            material_id="mp-1",
            formula="SiC",
            crystal_structure={},
            formation_energy=-0.5,
            energy_above_hull=0.0,
            band_gap=2.3,
            density=3.2,
            properties={
                "formation_energy_per_atom": -0.5,
                "bulk_modulus": 220.0,
            },
            metadata={"source": "Materials Project"},
        )

        jarvis_material = MaterialData(
            material_id="jarvis-1",
            formula="SiC",
            crystal_structure={},
            formation_energy=-0.52,
            energy_above_hull=0.0,
            band_gap=2.4,
            density=3.21,
            properties={
                "formation_energy_per_atom": -0.52,
                "bulk_modulus": 225.0,
            },
            metadata={"source": "JARVIS-DFT"},
        )

        # Combine
        df = combiner.combine_sources(
            mp_data=[mp_material], jarvis_data=[jarvis_material]
        )

        # Should have 1 material with merged properties
        assert len(df) == 1

        # Materials Project has higher priority, so should use its values
        sic_row = df[df["formula"] == "SiC"].iloc[0]
        assert sic_row["formation_energy"] == -0.5  # From MP (higher priority)

    def test_deduplication_accuracy(self):
        """Test that deduplication correctly identifies duplicates."""
        combiner = DataCombiner()

        # Create materials with same formula but different IDs
        materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={},
                metadata={"source": "Materials Project"},
            ),
            MaterialData(
                material_id="jarvis-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.52,
                energy_above_hull=0.0,
                band_gap=2.4,
                density=3.21,
                properties={},
                metadata={"source": "JARVIS-DFT"},
            ),
            MaterialData(
                material_id="mp-2",
                formula="TiC",
                crystal_structure={},
                formation_energy=-0.7,
                energy_above_hull=0.0,
                band_gap=None,
                density=4.9,
                properties={},
                metadata={"source": "Materials Project"},
            ),
        ]

        deduplicated = combiner.deduplicate_materials(materials)

        # Should have 2 unique materials
        assert len(deduplicated) == 2

        # Check formulas
        formulas = [m.formula for m in deduplicated]
        assert "SiC" in formulas
        assert "TiC" in formulas

        # SiC should be merged from 2 sources
        sic_material = [m for m in deduplicated if m.formula == "SiC"][0]
        assert sic_material.metadata.get("merged_from", 1) == 2
        assert len(sic_material.metadata.get("sources", [])) == 2

    def test_combine_all_sources(self):
        """Test combining all four data sources together."""
        combiner = DataCombiner()

        # MP data
        mp_materials = [
            MaterialData(
                material_id="mp-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={"formation_energy_per_atom": -0.5},
                metadata={"source": "Materials Project"},
            ),
        ]

        # JARVIS data
        jarvis_materials = [
            MaterialData(
                material_id="jarvis-1",
                formula="SiC",
                crystal_structure={},
                formation_energy=-0.52,
                energy_above_hull=0.0,
                band_gap=2.4,
                density=3.21,
                properties={"formation_energy_per_atom": -0.52},
                metadata={"source": "JARVIS-DFT"},
            ),
        ]

        # NIST data
        nist_data = {
            "SiC": pd.DataFrame(
                {"Cp": [26.86], "S": [16.55]},
                index=[298.15],
            )
        }

        # Literature data
        literature_data = {
            "SiC": {
                "hardness_vickers": 2800,
                "thermal_conductivity": 120,
            }
        }

        # Combine all sources
        df = combiner.combine_sources(
            mp_data=mp_materials,
            jarvis_data=jarvis_materials,
            nist_data=nist_data,
            literature_data=literature_data,
        )

        # Should have 1 material with data from all 4 sources
        assert len(df) == 1
        sic_row = df[df["formula"] == "SiC"].iloc[0]

        # Check all sources are present
        sources = sic_row["sources"]
        assert "Materials Project" in sources
        assert "JARVIS-DFT" in sources
        assert "NIST-JANAF" in sources
        assert "Literature" in sources

        # Check statistics
        stats = combiner.get_statistics()
        assert stats["total_materials"] == 1
        assert stats["source_counts"]["Materials Project"] == 1
        assert stats["source_counts"]["JARVIS-DFT"] == 1
        assert stats["source_counts"]["NIST-JANAF"] == 1
        assert stats["source_counts"]["Literature"] == 1
