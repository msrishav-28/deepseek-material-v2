"""Tests for JARVIS-DFT client."""

import json
import pytest
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

from ceramic_discovery.dft import JarvisClient, MaterialData


class TestJarvisClient:
    """Test JARVIS-DFT client functionality."""

    def test_client_initialization_requires_file_path(self):
        """Test that client requires valid file path."""
        with pytest.raises(ValueError, match="file path cannot be empty"):
            JarvisClient("")

    def test_client_initialization_file_not_found(self):
        """Test that client raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            JarvisClient("nonexistent_file.json")

    def test_client_initialization_with_valid_file(self, temp_data_dir):
        """Test client initializes with valid JARVIS file."""
        # Create sample JARVIS file
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [
            {
                "jid": "JVASP-1-SiC",
                "formula": "SiC",
                "formation_energy_peratom": -0.65,
                "optb88vdw_bandgap": 2.3,
            }
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        assert client.jarvis_file_path == jarvis_file
        assert len(client.data) == 1

    def test_client_handles_invalid_json(self, temp_data_dir):
        """Test client handles invalid JSON gracefully."""
        jarvis_file = temp_data_dir / "invalid.json"
        with open(jarvis_file, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            JarvisClient(str(jarvis_file))

    def test_parse_formula_from_jid_with_formula(self):
        """Test parsing formula from jid with formula component."""
        # Create minimal client
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump([], temp_file)
        temp_file.close()

        try:
            client = JarvisClient(temp_file.name)

            assert client.parse_formula_from_jid("JVASP-12345-SiC") == "SiC"
            assert client.parse_formula_from_jid("JVASP-SiC-12345") == "SiC"
            assert client.parse_formula_from_jid("JVASP-1234-TiC") == "TiC"
            assert client.parse_formula_from_jid("JVASP-B4C-5678") == "B4C"
        finally:
            Path(temp_file.name).unlink()

    def test_parse_formula_from_jid_without_formula(self):
        """Test parsing formula from jid without formula component."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump([], temp_file)
        temp_file.close()

        try:
            client = JarvisClient(temp_file.name)

            assert client.parse_formula_from_jid("JVASP-12345") is None
            assert client.parse_formula_from_jid("") is None
            assert client.parse_formula_from_jid(None) is None
        finally:
            Path(temp_file.name).unlink()

    def test_extract_properties_complete_entry(self):
        """Test property extraction from complete JARVIS entry."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump([], temp_file)
        temp_file.close()

        try:
            client = JarvisClient(temp_file.name)

            entry = {
                "jid": "JVASP-1-SiC",
                "formation_energy_peratom": -0.65,
                "optb88vdw_bandgap": 2.3,
                "bulk_modulus_kv": 220.0,
                "shear_modulus_gv": 190.0,
                "ehull": 0.0,
                "density": 3.21,
                "natoms": 2,
            }

            properties = client.extract_properties(entry)

            assert properties["formation_energy_per_atom"] == -0.65
            assert properties["band_gap"] == 2.3
            assert properties["bulk_modulus"] == 220.0
            assert properties["shear_modulus"] == 190.0
            assert properties["energy_above_hull"] == 0.0
            assert properties["density"] == 3.21
            assert properties["natoms"] == 2.0
        finally:
            Path(temp_file.name).unlink()

    def test_extract_properties_partial_entry(self):
        """Test property extraction from partial JARVIS entry."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump([], temp_file)
        temp_file.close()

        try:
            client = JarvisClient(temp_file.name)

            entry = {
                "jid": "JVASP-1-SiC",
                "formation_energy_peratom": -0.65,
                # Missing other properties
            }

            properties = client.extract_properties(entry)

            assert "formation_energy_per_atom" in properties
            assert properties["formation_energy_per_atom"] == -0.65
            assert "band_gap" not in properties
            assert "bulk_modulus" not in properties
        finally:
            Path(temp_file.name).unlink()

    def test_extract_properties_band_gap_fallback(self):
        """Test band gap extraction with fallback options."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump([], temp_file)
        temp_file.close()

        try:
            client = JarvisClient(temp_file.name)

            # Test optb88vdw_bandgap priority
            entry1 = {
                "optb88vdw_bandgap": 2.3,
                "mbj_bandgap": 2.5,
                "bandgap": 2.1,
            }
            assert client.extract_properties(entry1)["band_gap"] == 2.3

            # Test mbj_bandgap fallback
            entry2 = {
                "mbj_bandgap": 2.5,
                "bandgap": 2.1,
            }
            assert client.extract_properties(entry2)["band_gap"] == 2.5

            # Test bandgap fallback
            entry3 = {
                "bandgap": 2.1,
            }
            assert client.extract_properties(entry3)["band_gap"] == 2.1
        finally:
            Path(temp_file.name).unlink()

    def test_load_carbides_filters_carbon_materials(self, temp_data_dir):
        """Test that load_carbides filters for carbon-containing materials."""
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [
            {
                "jid": "JVASP-1-SiC",
                "formula": "SiC",
                "formation_energy_peratom": -0.65,
            },
            {
                "jid": "JVASP-2-Si",
                "formula": "Si",
                "formation_energy_peratom": -0.50,
            },
            {
                "jid": "JVASP-3-TiC",
                "formula": "TiC",
                "formation_energy_peratom": -0.70,
            },
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        carbides = client.load_carbides()

        assert len(carbides) == 2
        formulas = [m.formula for m in carbides]
        assert "SiC" in formulas
        assert "TiC" in formulas
        assert "Si" not in formulas

    def test_load_carbides_filters_by_metal_elements(self, temp_data_dir):
        """Test that load_carbides filters by specified metal elements."""
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [
            {
                "jid": "JVASP-1-SiC",
                "formula": "SiC",
                "formation_energy_peratom": -0.65,
            },
            {
                "jid": "JVASP-2-TiC",
                "formula": "TiC",
                "formation_energy_peratom": -0.70,
            },
            {
                "jid": "JVASP-3-ZrC",
                "formula": "ZrC",
                "formation_energy_peratom": -0.75,
            },
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        carbides = client.load_carbides(metal_elements={"Si", "Ti"})

        assert len(carbides) == 2
        formulas = [m.formula for m in carbides]
        assert "SiC" in formulas
        assert "TiC" in formulas
        assert "ZrC" not in formulas

    def test_load_carbides_returns_material_data_objects(self, temp_data_dir):
        """Test that load_carbides returns MaterialData objects."""
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [
            {
                "jid": "JVASP-1-SiC",
                "formula": "SiC",
                "formation_energy_peratom": -0.65,
                "optb88vdw_bandgap": 2.3,
                "density": 3.21,
            }
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        carbides = client.load_carbides()

        assert len(carbides) == 1
        material = carbides[0]
        assert isinstance(material, MaterialData)
        assert material.formula == "SiC"
        assert material.formation_energy == -0.65
        assert material.band_gap == 2.3
        assert material.density == 3.21
        assert material.metadata["source"] == "JARVIS-DFT"

    def test_load_carbides_skips_invalid_entries(self, temp_data_dir):
        """Test that load_carbides skips entries without required data."""
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [
            {
                "jid": "JVASP-1-SiC",
                "formula": "SiC",
                "formation_energy_peratom": -0.65,
            },
            {
                "jid": "JVASP-2-TiC",
                "formula": "TiC",
                # Missing formation_energy_peratom
            },
            {
                "jid": "JVASP-3-ZrC",
                "formula": "ZrC",
                "formation_energy_peratom": -0.75,
            },
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        carbides = client.load_carbides()

        # Should skip TiC due to missing formation energy
        assert len(carbides) == 2
        formulas = [m.formula for m in carbides]
        assert "SiC" in formulas
        assert "ZrC" in formulas
        assert "TiC" not in formulas

    def test_get_statistics(self, temp_data_dir):
        """Test statistics retrieval."""
        jarvis_file = temp_data_dir / "jarvis_test.json"
        sample_data = [{"jid": f"JVASP-{i}"} for i in range(10)]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f)

        client = JarvisClient(str(jarvis_file))
        stats = client.get_statistics()

        assert stats["total_materials"] == 10
        assert "file_path" in stats


class TestJarvisClientPropertyBased:
    """Property-based tests for JARVIS client."""

    @given(
        metal_elements=st.sets(
            st.sampled_from(["Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr"]),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_jarvis_carbide_extraction_completeness(self, metal_elements):
        """
        **Feature: sic-alloy-integration, Property 1: JARVIS carbide extraction completeness**
        
        For any JARVIS-DFT dataset, when filtering for carbides with specified metal elements,
        the system should identify all materials where the formula contains both carbon and
        at least one specified metal element.
        
        **Validates: Requirements 1.1**
        """
        # Generate test dataset with known carbides and non-carbides
        test_data = []
        expected_carbides = []
        
        # Add carbides with specified metals
        for i, metal in enumerate(metal_elements):
            formula = f"{metal}C"
            entry = {
                "jid": f"JVASP-{i}-{formula}",
                "formula": formula,
                "formation_energy_peratom": -0.65 - i * 0.01,
            }
            test_data.append(entry)
            expected_carbides.append(formula)
        
        # Add carbides with other metals (should not be included)
        other_metals = ["Fe", "Co", "Ni", "Cu"]
        for i, metal in enumerate(other_metals):
            formula = f"{metal}C"
            entry = {
                "jid": f"JVASP-{100+i}-{formula}",
                "formula": formula,
                "formation_energy_peratom": -0.50 - i * 0.01,
            }
            test_data.append(entry)
        
        # Add non-carbides (should not be included)
        for i, metal in enumerate(metal_elements):
            formula = f"{metal}O"
            entry = {
                "jid": f"JVASP-{200+i}-{formula}",
                "formula": formula,
                "formation_energy_peratom": -0.40 - i * 0.01,
            }
            test_data.append(entry)
        
        # Create JARVIS file in temporary location
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_data, temp_file)
        temp_file.close()
        jarvis_file = temp_file.name
        
        try:
            # Load carbides with metal filter
            client = JarvisClient(jarvis_file)
            carbides = client.load_carbides(metal_elements=metal_elements)
            
            # Extract formulas from results
            result_formulas = {m.formula for m in carbides}
            
            # Verify completeness: all expected carbides should be found
            for expected_formula in expected_carbides:
                assert expected_formula in result_formulas, \
                    f"Expected carbide {expected_formula} not found in results"
            
            # Verify correctness: no non-carbides or wrong-metal carbides
            for material in carbides:
                # Must contain carbon
                assert 'C' in material.formula, \
                    f"Non-carbide {material.formula} found in results"
                
                # Must contain at least one specified metal
                has_metal = any(metal in material.formula for metal in metal_elements)
                assert has_metal, \
                    f"Carbide {material.formula} does not contain any specified metals"
            
            # Verify count matches expected
            assert len(carbides) == len(expected_carbides), \
                f"Expected {len(expected_carbides)} carbides, found {len(carbides)}"
        finally:
            # Clean up temporary file
            Path(jarvis_file).unlink(missing_ok=True)
