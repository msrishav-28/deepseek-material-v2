"""Tests for literature database."""

import pytest

from ceramic_discovery.dft.literature_database import LiteratureDatabase


class TestLiteratureDatabase:
    """Test literature database functionality."""

    def test_database_initialization(self):
        """Test that database initializes with embedded data."""
        db = LiteratureDatabase()
        assert db is not None
        assert len(db._data) > 0

    def test_list_available_materials(self):
        """Test listing available materials."""
        db = LiteratureDatabase()
        materials = db.list_available_materials()
        
        # Should have all required materials
        assert "SiC" in materials
        assert "TiC" in materials
        assert "ZrC" in materials
        assert "HfC" in materials
        assert "Ta" in materials
        assert "Zr" in materials
        assert "Hf" in materials
        
        # Should have exactly 7 materials
        assert len(materials) == 7
        
        # Should be sorted
        assert materials == sorted(materials)

    def test_get_material_data_known_material(self):
        """Test data retrieval for known materials."""
        db = LiteratureDatabase()
        
        # Test SiC
        sic_data = db.get_material_data("SiC")
        assert sic_data is not None
        assert "hardness_vickers" in sic_data
        assert "thermal_conductivity" in sic_data
        assert "melting_point" in sic_data
        assert "density" in sic_data
        assert "sources" in sic_data
        
        # Verify SiC has expected values
        assert sic_data["hardness_vickers"] == 2800.0
        assert sic_data["thermal_conductivity"] == 120.0
        assert sic_data["melting_point"] == 3103.0
        assert sic_data["density"] == 3.21
        
        # Test TiC
        tic_data = db.get_material_data("TiC")
        assert tic_data is not None
        assert tic_data["hardness_vickers"] == 3200.0
        assert tic_data["melting_point"] == 3433.0

    def test_get_material_data_unknown_material(self):
        """Test missing material handling."""
        db = LiteratureDatabase()
        
        # Unknown material should return None
        data = db.get_material_data("UnknownMaterial")
        assert data is None
        
        # Empty string should return None
        data = db.get_material_data("")
        assert data is None

    def test_get_material_data_returns_copy(self):
        """Test that get_material_data returns a copy, not reference."""
        db = LiteratureDatabase()
        
        data1 = db.get_material_data("SiC")
        data2 = db.get_material_data("SiC")
        
        # Should be equal but not the same object
        assert data1 == data2
        assert data1 is not data2
        
        # Modifying one should not affect the other
        data1["hardness_vickers"] = 9999.0
        data2_check = db.get_material_data("SiC")
        assert data2_check["hardness_vickers"] == 2800.0

    def test_get_property_with_uncertainty_valid_property(self):
        """Test uncertainty handling for valid properties."""
        db = LiteratureDatabase()
        
        # Test SiC hardness
        value, uncertainty = db.get_property_with_uncertainty("SiC", "hardness_vickers")
        assert value == 2800.0
        assert uncertainty == 200.0
        
        # Test SiC thermal conductivity
        value, uncertainty = db.get_property_with_uncertainty("SiC", "thermal_conductivity")
        assert value == 120.0
        assert uncertainty == 10.0
        
        # Test TiC melting point
        value, uncertainty = db.get_property_with_uncertainty("TiC", "melting_point")
        assert value == 3433.0
        assert uncertainty == 50.0

    def test_get_property_with_uncertainty_missing_property(self):
        """Test handling of missing properties."""
        db = LiteratureDatabase()
        
        # Unknown property should return (None, None)
        value, uncertainty = db.get_property_with_uncertainty("SiC", "unknown_property")
        assert value is None
        assert uncertainty is None

    def test_get_property_with_uncertainty_missing_material(self):
        """Test handling of missing materials."""
        db = LiteratureDatabase()
        
        # Unknown material should return (None, None)
        value, uncertainty = db.get_property_with_uncertainty("UnknownMaterial", "hardness_vickers")
        assert value is None
        assert uncertainty is None

    def test_get_available_properties(self):
        """Test getting list of available properties for a material."""
        db = LiteratureDatabase()
        
        props = db.get_available_properties("SiC")
        
        # Should include main properties
        assert "hardness_vickers" in props
        assert "thermal_conductivity" in props
        assert "melting_point" in props
        assert "density" in props
        assert "youngs_modulus" in props
        assert "bulk_modulus" in props
        assert "shear_modulus" in props
        
        # Should NOT include uncertainty keys
        assert "hardness_vickers_std" not in props
        assert "thermal_conductivity_std" not in props
        
        # Should NOT include sources
        assert "sources" not in props
        
        # Should be sorted
        assert props == sorted(props)

    def test_get_available_properties_unknown_material(self):
        """Test getting properties for unknown material."""
        db = LiteratureDatabase()
        
        props = db.get_available_properties("UnknownMaterial")
        assert props == []

    def test_get_sources(self):
        """Test getting literature sources for a material."""
        db = LiteratureDatabase()
        
        sources = db.get_sources("SiC")
        assert len(sources) > 0
        assert isinstance(sources, list)
        assert all(isinstance(s, str) for s in sources)
        
        # Check that sources contain expected references
        sources_str = " ".join(sources)
        assert "Handbook" in sources_str or "Journal" in sources_str

    def test_get_sources_unknown_material(self):
        """Test getting sources for unknown material."""
        db = LiteratureDatabase()
        
        sources = db.get_sources("UnknownMaterial")
        assert sources == []

    def test_has_material(self):
        """Test checking if material exists in database."""
        db = LiteratureDatabase()
        
        # Known materials should return True
        assert db.has_material("SiC") is True
        assert db.has_material("TiC") is True
        assert db.has_material("ZrC") is True
        assert db.has_material("HfC") is True
        assert db.has_material("Ta") is True
        assert db.has_material("Zr") is True
        assert db.has_material("Hf") is True
        
        # Unknown material should return False
        assert db.has_material("UnknownMaterial") is False
        assert db.has_material("") is False

    def test_has_material_handles_whitespace(self):
        """Test that has_material handles whitespace correctly."""
        db = LiteratureDatabase()
        
        # Should handle leading/trailing whitespace
        assert db.has_material(" SiC ") is True
        assert db.has_material("  TiC  ") is True

    def test_get_statistics(self):
        """Test getting database statistics."""
        db = LiteratureDatabase()
        
        stats = db.get_statistics()
        
        assert "total_materials" in stats
        assert "total_properties" in stats
        assert "materials" in stats
        
        assert stats["total_materials"] == 7
        assert stats["total_properties"] > 0
        assert len(stats["materials"]) == 7

    def test_all_materials_have_required_properties(self):
        """Test that all materials have core properties."""
        db = LiteratureDatabase()
        
        required_properties = [
            "hardness_vickers",
            "thermal_conductivity",
            "melting_point",
            "density"
        ]
        
        for material in db.list_available_materials():
            data = db.get_material_data(material)
            for prop in required_properties:
                assert prop in data, f"{material} missing {prop}"
                assert data[prop] is not None
                assert isinstance(data[prop], (int, float))

    def test_all_properties_have_uncertainties(self):
        """Test that all numeric properties have uncertainty values."""
        db = LiteratureDatabase()
        
        for material in db.list_available_materials():
            data = db.get_material_data(material)
            
            # Get all numeric properties (excluding sources)
            numeric_props = [
                key for key in data.keys()
                if not key.endswith('_std') and key != 'sources'
            ]
            
            for prop in numeric_props:
                uncertainty_key = f"{prop}_std"
                assert uncertainty_key in data, f"{material} missing uncertainty for {prop}"
                assert data[uncertainty_key] is not None
                assert isinstance(data[uncertainty_key], (int, float))
                assert data[uncertainty_key] >= 0, f"Uncertainty should be non-negative"

    def test_all_materials_have_sources(self):
        """Test that all materials have literature sources."""
        db = LiteratureDatabase()
        
        for material in db.list_available_materials():
            sources = db.get_sources(material)
            assert len(sources) > 0, f"{material} has no sources"
            assert all(isinstance(s, str) for s in sources)
            assert all(len(s) > 0 for s in sources)

    def test_property_values_are_physically_reasonable(self):
        """Test that property values are within physically reasonable ranges."""
        db = LiteratureDatabase()
        
        for material in db.list_available_materials():
            data = db.get_material_data(material)
            
            # Hardness should be positive and reasonable (0-10000 HV)
            if "hardness_vickers" in data:
                assert 0 < data["hardness_vickers"] < 10000
            
            # Thermal conductivity should be positive (0-500 W/m·K)
            if "thermal_conductivity" in data:
                assert 0 < data["thermal_conductivity"] < 500
            
            # Melting point should be positive and reasonable (0-5000 K)
            if "melting_point" in data:
                assert 0 < data["melting_point"] < 5000
            
            # Density should be positive and reasonable (0-25 g/cm³)
            if "density" in data:
                assert 0 < data["density"] < 25
            
            # Moduli should be positive
            if "youngs_modulus" in data:
                assert data["youngs_modulus"] > 0
            if "bulk_modulus" in data:
                assert data["bulk_modulus"] > 0
            if "shear_modulus" in data:
                assert data["shear_modulus"] > 0

    def test_carbide_materials_have_expected_properties(self):
        """Test that carbide materials (SiC, TiC, ZrC, HfC) have high hardness."""
        db = LiteratureDatabase()
        
        carbides = ["SiC", "TiC", "ZrC", "HfC"]
        
        for carbide in carbides:
            data = db.get_material_data(carbide)
            # Carbides should have high hardness (>2000 HV)
            assert data["hardness_vickers"] > 2000
            # Carbides should have high melting points (>3000 K)
            assert data["melting_point"] > 3000

    def test_metal_materials_have_expected_properties(self):
        """Test that metal materials (Ta, Zr, Hf) have lower hardness than carbides."""
        db = LiteratureDatabase()
        
        metals = ["Ta", "Zr", "Hf"]
        
        for metal in metals:
            data = db.get_material_data(metal)
            # Metals should have lower hardness than carbides (<2000 HV)
            assert data["hardness_vickers"] < 2000
            # Metals should have lower melting points than carbides (<3500 K)
            assert data["melting_point"] < 3500
