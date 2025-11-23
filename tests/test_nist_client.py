"""Tests for NIST-JANAF thermochemical data client."""

import pytest
import numpy as np
from pathlib import Path

from ceramic_discovery.dft import NISTClient


class TestNISTClient:
    """Test NIST-JANAF client functionality."""

    def test_client_initialization(self):
        """Test that client initializes correctly."""
        client = NISTClient()
        assert client.loaded_data == {}
        assert isinstance(client.loaded_data, dict)

    def test_load_thermochemical_data_file_not_found(self):
        """Test that client raises error for non-existent file."""
        client = NISTClient()
        with pytest.raises(FileNotFoundError):
            client.load_thermochemical_data("nonexistent_file.txt", "test")

    def test_load_thermochemical_data_basic_format(self, temp_data_dir):
        """Test loading basic NIST format data."""
        # Create sample NIST file with basic format
        nist_file = temp_data_dir / "sic_test.txt"
        nist_content = """# NIST-JANAF Thermochemical Data for SiC
# Temperature(K)  Cp(J/mol-K)  S(J/mol-K)  H-H298(kJ/mol)  DfG(kJ/mol)
298.15           26.86        16.55        0.000             -62.76
500.00           31.23        24.89        5.234             -65.43
1000.00          38.45        38.12        18.567            -71.89
1500.00          41.23        47.56        34.123            -76.45
2000.00          42.89        54.78        50.234            -80.12
"""
        with open(nist_file, 'w', encoding='utf-8') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "SiC")

        # Verify data loaded
        assert not data.empty
        assert len(data) == 5
        assert 'temperature' in data.columns
        assert 'cp' in data.columns
        assert 'entropy' in data.columns

        # Verify specific values
        assert data.iloc[0]['temperature'] == 298.15
        assert data.iloc[0]['cp'] == 26.86
        assert data.iloc[0]['entropy'] == 16.55

    def test_load_thermochemical_data_with_infinite_values(self, temp_data_dir):
        """Test handling of INFINITE values in NIST data."""
        nist_file = temp_data_dir / "test_infinite.txt"
        nist_content = """# Test data with INFINITE values
298.15  26.86  16.55  0.000  -62.76
500.00  31.23  INFINITE  5.234  -65.43
1000.00  38.45  38.12  INFINITE  -71.89
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Verify data loaded
        assert not data.empty
        assert len(data) == 3

        # Verify INFINITE values are handled (converted to NaN/None)
        # Row with INFINITE in entropy should have NaN for that column
        row_500 = data[data['temperature'] == 500.00].iloc[0]
        assert pd.isna(row_500['entropy']) or row_500.get('entropy') is None

    def test_load_thermochemical_data_minimal_format(self, temp_data_dir):
        """Test loading minimal NIST format (just temperature and one property)."""
        nist_file = temp_data_dir / "minimal.txt"
        nist_content = """298.15  26.86
500.00  31.23
1000.00  38.45
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "minimal")

        assert not data.empty
        assert len(data) == 3
        assert 'temperature' in data.columns
        assert 'cp' in data.columns

    def test_load_thermochemical_data_with_comments(self, temp_data_dir):
        """Test that comments are properly skipped."""
        nist_file = temp_data_dir / "with_comments.txt"
        nist_content = """# This is a comment
# Another comment line
298.15  26.86  16.55
# Comment in the middle
500.00  31.23  24.89
1000.00  38.45  38.12
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        assert len(data) == 3
        assert data.iloc[0]['temperature'] == 298.15

    def test_load_thermochemical_data_invalid_format(self, temp_data_dir):
        """Test error handling for invalid file format."""
        nist_file = temp_data_dir / "invalid.txt"
        nist_content = """This is not valid NIST data
No numeric values here
Just text
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        with pytest.raises(ValueError, match="No valid data rows"):
            client.load_thermochemical_data(str(nist_file), "invalid")

    def test_interpolate_property_linear(self, temp_data_dir):
        """Test linear interpolation of property values."""
        nist_file = temp_data_dir / "interp_test.txt"
        nist_content = """298.15  26.86  16.55
500.00  31.23  24.89
1000.00  38.45  38.12
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Test interpolation at midpoint
        cp_750 = client.interpolate_property(data, 750.0, "cp")
        
        # Should be between 31.23 and 38.45
        assert cp_750 is not None
        assert 31.23 < cp_750 < 38.45
        
        # Linear interpolation: 750 is halfway between 500 and 1000
        # Expected: 31.23 + 0.5 * (38.45 - 31.23) = 34.84
        assert abs(cp_750 - 34.84) < 0.01

    def test_interpolate_property_exact_match(self, temp_data_dir):
        """Test interpolation returns exact value when temperature matches."""
        nist_file = temp_data_dir / "exact_test.txt"
        nist_content = """298.15  26.86  16.55
500.00  31.23  24.89
1000.00  38.45  38.12
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Test exact temperature match
        cp_500 = client.interpolate_property(data, 500.0, "cp")
        assert cp_500 == 31.23

    def test_interpolate_property_out_of_range(self, temp_data_dir):
        """Test interpolation returns None for out-of-range temperatures."""
        nist_file = temp_data_dir / "range_test.txt"
        nist_content = """298.15  26.86  16.55
500.00  31.23  24.89
1000.00  38.45  38.12
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Test below range
        cp_low = client.interpolate_property(data, 200.0, "cp")
        assert cp_low is None

        # Test above range
        cp_high = client.interpolate_property(data, 2000.0, "cp")
        assert cp_high is None

    def test_interpolate_property_missing_property(self, temp_data_dir):
        """Test interpolation returns None for non-existent property."""
        nist_file = temp_data_dir / "missing_prop.txt"
        nist_content = """298.15  26.86
500.00  31.23
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Try to interpolate non-existent property
        value = client.interpolate_property(data, 400.0, "nonexistent")
        assert value is None

    def test_get_property_at_temperature(self, temp_data_dir):
        """Test convenience method for getting property at temperature."""
        nist_file = temp_data_dir / "convenience_test.txt"
        nist_content = """298.15  26.86  16.55
500.00  31.23  24.89
1000.00  38.45  38.12
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        client.load_thermochemical_data(str(nist_file), "SiC")

        # Test convenience method
        cp_750 = client.get_property_at_temperature("SiC", 750.0, "cp")
        assert cp_750 is not None
        assert 31.23 < cp_750 < 38.45

    def test_get_property_at_temperature_not_loaded(self):
        """Test that method returns None for unloaded material."""
        client = NISTClient()
        value = client.get_property_at_temperature("NotLoaded", 500.0, "cp")
        assert value is None

    def test_get_loaded_materials(self, temp_data_dir):
        """Test getting list of loaded materials."""
        nist_file1 = temp_data_dir / "sic.txt"
        nist_file2 = temp_data_dir / "tic.txt"
        
        nist_content = """298.15  26.86
500.00  31.23
"""
        with open(nist_file1, 'w') as f:
            f.write(nist_content)
        with open(nist_file2, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        assert client.get_loaded_materials() == []

        client.load_thermochemical_data(str(nist_file1), "SiC")
        assert "SiC" in client.get_loaded_materials()

        client.load_thermochemical_data(str(nist_file2), "TiC")
        loaded = client.get_loaded_materials()
        assert "SiC" in loaded
        assert "TiC" in loaded
        assert len(loaded) == 2

    def test_get_available_properties(self, temp_data_dir):
        """Test getting list of available properties for a material."""
        nist_file = temp_data_dir / "props_test.txt"
        nist_content = """298.15  26.86  16.55  0.000
500.00  31.23  24.89  5.234
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        client.load_thermochemical_data(str(nist_file), "test")

        props = client.get_available_properties("test")
        assert 'temperature' not in props  # Should exclude temperature
        assert 'cp' in props
        assert 'entropy' in props
        assert 'enthalpy' in props

    def test_get_available_properties_not_loaded(self):
        """Test that method returns empty list for unloaded material."""
        client = NISTClient()
        props = client.get_available_properties("NotLoaded")
        assert props == []

    def test_get_temperature_range(self, temp_data_dir):
        """Test getting temperature range for a material."""
        nist_file = temp_data_dir / "range_test.txt"
        nist_content = """298.15  26.86
500.00  31.23
1000.00  38.45
1500.00  41.23
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        client.load_thermochemical_data(str(nist_file), "test")

        temp_range = client.get_temperature_range("test")
        assert temp_range is not None
        assert temp_range[0] == 298.15
        assert temp_range[1] == 1500.00

    def test_get_temperature_range_not_loaded(self):
        """Test that method returns None for unloaded material."""
        client = NISTClient()
        temp_range = client.get_temperature_range("NotLoaded")
        assert temp_range is None

    def test_parse_value_handles_special_cases(self):
        """Test that _parse_value handles special cases correctly."""
        client = NISTClient()

        # Test INFINITE
        assert client._parse_value("INFINITE") is None
        assert client._parse_value("infinite") is None
        assert client._parse_value("INF") is None

        # Test missing data markers
        assert client._parse_value("---") is None
        assert client._parse_value("...") is None
        assert client._parse_value("N/A") is None
        assert client._parse_value("NA") is None
        assert client._parse_value("") is None

        # Test valid numbers
        assert client._parse_value("123.45") == 123.45
        assert client._parse_value("  -67.89  ") == -67.89

    def test_data_sorted_by_temperature(self, temp_data_dir):
        """Test that loaded data is sorted by temperature."""
        nist_file = temp_data_dir / "unsorted.txt"
        # Intentionally unsorted data
        nist_content = """1000.00  38.45
298.15  26.86
500.00  31.23
"""
        with open(nist_file, 'w') as f:
            f.write(nist_content)

        client = NISTClient()
        data = client.load_thermochemical_data(str(nist_file), "test")

        # Verify data is sorted
        temps = data['temperature'].values
        assert np.all(temps[:-1] <= temps[1:])  # Check ascending order
        assert temps[0] == 298.15
        assert temps[-1] == 1000.00

    def test_multiple_materials_cached(self, temp_data_dir):
        """Test that multiple materials can be loaded and cached."""
        nist_file1 = temp_data_dir / "mat1.txt"
        nist_file2 = temp_data_dir / "mat2.txt"
        
        content1 = """298.15  26.86
500.00  31.23
"""
        content2 = """298.15  30.00
500.00  35.00
"""
        
        with open(nist_file1, 'w') as f:
            f.write(content1)
        with open(nist_file2, 'w') as f:
            f.write(content2)

        client = NISTClient()
        data1 = client.load_thermochemical_data(str(nist_file1), "Material1")
        data2 = client.load_thermochemical_data(str(nist_file2), "Material2")

        # Verify both are cached
        assert "Material1" in client.loaded_data
        assert "Material2" in client.loaded_data

        # Verify they have different data
        assert data1.iloc[0]['cp'] == 26.86
        assert data2.iloc[0]['cp'] == 30.00


# Import pandas for the test that uses pd.isna
import pandas as pd
