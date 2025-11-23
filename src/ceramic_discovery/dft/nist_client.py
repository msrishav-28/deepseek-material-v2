"""
NIST-JANAF thermochemical data client for loading temperature-dependent properties.

This module provides functionality to load and parse NIST-JANAF thermochemical
tables, which contain temperature-dependent properties like heat capacity,
entropy, enthalpy, and Gibbs free energy.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ceramic_discovery.utils.data_conversion import safe_float

logger = logging.getLogger(__name__)


class NISTClient:
    """Client for loading and parsing NIST-JANAF thermochemical data files."""

    def __init__(self):
        """Initialize NIST client."""
        self.loaded_data: Dict[str, pd.DataFrame] = {}

    def load_thermochemical_data(
        self, filepath: str, material_name: str
    ) -> pd.DataFrame:
        """
        Load temperature-dependent properties from NIST-JANAF file.

        NIST-JANAF files contain tabular data with columns:
        - Temperature (K)
        - Cp (J/mol·K) - Heat capacity
        - S° (J/mol·K) - Standard entropy
        - H°-H°298 (kJ/mol) - Enthalpy relative to 298.15 K
        - ΔfH° (kJ/mol) - Standard enthalpy of formation
        - ΔfG° (kJ/mol) - Standard Gibbs free energy of formation
        - log(Kf) - Logarithm of equilibrium constant

        Args:
            filepath: Path to NIST-JANAF text file
            material_name: Name identifier for the material

        Returns:
            DataFrame with temperature-dependent properties

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid

        Examples:
            >>> client = NISTClient()
            >>> data = client.load_thermochemical_data("SiC.txt", "SiC")
            >>> data.columns
            Index(['temperature', 'cp', 'entropy', 'enthalpy', 'gibbs_energy'])
        """
        filepath_obj = Path(filepath)

        if not filepath_obj.exists():
            raise FileNotFoundError(f"NIST file not found: {filepath}")

        try:
            # Read file content
            with open(filepath_obj, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the data
            df = self._parse_nist_format(content, material_name)

            # Cache the loaded data
            self.loaded_data[material_name] = df

            logger.info(
                f"Loaded {len(df)} temperature points for {material_name} from NIST"
            )

            return df

        except Exception as e:
            raise ValueError(f"Error parsing NIST file {filepath}: {e}")

    def _parse_nist_format(self, content: str, material_name: str) -> pd.DataFrame:
        """
        Parse NIST-JANAF text format.

        Args:
            content: File content as string
            material_name: Material identifier

        Returns:
            DataFrame with parsed data
        """
        lines = content.strip().split('\n')

        # Find data section (skip header lines)
        data_start_idx = 0
        for i, line in enumerate(lines):
            # Look for lines that start with a number (temperature values)
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                data_start_idx = i
                break

        if data_start_idx == 0:
            # Try to find header line with column names
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['temperature', 'temp', 'cp', 'entropy']):
                    data_start_idx = i + 1
                    break

        # Parse data rows
        data_rows = []
        for line in lines[data_start_idx:]:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Split by whitespace
            parts = stripped.split()

            # Need at least temperature and one property
            if len(parts) < 2:
                continue

            # Try to parse as numeric data
            try:
                row_data = self._parse_data_row(parts)
                if row_data:
                    data_rows.append(row_data)
            except Exception as e:
                logger.debug(f"Skipping line: {line.strip()} - {e}")
                continue

        if not data_rows:
            raise ValueError("No valid data rows found in NIST file")

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Ensure temperature column exists
        if 'temperature' not in df.columns:
            raise ValueError("Temperature column not found in parsed data")

        # Sort by temperature
        df = df.sort_values('temperature').reset_index(drop=True)

        return df

    def _parse_data_row(self, parts: List[str]) -> Optional[Dict[str, float]]:
        """
        Parse a single data row from NIST format.

        Args:
            parts: List of string values from split line

        Returns:
            Dictionary with parsed values or None if invalid
        """
        row = {}

        # First value is always temperature
        temp = self._parse_value(parts[0])
        if temp is None:
            return None
        row['temperature'] = temp

        # Map remaining columns based on typical NIST format
        # Common format: T, Cp, S°, H°-H°298, ΔfH°, ΔfG°, log(Kf)
        column_names = [
            'cp',  # Heat capacity
            'entropy',  # Standard entropy
            'enthalpy',  # Enthalpy relative to 298K
            'formation_enthalpy',  # Standard enthalpy of formation
            'gibbs_energy',  # Standard Gibbs free energy
            'log_kf',  # Log equilibrium constant
        ]

        # Parse remaining columns
        for i, col_name in enumerate(column_names):
            if i + 1 < len(parts):
                value = self._parse_value(parts[i + 1])
                if value is not None:
                    row[col_name] = value

        return row if len(row) > 1 else None

    def _parse_value(self, value_str: str) -> Optional[float]:
        """
        Parse a single value, handling INFINITE and other special cases.

        Args:
            value_str: String value to parse

        Returns:
            Float value or None if invalid/infinite
        """
        value_str = value_str.strip().upper()

        # Handle INFINITE values
        if 'INFINITE' in value_str or 'INF' in value_str:
            return None

        # Handle missing data markers
        if value_str in ['---', '...', 'N/A', 'NA', '']:
            return None

        # Try to convert to float
        return safe_float(value_str)

    def interpolate_property(
        self,
        data: pd.DataFrame,
        temperature: float,
        property_name: str
    ) -> Optional[float]:
        """
        Interpolate property value at specific temperature.

        Uses linear interpolation between adjacent temperature points.
        Returns None if temperature is outside data range or property not available.

        Args:
            data: DataFrame with temperature-dependent properties
            temperature: Target temperature in Kelvin
            property_name: Name of property to interpolate

        Returns:
            Interpolated property value or None

        Examples:
            >>> client = NISTClient()
            >>> data = client.load_thermochemical_data("SiC.txt", "SiC")
            >>> cp_500K = client.interpolate_property(data, 500.0, "cp")
            >>> cp_500K
            31.23
        """
        if data.empty:
            return None

        if 'temperature' not in data.columns:
            logger.warning("Temperature column not found in data")
            return None

        if property_name not in data.columns:
            logger.warning(f"Property '{property_name}' not found in data")
            return None

        # Filter out rows with missing property values
        valid_data = data[['temperature', property_name]].dropna()

        if valid_data.empty:
            return None

        temperatures = valid_data['temperature'].values
        property_values = valid_data[property_name].values

        # Check if temperature is in range
        if temperature < temperatures.min() or temperature > temperatures.max():
            logger.debug(
                f"Temperature {temperature} K outside data range "
                f"[{temperatures.min()}, {temperatures.max()}] K"
            )
            return None

        # Perform linear interpolation
        try:
            interpolated = np.interp(temperature, temperatures, property_values)
            return float(interpolated)
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return None

    def get_property_at_temperature(
        self,
        material_name: str,
        temperature: float,
        property_name: str
    ) -> Optional[float]:
        """
        Get property value for a loaded material at specific temperature.

        Convenience method that combines data lookup and interpolation.

        Args:
            material_name: Name of loaded material
            temperature: Target temperature in Kelvin
            property_name: Name of property to retrieve

        Returns:
            Interpolated property value or None

        Examples:
            >>> client = NISTClient()
            >>> client.load_thermochemical_data("SiC.txt", "SiC")
            >>> cp = client.get_property_at_temperature("SiC", 1000.0, "cp")
        """
        if material_name not in self.loaded_data:
            logger.warning(f"Material '{material_name}' not loaded")
            return None

        data = self.loaded_data[material_name]
        return self.interpolate_property(data, temperature, property_name)

    def get_loaded_materials(self) -> List[str]:
        """
        Get list of loaded material names.

        Returns:
            List of material names that have been loaded
        """
        return list(self.loaded_data.keys())

    def get_available_properties(self, material_name: str) -> List[str]:
        """
        Get list of available properties for a loaded material.

        Args:
            material_name: Name of loaded material

        Returns:
            List of property names available in the data
        """
        if material_name not in self.loaded_data:
            return []

        data = self.loaded_data[material_name]
        # Exclude temperature column
        return [col for col in data.columns if col != 'temperature']

    def get_temperature_range(self, material_name: str) -> Optional[tuple]:
        """
        Get temperature range for a loaded material.

        Args:
            material_name: Name of loaded material

        Returns:
            Tuple of (min_temp, max_temp) or None if not loaded
        """
        if material_name not in self.loaded_data:
            return None

        data = self.loaded_data[material_name]
        if 'temperature' not in data.columns or data.empty:
            return None

        return (float(data['temperature'].min()), float(data['temperature'].max()))
