"""Validate test data integrity and consistency."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class TestDataValidator:
    """Validator for test data files."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_material_data(self, data: Dict[str, Any]) -> bool:
        """Validate material data structure and values."""
        required_fields = [
            "material_id",
            "formula",
            "energy_above_hull",
            "formation_energy_per_atom",
        ]

        # Check required fields
        for field in required_fields:
            if field not in data:
                self.errors.append(f"Missing required field: {field}")
                return False

        # Validate energy above hull
        e_hull = data["energy_above_hull"]
        if e_hull < 0:
            self.errors.append(
                f"Material {data['material_id']}: "
                f"energy_above_hull cannot be negative ({e_hull})"
            )

        # Validate properties if present
        if "properties" in data:
            self._validate_properties(data["material_id"], data["properties"])

        return len(self.errors) == 0

    def _validate_properties(self, material_id: str, properties: Dict[str, float]):
        """Validate material properties."""
        # Physical bounds for properties
        bounds = {
            "hardness": (0, 50),  # GPa
            "fracture_toughness": (0, 20),  # MPa·m^0.5
            "density": (0.5, 25),  # g/cm³
            "youngs_modulus": (0, 1000),  # GPa
            "thermal_conductivity_25C": (0, 500),  # W/m·K
            "thermal_conductivity_1000C": (0, 500),  # W/m·K
            "melting_point": (0, 5000),  # K
            "band_gap": (0, 15),  # eV
        }

        for prop, (min_val, max_val) in bounds.items():
            if prop in properties:
                value = properties[prop]
                if not (min_val <= value <= max_val):
                    self.errors.append(
                        f"Material {material_id}: {prop} = {value} "
                        f"outside physical bounds [{min_val}, {max_val}]"
                    )

        # Check thermal conductivity temperature dependence
        if (
            "thermal_conductivity_25C" in properties
            and "thermal_conductivity_1000C" in properties
        ):
            tc_25 = properties["thermal_conductivity_25C"]
            tc_1000 = properties["thermal_conductivity_1000C"]

            # For most ceramics, thermal conductivity decreases with temperature
            if tc_1000 > tc_25:
                self.warnings.append(
                    f"Material {material_id}: thermal conductivity increases "
                    f"with temperature ({tc_25} -> {tc_1000}), unusual for ceramics"
                )

    def validate_json_file(self, filepath: Path) -> bool:
        """Validate a JSON test data file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {filepath}: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"File not found: {filepath}")
            return False

        # Validate based on file type
        if "materials" in data:
            for material in data["materials"]:
                self.validate_material_data(material)

        return len(self.errors) == 0

    def validate_all_test_data(self, test_data_dir: Path) -> bool:
        """Validate all test data files."""
        print(f"Validating test data in: {test_data_dir}")

        if not test_data_dir.exists():
            print(f"Test data directory not found: {test_data_dir}")
            return True  # Not an error if directory doesn't exist yet

        # Find all JSON files
        json_files = list(test_data_dir.glob("**/*.json"))

        if not json_files:
            print("No JSON files found in test data directory")
            return True

        print(f"Found {len(json_files)} JSON files to validate")

        for json_file in json_files:
            print(f"\nValidating: {json_file.relative_to(test_data_dir)}")
            self.validate_json_file(json_file)

        # Print results
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All test data validated successfully!")

        return len(self.errors) == 0

    def get_summary(self) -> Dict[str, int]:
        """Get validation summary."""
        return {
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


def main():
    """Main validation function."""
    # Get test data directory
    script_dir = Path(__file__).parent
    test_data_dir = script_dir.parent / "data"

    # Validate
    validator = TestDataValidator()
    success = validator.validate_all_test_data(test_data_dir)

    # Print summary
    summary = validator.get_summary()
    print(f"\n{'='*60}")
    print(f"Validation Summary:")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"{'='*60}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
