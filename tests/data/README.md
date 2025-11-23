# Test Data Management

This directory contains test data for the Ceramic Armor Discovery Framework test suite.

## Directory Structure

```
tests/data/
├── materials/          # Sample material data
├── dft_results/        # Mock DFT calculation results
├── ml_models/          # Pre-trained test models
├── validation/         # Validation reference data
└── fixtures/           # Test fixtures and mock data
```

## Data Sources

### Materials Data
- `materials/baseline_ceramics.json`: Baseline properties for SiC, B₄C, WC, TiC, Al₂O₃
- `materials/doped_systems.json`: Sample doped ceramic systems
- `materials/unstable_materials.json`: Materials with high energy above hull

### DFT Results
- `dft_results/sample_calculations.json`: Mock DFT calculation outputs
- `dft_results/stability_data.json`: Energy above hull and formation energies

### Validation Data
- `validation/literature_references.json`: Literature values for cross-validation
- `validation/experimental_data.json`: Experimental ballistic performance data

## Data Format

All JSON files follow standardized schemas:

### Material Data Schema
```json
{
  "material_id": "string",
  "formula": "string",
  "energy_above_hull": "float (eV/atom)",
  "formation_energy_per_atom": "float (eV/atom)",
  "properties": {
    "hardness": "float (GPa)",
    "fracture_toughness": "float (MPa·m^0.5)",
    "density": "float (g/cm³)",
    ...
  }
}
```

## Usage in Tests

Test data can be loaded using the provided fixtures:

```python
import pytest
from tests.utils import load_test_data

def test_example():
    materials = load_test_data("materials/baseline_ceramics.json")
    assert len(materials) > 0
```

## Data Integrity

All test data is validated during CI/CD pipeline execution to ensure:
- JSON files are valid
- Required fields are present
- Values are within physical bounds
- Units are consistent

## Updating Test Data

When updating test data:
1. Ensure data follows the schema
2. Update this README if adding new data types
3. Run validation: `python tests/utils/validate_test_data.py`
4. Commit changes with descriptive message

## Data Provenance

Test data is derived from:
- Materials Project database (mock data based on real structures)
- Published literature (cited in validation data)
- Synthetic data generated for specific test scenarios
