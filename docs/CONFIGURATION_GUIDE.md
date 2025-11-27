# Configuration Guide

This guide explains the configuration options available in the Ceramic Armor Discovery Framework, with a focus on the SiC Alloy Designer Integration features.

## Overview

The framework uses a hierarchical configuration system based on Python dataclasses. Configuration can be set through:

1. Environment variables (`.env` file)
2. Direct modification of the `Config` object
3. Default values

## Configuration Structure

### Data Sources Configuration

Control which data sources are enabled and their settings.

#### JARVIS-DFT Configuration

```python
config.data_sources.jarvis.enabled = True
config.data_sources.jarvis.file_path = Path("./data/jdft_3d-7-7-2018.json")
config.data_sources.jarvis.metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr", "B", "Al"}
```

**Environment Variables:**
- `JARVIS_ENABLED`: Enable/disable JARVIS data loading (default: `true`)
- `JARVIS_FILE_PATH`: Path to JARVIS JSON database file (default: `./data/jdft_3d-7-7-2018.json`)

**Description:**
- `enabled`: Whether to load data from JARVIS-DFT database
- `file_path`: Location of the JARVIS JSON database file
- `metal_elements`: Set of metal elements to filter for carbide extraction

#### NIST-JANAF Configuration

```python
config.data_sources.nist.enabled = True
config.data_sources.nist.data_dir = Path("./data/nist")
config.data_sources.nist.default_temperature_K = 298.15
```

**Environment Variables:**
- `NIST_ENABLED`: Enable/disable NIST data loading (default: `true`)
- `NIST_DATA_DIR`: Directory containing NIST data files (default: `./data/nist`)

**Description:**
- `enabled`: Whether to load data from NIST-JANAF database
- `data_dir`: Directory containing NIST thermochemical data files
- `default_temperature_K`: Default temperature for property interpolation

#### Literature Database Configuration

```python
config.data_sources.literature.enabled = True
```

**Environment Variables:**
- `LITERATURE_ENABLED`: Enable/disable literature data (default: `true`)

**Description:**
- `enabled`: Whether to use embedded literature reference data

#### Data Combination Settings

```python
config.data_sources.source_priority = ["literature", "materials_project", "jarvis", "nist"]
config.data_sources.enable_deduplication = True
config.data_sources.conflict_resolution_strategy = "priority"
```

**Description:**
- `source_priority`: Order of preference when resolving conflicting property values
- `enable_deduplication`: Remove duplicate materials based on formula
- `conflict_resolution_strategy`: How to resolve conflicts (`"priority"`, `"average"`, `"newest"`)

### Feature Engineering Configuration

Control which descriptors are calculated and feature selection options.

#### Composition Descriptors

```python
config.feature_engineering.composition_descriptors.enabled = True
config.feature_engineering.composition_descriptors.calculate_entropy = True
config.feature_engineering.composition_descriptors.calculate_electronegativity = True
config.feature_engineering.composition_descriptors.calculate_atomic_properties = True
```

**Description:**
- `enabled`: Enable composition descriptor calculation
- `calculate_entropy`: Calculate Shannon entropy of composition
- `calculate_electronegativity`: Calculate electronegativity-based features
- `calculate_atomic_properties`: Calculate atomic radius, mass, and number features

#### Structure Descriptors

```python
config.feature_engineering.structure_descriptors.enabled = True
config.feature_engineering.structure_descriptors.calculate_pugh_ratio = True
config.feature_engineering.structure_descriptors.calculate_elastic_properties = True
config.feature_engineering.structure_descriptors.calculate_energy_density = True
config.feature_engineering.structure_descriptors.validate_physical_bounds = True
```

**Description:**
- `enabled`: Enable structure descriptor calculation
- `calculate_pugh_ratio`: Calculate Pugh ratio (B/G) for ductility
- `calculate_elastic_properties`: Calculate Young's modulus and Poisson ratio
- `calculate_energy_density`: Calculate energy per unit volume
- `validate_physical_bounds`: Enforce physical constraints on calculated values

#### Feature Selection

```python
config.feature_engineering.min_feature_variance = 0.01
config.feature_engineering.remove_correlated_features = False
config.feature_engineering.correlation_threshold = 0.95
```

**Description:**
- `min_feature_variance`: Minimum variance threshold for feature selection
- `remove_correlated_features`: Remove highly correlated features
- `correlation_threshold`: Correlation threshold for feature removal (0-1)

### Application Ranking Configuration

Configure application-specific material ranking.

```python
config.application_ranking.enabled = True
config.application_ranking.applications = [
    "aerospace_hypersonic",
    "cutting_tools",
    "thermal_barriers",
    "wear_resistant",
    "electronic"
]
config.application_ranking.allow_partial_scoring = True
config.application_ranking.min_properties_for_ranking = 2
config.application_ranking.confidence_threshold = 0.5
```

**Description:**
- `enabled`: Enable application-specific ranking
- `applications`: List of applications to rank materials for
- `allow_partial_scoring`: Allow scoring with missing properties
- `min_properties_for_ranking`: Minimum number of properties required for ranking
- `confidence_threshold`: Minimum confidence score to include in results

**Available Applications:**
- `aerospace_hypersonic`: Hypersonic aerospace applications (turbine blades, leading edges)
- `cutting_tools`: Cutting tools and machining applications
- `thermal_barriers`: Thermal barrier coatings
- `wear_resistant`: Wear-resistant coatings and surfaces
- `electronic`: Electronic and semiconductor applications

### Experimental Planning Configuration

Configure experimental validation planning.

#### Synthesis Configuration

```python
config.experimental_planning.synthesis.default_methods = [
    "Hot Pressing",
    "Solid State Sintering",
    "Spark Plasma Sintering"
]
config.experimental_planning.synthesis.cost_multiplier = 1.0
config.experimental_planning.synthesis.time_multiplier = 1.0
```

**Description:**
- `default_methods`: List of synthesis methods to consider
- `cost_multiplier`: Multiplier for cost estimates (adjust for local costs)
- `time_multiplier`: Multiplier for time estimates (adjust for facility capabilities)

#### Characterization Configuration

```python
config.experimental_planning.characterization.default_techniques = [
    "XRD",
    "SEM",
    "TEM",
    "Hardness Testing",
    "Thermal Conductivity",
    "Density Measurement"
]
config.experimental_planning.characterization.cost_multiplier = 1.0
config.experimental_planning.characterization.time_multiplier = 1.0
```

**Description:**
- `default_techniques`: List of characterization techniques to include
- `cost_multiplier`: Multiplier for cost estimates
- `time_multiplier`: Multiplier for time estimates

#### Resource Estimation

```python
config.experimental_planning.default_timeline_months = 6.0
config.experimental_planning.default_cost_k_dollars = 50.0
config.experimental_planning.confidence_level = 0.8
```

**Description:**
- `default_timeline_months`: Default timeline estimate in months
- `default_cost_k_dollars`: Default cost estimate in thousands of dollars
- `confidence_level`: Confidence level for estimates (0-1)

## Usage Examples

### Example 1: Disable JARVIS Data Source

```python
from ceramic_discovery.config import config

# Disable JARVIS data loading
config.data_sources.jarvis.enabled = False

# Validate configuration
config.validate()
```

### Example 2: Customize Application Ranking

```python
from ceramic_discovery.config import config

# Only rank for aerospace and cutting tools
config.application_ranking.applications = [
    "aerospace_hypersonic",
    "cutting_tools"
]

# Require at least 3 properties for ranking
config.application_ranking.min_properties_for_ranking = 3
```

### Example 3: Adjust Experimental Planning Costs

```python
from ceramic_discovery.config import config

# Adjust for higher local costs
config.experimental_planning.synthesis.cost_multiplier = 1.5
config.experimental_planning.characterization.cost_multiplier = 1.3

# Adjust for faster facility
config.experimental_planning.synthesis.time_multiplier = 0.8
```

### Example 4: Disable Structure Descriptors

```python
from ceramic_discovery.config import config

# Only use composition descriptors
config.feature_engineering.structure_descriptors.enabled = False
```

### Example 5: Using Environment Variables

Create a `.env` file:

```bash
# Disable JARVIS
JARVIS_ENABLED=false

# Custom NIST directory
NIST_DATA_DIR=/data/nist_custom

# Custom JARVIS path
JARVIS_FILE_PATH=/data/custom_jarvis.json
```

The configuration will automatically load these values on initialization.

## Validation

The configuration system includes validation to ensure:

1. Required API keys are present
2. File paths exist when data sources are enabled
3. Numeric thresholds are within valid ranges
4. At least one application is specified for ranking

Call `config.validate()` to check configuration validity:

```python
from ceramic_discovery.config import config

try:
    config.validate()
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

1. **Use environment variables for sensitive data**: Store API keys and database URLs in `.env` files
2. **Validate after modifications**: Always call `config.validate()` after changing configuration
3. **Document custom configurations**: Keep track of non-default settings for reproducibility
4. **Test with minimal configuration**: Start with default settings and adjust as needed
5. **Use multipliers for local adjustments**: Adjust cost and time multipliers rather than modifying base estimates

## Troubleshooting

### JARVIS file not found warning

If you see a warning about JARVIS file not found, either:
1. Download the JARVIS database to the specified path
2. Update `JARVIS_FILE_PATH` to point to your file
3. Disable JARVIS with `JARVIS_ENABLED=false`

### Validation errors

Common validation errors and solutions:

- **"MATERIALS_PROJECT_API_KEY is required"**: Set the API key in your `.env` file
- **"At least one application must be specified"**: Ensure `applications` list is not empty
- **"Correlation threshold must be between 0 and 1"**: Check `correlation_threshold` value

### Performance tuning

For large datasets:
1. Disable unused data sources
2. Disable unused descriptor calculations
3. Reduce the number of applications for ranking
4. Increase `min_properties_for_ranking` to filter out incomplete materials

## See Also

- [User Guide](USER_GUIDE.md) - General usage instructions
- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [JARVIS Integration Guide](JARVIS_INTEGRATION.md) - JARVIS-specific documentation
