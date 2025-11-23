# Design Document: SiC Alloy Designer Integration

## Overview

This design document specifies the integration of comprehensive SiC Alloy Designer functionality into the existing Ceramic Armor Discovery Framework. The integration enhances the framework with multi-source data loading (JARVIS-DFT, NIST-JANAF, literature databases), advanced feature engineering capabilities, application-specific material ranking, and experimental validation planning.

The design follows the existing framework architecture, extending current modules rather than creating parallel systems. All new functionality integrates seamlessly with existing workflows while maintaining backward compatibility.

## Architecture

### Integration Strategy

The SiC Alloy Designer functionality will be distributed across existing modules:

```
src/ceramic_discovery/
├── dft/
│   ├── jarvis_client.py          # NEW: JARVIS-DFT data loader
│   ├── nist_client.py             # NEW: NIST-JANAF data loader
│   ├── literature_database.py     # NEW: Literature data management
│   └── data_combiner.py           # NEW: Multi-source data merging
│
├── ml/
│   ├── composition_descriptors.py # NEW: Composition-based features
│   ├── structure_descriptors.py   # NEW: Structure-derived features
│   └── feature_engineering.py     # ENHANCED: Add new descriptors
│
├── screening/
│   ├── application_ranker.py      # NEW: Application-specific scoring
│   └── screening_engine.py        # ENHANCED: Add ranking capabilities
│
├── validation/
│   └── experimental_planner.py    # NEW: Experimental design
│
├── reporting/
│   └── report_generator.py        # ENHANCED: Add summary reports
│
└── utils/
    └── data_conversion.py         # NEW: Safe conversion utilities
```

### Technology Stack

- **Existing Stack**: Python 3.11+, pymatgen, scikit-learn, pandas, numpy
- **New Dependencies**: None (uses existing dependencies)
- **Data Formats**: JSON (JARVIS), text (NIST), Python dicts (literature)

## Components and Interfaces

### 1. JARVIS-DFT Client (`dft/jarvis_client.py`)


**Purpose**: Load and parse JARVIS-DFT database files to extract carbide materials.

**Interface**:
```python
class JarvisClient:
    def __init__(self, jarvis_file_path: str):
        """Initialize with path to JARVIS JSON file."""
        
    def load_carbides(self, metal_elements: Set[str]) -> List[MaterialData]:
        """Extract carbide materials containing specified metals."""
        
    def extract_properties(self, jarvis_entry: Dict) -> Dict[str, float]:
        """Extract properties from JARVIS entry."""
        
    def parse_formula_from_jid(self, jid: str) -> Optional[str]:
        """Parse chemical formula from JARVIS identifier."""
```

**Key Methods**:
- `load_carbides()`: Filters 25,000+ materials to find 400+ carbides
- `extract_properties()`: Extracts formation energy, band gap, bulk modulus, etc.
- `parse_formula_from_jid()`: Handles JARVIS jid format (e.g., "JVASP-12345-SiC")

**Integration**: Extends `MaterialsProjectClient` pattern, returns `MaterialData` objects compatible with existing pipeline.

### 2. NIST-JANAF Client (`dft/nist_client.py`)

**Purpose**: Load and parse NIST-JANAF thermochemical tables for temperature-dependent properties.

**Interface**:
```python
class NISTClient:
    def __init__(self):
        """Initialize NIST data loader."""
        
    def load_thermochemical_data(self, filepath: str, material_name: str) -> pd.DataFrame:
        """Load temperature-dependent properties from NIST file."""
        
    def interpolate_property(self, data: pd.DataFrame, temperature: float, property_name: str) -> float:
        """Interpolate property value at specific temperature."""
```

**Data Format**:
```
# Temperature(K)  Cp(J/mol·K)  S°(J/mol·K)  H°-H°298(kJ/mol)  ΔfG°(kJ/mol)
298.15           26.86        16.55        0.000             -62.76
500.00           31.23        24.89        5.234             -65.43
1000.00          38.45        38.12        18.567            -71.89
```

**Integration**: Provides temperature-dependent data for materials, enhances property database.

### 3. Literature Database (`dft/literature_database.py`)

**Purpose**: Manage embedded literature reference data with experimental values.

**Interface**:
```python
class LiteratureDatabase:
    def __init__(self):
        """Initialize with embedded literature data."""
        
    def get_material_data(self, formula: str) -> Optional[Dict]:
        """Retrieve literature data for material."""
        
    def get_property_with_uncertainty(self, formula: str, property_name: str) -> Tuple[float, float]:
        """Get property value and uncertainty from literature."""
        
    def list_available_materials(self) -> List[str]:
        """List materials with literature data."""
```

**Data Structure**:
```python
{
    "SiC": {
        "hardness_vickers": 2800,
        "hardness_std": 200,
        "thermal_conductivity": 120,
        "tc_std": 10,
        "sources": ["Handbook of Advanced Ceramics", "INSPEC 1997"]
    }
}
```

**Integration**: Provides validation data and fills gaps in DFT databases.

### 4. Data Combiner (`dft/data_combiner.py`)

**Purpose**: Merge data from multiple sources with deduplication and conflict resolution.

**Interface**:
```python
class DataCombiner:
    def __init__(self):
        """Initialize data combiner."""
        
    def combine_sources(
        self,
        mp_data: List[MaterialData],
        jarvis_data: List[MaterialData],
        nist_data: Dict[str, pd.DataFrame],
        literature_data: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Combine all data sources into unified dataset."""
        
    def resolve_conflicts(self, material_id: str, properties: List[Dict]) -> Dict:
        """Resolve conflicting property values from different sources."""
        
    def deduplicate_materials(self, materials: List[MaterialData]) -> List[MaterialData]:
        """Remove duplicate materials based on formula."""
```

**Conflict Resolution Strategy**:
1. Experimental literature data (highest priority)
2. Materials Project DFT data
3. JARVIS-DFT data
4. NIST thermochemical data
5. Average if multiple sources agree within uncertainty

**Integration**: Creates unified dataset for downstream analysis.

### 5. Composition Descriptors (`ml/composition_descriptors.py`)

**Purpose**: Calculate composition-based material descriptors for ML features.

**Interface**:
```python
class CompositionDescriptorCalculator:
    def __init__(self):
        """Initialize descriptor calculator."""
        
    def calculate_descriptors(self, formula: str) -> Dict[str, float]:
        """Calculate all composition descriptors."""
        
    def calculate_composition_entropy(self, composition: Composition) -> float:
        """Calculate Shannon entropy of composition."""
        
    def calculate_electronegativity_features(self, composition: Composition) -> Dict[str, float]:
        """Calculate electronegativity-based features."""
        
    def calculate_atomic_property_features(self, composition: Composition) -> Dict[str, float]:
        """Calculate features from atomic properties."""
```

**Descriptors Calculated**:
- Number of elements
- Number of atoms
- Composition entropy: -Σ(f_i * log(f_i))
- Mean atomic radius, std, delta
- Mean electronegativity, std, delta
- Mean atomic number, mass
- Weighted atomic number

**Integration**: Extends `FeatureEngineeringPipeline` with new descriptor types.

### 6. Structure Descriptors (`ml/structure_descriptors.py`)

**Purpose**: Calculate structure-derived descriptors from material properties.

**Interface**:
```python
class StructureDescriptorCalculator:
    def __init__(self):
        """Initialize structure descriptor calculator."""
        
    def calculate_descriptors(self, properties: Dict[str, float]) -> Dict[str, float]:
        """Calculate all structure descriptors."""
        
    def calculate_pugh_ratio(self, bulk_modulus: float, shear_modulus: float) -> float:
        """Calculate Pugh ratio (B/G) for ductility."""
        
    def calculate_elastic_properties(self, bulk_modulus: float, shear_modulus: float) -> Dict[str, float]:
        """Calculate Young's modulus, Poisson ratio."""
        
    def calculate_energy_density(self, formation_energy: float, density: float) -> float:
        """Calculate energy per unit volume."""
```

**Descriptors Calculated**:
- Pugh ratio: B/G (ductility indicator)
- Young's modulus: 9BG/(3B+G)
- Poisson ratio: (3B-2G)/(2(3B+G))
- Ductility indicator: "ductile" if B/G > 1.75 else "brittle"
- Energy density: |E_f| / ρ

**Integration**: Extends `FeatureEngineeringPipeline` with derived features.

### 7. Application Ranker (`screening/application_ranker.py`)

**Purpose**: Rank materials for specific applications based on property requirements.

**Interface**:
```python
class ApplicationRanker:
    def __init__(self, applications: Dict[str, ApplicationSpec]):
        """Initialize with application specifications."""
        
    def rank_materials(
        self,
        materials: List[MaterialData],
        application: str
    ) -> List[RankedMaterial]:
        """Rank materials for specific application."""
        
    def calculate_application_score(
        self,
        material: MaterialData,
        application: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate compatibility score and component scores."""
        
    def rank_for_all_applications(
        self,
        materials: List[MaterialData]
    ) -> pd.DataFrame:
        """Rank materials for all defined applications."""
```

**Application Specifications**:
```python
@dataclass
class ApplicationSpec:
    name: str
    target_hardness: Optional[Tuple[float, float]]
    target_thermal_cond: Optional[Tuple[float, float]]
    target_melting_point: Optional[Tuple[float, float]]
    target_formation_energy: Optional[Tuple[float, float]]
    target_bulk_modulus: Optional[Tuple[float, float]]
    target_band_gap: Optional[Tuple[float, float]]
    weight: float
    description: str
```

**Scoring Algorithm**:
1. For each property requirement:
   - Score = 1.0 if within target range
   - Score = max(0, 1.0 - distance/threshold) if outside range
2. Weighted average of property scores
3. Missing properties reduce total weight proportionally

**Predefined Applications**:
- Aerospace Hypersonic (weight: 0.35)
- Cutting Tools (weight: 0.25)
- Thermal Barriers (weight: 0.20)
- Wear-Resistant Coatings (weight: 0.15)
- Electronic/Semiconductor (weight: 0.05)

**Integration**: Adds ranking capability to `ScreeningEngine`.

### 8. Experimental Planner (`validation/experimental_planner.py`)

**Purpose**: Generate experimental design recommendations for material synthesis and characterization.

**Interface**:
```python
class ExperimentalPlanner:
    def __init__(self):
        """Initialize experimental planner."""
        
    def design_synthesis_protocol(
        self,
        formula: str,
        properties: Dict[str, float]
    ) -> List[SynthesisMethod]:
        """Recommend synthesis methods."""
        
    def design_characterization_plan(
        self,
        formula: str,
        target_properties: List[str]
    ) -> List[CharacterizationTechnique]:
        """Recommend characterization techniques."""
        
    def estimate_resources(
        self,
        synthesis_methods: List[SynthesisMethod],
        characterization_plan: List[CharacterizationTechnique]
    ) -> ResourceEstimate:
        """Estimate time and cost."""
```

**Data Models**:
```python
@dataclass
class SynthesisMethod:
    method: str  # "Hot Pressing", "Solid State Sintering", "SPS"
    temperature_K: float
    pressure_MPa: float
    time_hours: float
    difficulty: str  # "Easy", "Medium", "Hard"
    cost_factor: float

@dataclass
class CharacterizationTechnique:
    technique: str  # "XRD", "SEM", "TEM", "Hardness Testing"
    estimated_time_hours: float
    estimated_cost_dollars: float

@dataclass
class ResourceEstimate:
    timeline_months: float
    estimated_cost_k_dollars: float
    required_equipment: List[str]
    required_expertise: List[str]
```

**Integration**: Extends validation framework with experimental planning.

### 9. Safe Conversion Utilities (`utils/data_conversion.py`)

**Purpose**: Robust conversion of heterogeneous data formats to numeric values.

**Interface**:
```python
def safe_float(value: Any) -> Optional[float]:
    """Safely convert any value to float."""
    
def safe_int(value: Any) -> Optional[int]:
    """Safely convert any value to int."""
    
def extract_numeric_from_dict(data: Dict, keys: List[str]) -> Optional[float]:
    """Extract numeric value from dictionary trying multiple keys."""
    
def extract_numeric_from_list(data: List) -> Optional[float]:
    """Extract numeric value from list (mean of numeric elements)."""
```

**Conversion Logic**:
1. Handle None and NaN → return None
2. Handle int/float → convert directly
3. Handle string → try float conversion
4. Handle dict → check keys ["value", "magnitude", "mean", "0"]
5. Handle list/tuple/array → compute mean of numeric elements
6. Fail gracefully → return None

**Integration**: Used throughout data loading and processing pipeline.

## Data Models

### Enhanced MaterialData

```python
@dataclass
class MaterialData:
    # Existing fields
    material_id: str
    formula: str
    composition: str
    
    # Enhanced with multi-source tracking
    data_sources: List[str]  # ["MP", "JARVIS", "NIST", "Literature"]
    source_priority: Dict[str, int]  # Property → source priority
    
    # Composition descriptors
    composition_descriptors: Optional[Dict[str, float]] = None
    
    # Structure descriptors
    structure_descriptors: Optional[Dict[str, float]] = None
    
    # Application scores
    application_scores: Optional[Dict[str, float]] = None
    
    # Temperature-dependent properties
    temperature_dependent_properties: Optional[Dict[str, pd.DataFrame]] = None
```

### RankedMaterial

```python
@dataclass
class RankedMaterial:
    material: MaterialData
    application: str
    overall_score: float
    component_scores: Dict[str, float]
    rank: int
    confidence: float
```

### ExperimentalProtocol

```python
@dataclass
class ExperimentalProtocol:
    formula: str
    synthesis_methods: List[SynthesisMethod]
    characterization_plan: List[CharacterizationTechnique]
    resource_estimate: ResourceEstimate
    priority_score: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: JARVIS carbide extraction completeness
*For any* JARVIS-DFT dataset, when filtering for carbides with specified metal elements, the system should identify all materials where the formula contains both carbon and at least one specified metal element
**Validates: Requirements 1.1**

### Property 2: Data source deduplication consistency
*For any* set of materials from multiple sources, after deduplication by formula, no two materials in the result should have identical chemical formulas
**Validates: Requirements 1.3**

### Property 3: Safe conversion robustness
*For any* input value of supported types (None, NaN, int, float, string, dict, list, tuple, array), the safe_float function should either return a valid float or None without raising exceptions
**Validates: Requirements 6.1, 6.4**

### Property 4: Composition descriptor validity
*For any* valid chemical formula, all calculated composition descriptors should be finite numeric values (not NaN or infinite)
**Validates: Requirements 2.1**

### Property 5: Structure descriptor physical bounds
*For any* material with bulk modulus B and shear modulus G where both are positive, the calculated Pugh ratio should equal B/G and Poisson ratio should be in range [-1, 0.5]
**Validates: Requirements 2.2**

### Property 6: Application scoring monotonicity
*For any* material and application, if all property values move closer to target ranges, the application score should not decrease
**Validates: Requirements 3.4**

### Property 7: Application score bounds
*For any* material and application, the calculated application score should be in range [0, 1]
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 8: Missing property handling
*For any* material with missing properties, the application scoring should compute partial scores based on available properties without failing
**Validates: Requirements 3.5**

### Property 9: Experimental protocol completeness
*For any* material candidate, the generated experimental protocol should include at least one synthesis method and at least one characterization technique
**Validates: Requirements 4.1, 4.3**

### Property 10: Resource estimate positivity
*For any* experimental protocol, the estimated timeline and cost should be positive values
**Validates: Requirements 4.4**

### Property 11: Report generation robustness
*For any* dataset (including empty or incomplete), the report generation should complete without exceptions and indicate missing data sections
**Validates: Requirements 5.5**

### Property 12: Feature engineering consistency
*For any* material, if the same material is processed twice with the same configuration, the calculated features should be identical
**Validates: Requirements 2.5**

### Property 13: Data combination preserves sources
*For any* combined dataset, each material should retain information about which data sources contributed to its properties
**Validates: Requirements 9.1**

### Property 14: ML training data validation
*For any* training dataset, after preparation, all feature values should be finite and all target values should be finite
**Validates: Requirements 7.1**

### Property 15: Pipeline stage independence
*For any* pipeline execution, if a stage fails, subsequent stages should either skip gracefully or use available data without cascading failures
**Validates: Requirements 8.2**

## Error Handling

### Data Loading Errors

```python
class DataLoadingError(Exception):
    """Base exception for data loading failures."""

class JarvisFileNotFoundError(DataLoadingError):
    """JARVIS file not found."""

class NISTParseError(DataLoadingError):
    """Failed to parse NIST data file."""

class FormulaExtractionError(DataLoadingError):
    """Failed to extract formula from identifier."""
```

**Handling Strategy**:
- Log warnings for individual material failures
- Continue processing remaining materials
- Return partial results with metadata about failures

### Feature Calculation Errors

```python
class FeatureCalculationError(Exception):
    """Base exception for feature calculation failures."""

class InvalidCompositionError(FeatureCalculationError):
    """Invalid chemical formula."""

class MissingPropertyError(FeatureCalculationError):
    """Required property missing for calculation."""
```

**Handling Strategy**:
- Skip invalid materials with logging
- Return partial feature sets when possible
- Indicate missing features in metadata

### Application Ranking Errors

```python
class RankingError(Exception):
    """Base exception for ranking failures."""

class InvalidApplicationError(RankingError):
    """Unknown application type."""

class InsufficientPropertiesError(RankingError):
    """Too few properties for meaningful ranking."""
```

**Handling Strategy**:
- Compute partial scores with available properties
- Indicate confidence reduction due to missing data
- Log warnings for materials with insufficient data

## Testing Strategy

### Unit Tests

**Data Loading Tests**:
```python
def test_jarvis_carbide_extraction():
    """Test JARVIS carbide filtering identifies correct materials."""
    
def test_nist_temperature_interpolation():
    """Test NIST data interpolation at arbitrary temperatures."""
    
def test_safe_float_conversion():
    """Test safe_float handles all supported data types."""
```

**Feature Engineering Tests**:
```python
def test_composition_descriptors_known_materials():
    """Test composition descriptors against known values for SiC, TiC."""
    
def test_structure_descriptors_physical_bounds():
    """Test structure descriptors satisfy physical constraints."""
```

**Application Ranking Tests**:
```python
def test_application_scoring_target_ranges():
    """Test materials in target range score higher than those outside."""
    
def test_application_score_bounds():
    """Test all scores are in [0, 1] range."""
```

### Integration Tests

```python
def test_full_pipeline_execution():
    """Test complete pipeline from data loading to reporting."""
    
def test_multi_source_data_combination():
    """Test combining MP, JARVIS, NIST, and literature data."""
    
def test_end_to_end_material_ranking():
    """Test ranking materials for all applications."""
```

### Property-Based Tests

```python
def test_property_safe_conversion_robustness():
    """Property 3: Test safe_float never raises exceptions."""
    
def test_property_application_score_bounds():
    """Property 7: Test application scores always in [0, 1]."""
    
def test_property_feature_consistency():
    """Property 12: Test repeated feature calculation gives same results."""
```

## Performance Optimization

### Data Loading Optimization

- **Lazy Loading**: Load JARVIS data only when needed
- **Caching**: Cache parsed JARVIS and NIST data
- **Parallel Processing**: Load multiple data sources in parallel

### Feature Engineering Optimization

- **Vectorization**: Use numpy for batch descriptor calculations
- **Memoization**: Cache composition descriptors for repeated formulas
- **Selective Calculation**: Only calculate requested descriptors

### Application Ranking Optimization

- **Batch Scoring**: Score all materials for all applications in single pass
- **Early Termination**: Skip detailed scoring for materials failing basic criteria
- **Parallel Ranking**: Rank different applications in parallel

## Integration Testing

### Test Scenarios

1. **Scenario 1: Load and combine all data sources**
   - Load MP, JARVIS, NIST, literature data
   - Combine with deduplication
   - Verify no data loss

2. **Scenario 2: Calculate features for combined dataset**
   - Extract composition and structure descriptors
   - Verify all materials have features
   - Check feature validity

3. **Scenario 3: Rank materials for all applications**
   - Score materials for 5 applications
   - Verify ranking consistency
   - Check score distributions

4. **Scenario 4: Generate experimental protocols**
   - Select top candidates
   - Generate synthesis and characterization plans
   - Verify resource estimates

5. **Scenario 5: Generate comprehensive report**
   - Create summary report
   - Verify all sections present
   - Check output formats

## Deployment Considerations

### Backward Compatibility

- All existing APIs remain unchanged
- New functionality accessed through new classes
- Existing workflows continue to work

### Configuration

```yaml
# config.yaml
data_sources:
  jarvis:
    enabled: true
    file_path: "data/jdft_3d-7-7-2018.json"
  nist:
    enabled: true
    data_dir: "data/nist"
  literature:
    enabled: true

feature_engineering:
  composition_descriptors: true
  structure_descriptors: true

application_ranking:
  enabled: true
  applications:
    - aerospace_hypersonic
    - cutting_tools
    - thermal_barriers
    - wear_resistant
    - electronic
```

### Documentation

- API documentation for all new classes
- Integration examples in notebooks
- Migration guide for existing users
- Tutorial for new features

This design provides a comprehensive integration plan that enhances the framework while maintaining consistency with existing architecture and ensuring all requirements are met.
