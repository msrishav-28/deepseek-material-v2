# Requirements Document: SiC Alloy Designer Integration

## Introduction

This specification defines the integration of the comprehensive SiC Alloy Designer functionality into the existing Ceramic Armor Discovery Framework. The integration will enhance the framework with multi-source data loading (JARVIS-DFT, NIST-JANAF, literature databases), advanced feature engineering, application-specific ranking systems, and experimental validation planning capabilities.

## Glossary

- **JARVIS-DFT**: Joint Automated Repository for Various Integrated Simulations - DFT database
- **NIST-JANAF**: NIST thermochemical database with temperature-dependent properties
- **Application Ranker**: System for scoring materials against specific use-case requirements
- **Feature Engineer**: Component that calculates composition and structure-based descriptors
- **Experimental Validator**: Framework for designing synthesis and characterization protocols
- **Safe Float Converter**: Utility for robust numeric data extraction from heterogeneous sources

## Requirements

### Requirement 1: Multi-Source Data Integration

**User Story:** As a materials researcher, I want to load data from JARVIS-DFT, NIST-JANAF, and literature databases in addition to Materials Project, so that I can access comprehensive material property data from multiple authoritative sources.

#### Acceptance Criteria

1. WHEN loading JARVIS-DFT data THEN the system SHALL extract carbide materials with formation energies and identify 400+ carbide compounds
2. WHEN loading NIST-JANAF data THEN the system SHALL parse temperature-dependent thermochemical properties including Cp, entropy, enthalpy, and Gibbs energy
3. WHEN combining data sources THEN the system SHALL merge Materials Project, JARVIS-DFT, NIST-JANAF, and literature data with proper deduplication
4. WHEN extracting formulas from JARVIS identifiers THEN the system SHALL correctly parse jid format to identify metal-carbide compositions
5. WHEN handling heterogeneous data formats THEN the system SHALL use safe conversion utilities to extract numeric values from dictionaries, lists, and nested structures

### Requirement 2: Advanced Feature Engineering

**User Story:** As a computational researcher, I want to calculate composition-based and structure-derived descriptors, so that I can generate rich feature sets for machine learning models.

#### Acceptance Criteria

1. WHEN calculating composition descriptors THEN the system SHALL compute atomic fractions, composition entropy, mean atomic properties, and electronegativity differences
2. WHEN calculating structure descriptors THEN the system SHALL derive Pugh ratio, Young's modulus, Poisson ratio, and ductility indicators from bulk and shear moduli
3. WHEN processing material formulas THEN the system SHALL use pymatgen Composition objects to extract element properties including atomic radii and electronegativities
4. WHEN computing derived properties THEN the system SHALL calculate energy density from formation energy and density
5. WHEN handling missing data THEN the system SHALL gracefully skip calculations and log warnings without failing the entire pipeline

### Requirement 3: Application-Specific Material Ranking

**User Story:** As a materials designer, I want to rank materials for specific applications like hypersonic turbines, cutting tools, and thermal barriers, so that I can identify optimal candidates for each use case.

#### Acceptance Criteria

1. WHEN ranking for aerospace applications THEN the system SHALL score materials based on hardness (28-35 GPa), thermal conductivity (50-150 W/m·K), and melting point (2500-4000 K) targets
2. WHEN ranking for cutting tools THEN the system SHALL prioritize hardness (30-40 GPa) and formation energy (-4.0 to -2.0 eV/atom) criteria
3. WHEN ranking for thermal barriers THEN the system SHALL evaluate thermal conductivity (20-60 W/m·K) and density (1-6 g/cm³) ranges
4. WHEN calculating application scores THEN the system SHALL use weighted scoring with configurable weights for each application type
5. WHEN materials lack required properties THEN the system SHALL compute partial scores based on available data and indicate missing property contributions

### Requirement 4: Experimental Validation Planning

**User Story:** As an experimental researcher, I want automated experimental design recommendations for top material candidates, so that I can efficiently plan synthesis and characterization workflows.

#### Acceptance Criteria

1. WHEN designing experiments for candidates THEN the system SHALL recommend synthesis methods including hot pressing, solid-state sintering, and spark plasma sintering
2. WHEN specifying synthesis parameters THEN the system SHALL provide temperature, pressure, time, difficulty level, and cost factor estimates
3. WHEN planning characterization THEN the system SHALL list required techniques including XRD, SEM, TEM, hardness testing, and thermal conductivity measurement
4. WHEN estimating resources THEN the system SHALL provide timeline (months) and cost (thousands of dollars) projections
5. WHEN generating experimental protocols THEN the system SHALL output structured data suitable for laboratory workflow management systems

### Requirement 5: Comprehensive Reporting System

**User Story:** As a research team lead, I want automated generation of comprehensive analysis reports, so that I can quickly review results and make informed decisions.

#### Acceptance Criteria

1. WHEN generating summary reports THEN the system SHALL include dataset statistics, property distributions, ML model performance, and top material recommendations
2. WHEN reporting ML results THEN the system SHALL display model type, R² score, MAE, RMSE, number of features, and training samples
3. WHEN presenting top candidates THEN the system SHALL show formula, application scores, and ranking for multiple application types
4. WHEN saving reports THEN the system SHALL output both text format for logging and structured formats (JSON, CSV) for downstream analysis
5. WHEN reports are incomplete THEN the system SHALL gracefully handle missing sections and indicate data availability issues

### Requirement 6: Robust Data Handling Utilities

**User Story:** As a data engineer, I want robust utilities for handling heterogeneous data formats from multiple sources, so that the system can reliably extract numeric values regardless of source format variations.

#### Acceptance Criteria

1. WHEN converting values to float THEN the system SHALL handle None, NaN, integers, floats, strings, dictionaries, lists, tuples, and numpy arrays
2. WHEN extracting from dictionaries THEN the system SHALL check common keys including "value", "magnitude", "mean", and numeric keys
3. WHEN processing lists or arrays THEN the system SHALL compute mean of numeric elements
4. WHEN conversion fails THEN the system SHALL return None without raising exceptions
5. WHEN extracting formulas from identifiers THEN the system SHALL parse JARVIS jid format and validate presence of metal and carbon elements

### Requirement 7: Machine Learning Model Training

**User Story:** As a machine learning engineer, I want to train Random Forest and Gradient Boosting models on material properties, so that I can predict target properties for new materials.

#### Acceptance Criteria

1. WHEN preparing training data THEN the system SHALL remove rows with missing targets, filter non-numeric features, and remove infinite values
2. WHEN training Random Forest models THEN the system SHALL use 150 estimators, max depth 15, and proper train-test splitting
3. WHEN training Gradient Boosting models THEN the system SHALL use 150 estimators, learning rate 0.1, and max depth 5
4. WHEN evaluating models THEN the system SHALL compute R², MAE, RMSE, and MAPE metrics
5. WHEN insufficient data exists THEN the system SHALL skip training and log warnings without failing the pipeline

### Requirement 8: Pipeline Orchestration

**User Story:** As a computational scientist, I want a complete pipeline that orchestrates data loading, feature engineering, ML training, ranking, and reporting, so that I can run end-to-end analyses with a single command.

#### Acceptance Criteria

1. WHEN running the full pipeline THEN the system SHALL execute data loading, feature engineering, ML training, application ranking, experimental design, and reporting stages in sequence
2. WHEN any stage fails THEN the system SHALL log errors, continue with available data, and indicate incomplete results
3. WHEN the pipeline completes THEN the system SHALL save combined data, engineered features, and material rankings to CSV files
4. WHEN generating outputs THEN the system SHALL create analysis reports in text format and save to configurable output directories
5. WHEN no data is loaded THEN the system SHALL terminate gracefully with clear error messages indicating data source issues

### Requirement 9: Integration with Existing Framework

**User Story:** As a framework maintainer, I want the SiC alloy designer functionality integrated into existing modules, so that users can access enhanced capabilities through familiar interfaces.

#### Acceptance Criteria

1. WHEN integrating data loaders THEN the system SHALL extend existing DFT data collection modules with JARVIS and NIST capabilities
2. WHEN integrating feature engineering THEN the system SHALL add composition and structure descriptors to existing feature engineering pipelines
3. WHEN integrating ML models THEN the system SHALL register new model types with existing model trainer infrastructure
4. WHEN integrating ranking THEN the system SHALL add application-specific ranking to existing screening workflows
5. WHEN integrating reporting THEN the system SHALL extend existing report generators with new summary formats

### Requirement 10: Testing and Validation

**User Story:** As a quality assurance engineer, I want comprehensive tests for all integrated functionality, so that I can ensure reliability and correctness of the enhanced framework.

#### Acceptance Criteria

1. WHEN testing data loaders THEN the system SHALL verify correct parsing of JARVIS JSON, NIST text files, and literature dictionaries
2. WHEN testing feature engineering THEN the system SHALL validate composition descriptor calculations against known materials
3. WHEN testing application ranking THEN the system SHALL verify scoring logic for all application types with test materials
4. WHEN testing safe conversion utilities THEN the system SHALL validate handling of all supported data types and edge cases
5. WHEN testing the full pipeline THEN the system SHALL execute end-to-end integration tests with sample datasets

## Technical Implementation Requirements

### Data Standards
- JARVIS-DFT JSON format with jid-based material identification
- NIST-JANAF text format with temperature-dependent property tables
- Literature data as Python dictionaries with property values and uncertainties
- Consistent unit handling across all data sources

### Integration Points
- Extend `src/ceramic_discovery/dft/` with JARVIS and NIST clients
- Enhance `src/ceramic_discovery/ml/feature_engineering.py` with new descriptors
- Add application ranking to `src/ceramic_discovery/screening/`
- Extend `src/ceramic_discovery/reporting/` with new report formats
- Add utilities to `src/ceramic_discovery/utils/`

### Performance Requirements
- Handle JARVIS datasets with 25,000+ materials efficiently
- Process NIST temperature tables with hundreds of data points
- Complete full pipeline execution in under 30 minutes for typical datasets
- Support parallel processing for batch material analysis

This requirements document establishes the foundation for integrating the comprehensive SiC Alloy Designer functionality into the existing Ceramic Armor Discovery Framework, enhancing its capabilities while maintaining consistency with the established architecture.
