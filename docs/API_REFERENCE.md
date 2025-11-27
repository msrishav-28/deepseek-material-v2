# API Reference

## Ceramic Armor Discovery Framework - API Documentation

**Version:** 1.0  
**Date:** 2025-11-23  
**Requirements:** 7.3, 7.4, 7.5

---

## Table of Contents

1. [DFT Module](#dft-module)
2. [Ceramics Module](#ceramics-module)
3. [ML Module](#ml-module)
4. [Ballistics Module](#ballistics-module)
5. [Screening Module](#screening-module)
6. [Validation Module](#validation-module)
7. [Analysis Module](#analysis-module)
8. [Reporting Module](#reporting-module)
9. [Utilities](#utilities)

---

## DFT Module

### StabilityAnalyzer

Analyzes thermodynamic stability of materials using DFT data.

```python
from ceramic_discovery.dft import StabilityAnalyzer

analyzer = StabilityAnalyzer(metastable_threshold=0.1)
```

#### Methods

**`analyze_stability(material_id, formula, energy_above_hull, formation_energy_per_atom)`**

Analyze stability of a single material.

Parameters:
- `material_id` (str): Unique identifier for material
- `formula` (str): Chemical formula
- `energy_above_hull` (float): Energy above convex hull (eV/atom)
- `formation_energy_per_atom` (float): Formation energy (eV/atom)

Returns:
- `StabilityResult`: Object containing classification, viability, and confidence

Example:
```python
result = analyzer.analyze_stability(
    material_id="mp-149",
    formula="SiC",
    energy_above_hull=0.0,
    formation_energy_per_atom=-0.65
)

print(f"Classification: {result.classification.name}")
print(f"Viable: {result.is_viable}")
print(f"Confidence: {result.confidence_score:.2f}")
```

**`batch_analyze(materials)`**

Analyze stability of multiple materials.

Parameters:
- `materials` (List[Dict]): List of material dictionaries

Returns:
- `List[StabilityResult]`: List of stability results

Example:
```python
materials = [
    {"material_id": "mat1", "formula": "SiC", "energy_above_hull": 0.0, ...},
    {"material_id": "mat2", "formula": "Si0.98B0.02C", "energy_above_hull": 0.02, ...},
]

results = analyzer.batch_analyze(materials)
viable_count = sum(1 for r in results if r.is_viable)
```

**`is_viable(energy_above_hull)`**

Check if material is viable based on energy above hull.

Parameters:
- `energy_above_hull` (float): Energy above convex hull (eV/atom)

Returns:
- `bool`: True if ΔE_hull ≤ threshold

---

### MaterialsProjectClient

Client for Materials Project API.

```python
from ceramic_discovery.dft import MaterialsProjectClient

client = MaterialsProjectClient(
    api_key="your_api_key",
    requests_per_second=10
)
```

#### Methods

**`get_material_data(material_id)`**

Retrieve material data from Materials Project.

Parameters:
- `material_id` (str): Materials Project ID (e.g., "mp-149")

Returns:
- `Dict`: Material data including structure, properties, and energies

**`search_materials(formula, properties=None)`**

Search for materials by formula.

Parameters:
- `formula` (str): Chemical formula
- `properties` (List[str], optional): Properties to retrieve

Returns:
- `List[Dict]`: List of matching materials

---

### PropertyExtractor

Extract and standardize material properties.

```python
from ceramic_discovery.dft import PropertyExtractor

extractor = PropertyExtractor()
```

#### Methods

**`extract_all_properties(dft_result)`**

Extract all 58 standardized properties from DFT result.

Parameters:
- `dft_result` (Dict): DFT calculation result

Returns:
- `Dict`: Standardized properties with units

---

### JarvisClient

Client for loading and parsing JARVIS-DFT database files.

```python
from ceramic_discovery.dft import JarvisClient

client = JarvisClient(jarvis_file_path="./data/jdft_3d-7-7-2018.json")
```

#### Methods

**`load_carbides(metal_elements=None)`**

Extract carbide materials containing specified metals.

Parameters:
- `metal_elements` (Set[str], optional): Set of metal element symbols to filter for

Returns:
- `List[MaterialData]`: List of carbide materials

Example:
```python
# Load all carbides
all_carbides = client.load_carbides()

# Load carbides with specific metals
sic_tic_carbides = client.load_carbides(metal_elements={"Si", "Ti"})
print(f"Found {len(sic_tic_carbides)} Si/Ti carbides")
```

**`parse_formula_from_jid(jid)`**

Parse chemical formula from JARVIS identifier.

Parameters:
- `jid` (str): JARVIS identifier (e.g., "JVASP-12345")

Returns:
- `Optional[str]`: Chemical formula or None

---

### NISTClient

Client for loading NIST-JANAF thermochemical data.

```python
from ceramic_discovery.dft import NISTClient

client = NISTClient()
```

#### Methods

**`load_thermochemical_data(filepath, material_name)`**

Load temperature-dependent properties from NIST file.

Parameters:
- `filepath` (str): Path to NIST data file
- `material_name` (str): Material name for identification

Returns:
- `pd.DataFrame`: Temperature-dependent properties

Example:
```python
sic_data = client.load_thermochemical_data(
    filepath="./data/nist/SiC.txt",
    material_name="SiC"
)

print(sic_data.columns)  # ['Temperature', 'Cp', 'S', 'H-H298', 'DeltaG']
```

**`interpolate_property(data, temperature, property_name)`**

Interpolate property value at specific temperature.

Parameters:
- `data` (pd.DataFrame): Thermochemical data
- `temperature` (float): Temperature in Kelvin
- `property_name` (str): Property to interpolate

Returns:
- `float`: Interpolated property value

Example:
```python
cp_1000K = client.interpolate_property(
    data=sic_data,
    temperature=1000.0,
    property_name="Cp"
)
```

---

### LiteratureDatabase

Manage embedded literature reference data.

```python
from ceramic_discovery.dft import LiteratureDatabase

lit_db = LiteratureDatabase()
```

#### Methods

**`get_material_data(formula)`**

Retrieve literature data for material.

Parameters:
- `formula` (str): Chemical formula

Returns:
- `Optional[Dict]`: Material data or None

**`get_property_with_uncertainty(formula, property_name)`**

Get property value and uncertainty from literature.

Parameters:
- `formula` (str): Chemical formula
- `property_name` (str): Property name

Returns:
- `Tuple[float, float]`: (value, uncertainty)

Example:
```python
hardness, uncertainty = lit_db.get_property_with_uncertainty(
    formula="SiC",
    property_name="hardness_vickers"
)
print(f"SiC hardness: {hardness:.0f} ± {uncertainty:.0f} HV")
```

**`list_available_materials()`**

List materials with literature data.

Returns:
- `List[str]`: List of available material formulas

---

### DataCombiner

Merge data from multiple sources with deduplication.

```python
from ceramic_discovery.dft import DataCombiner

combiner = DataCombiner()
```

#### Methods

**`combine_sources(mp_data, jarvis_data, nist_data, literature_data)`**

Combine all data sources into unified dataset.

Parameters:
- `mp_data` (List[MaterialData]): Materials Project data
- `jarvis_data` (List[MaterialData]): JARVIS-DFT data
- `nist_data` (Dict[str, pd.DataFrame]): NIST thermochemical data
- `literature_data` (Dict[str, Dict]): Literature data

Returns:
- `pd.DataFrame`: Combined dataset

Example:
```python
combined = combiner.combine_sources(
    mp_data=mp_materials,
    jarvis_data=jarvis_carbides,
    nist_data={"SiC": sic_thermo},
    literature_data=lit_db.get_all_data()
)

print(f"Combined {len(combined)} unique materials")
```

**`deduplicate_materials(materials)`**

Remove duplicate materials based on formula.

Parameters:
- `materials` (List[MaterialData]): Materials to deduplicate

Returns:
- `List[MaterialData]`: Deduplicated materials

---

## Ceramics Module

### CeramicSystemFactory

Factory for creating baseline ceramic systems.

```python
from ceramic_discovery.ceramics import CeramicSystemFactory
```

#### Methods

**`create_sic()`**

Create SiC baseline system.

Returns:
- `CeramicSystem`: SiC with literature-validated properties

Example:
```python
sic = CeramicSystemFactory.create_sic()
print(f"Hardness: {sic.mechanical.hardness} GPa")
print(f"K_IC: {sic.mechanical.fracture_toughness} MPa·m^0.5")
```

**`create_b4c()`**, **`create_wc()`**, **`create_tic()`**, **`create_al2o3()`**

Create other baseline ceramic systems.

---

### CompositeCalculator

Calculate composite material properties.

```python
from ceramic_discovery.ceramics import CompositeCalculator

calculator = CompositeCalculator()
```

#### Methods

**`calculate_composite_properties(system1, system2, volume_fraction_1, volume_fraction_2)`**

Calculate composite properties using rule-of-mixtures.

Parameters:
- `system1` (CeramicSystem): First constituent
- `system2` (CeramicSystem): Second constituent
- `volume_fraction_1` (float): Volume fraction of first constituent
- `volume_fraction_2` (float): Volume fraction of second constituent

Returns:
- `CeramicSystem`: Composite system with estimated properties

**Warning:** Includes explicit limitations warning.

---

## ML Module

### ModelTrainer

Train and evaluate ML models.

```python
from ceramic_discovery.ml import ModelTrainer

trainer = ModelTrainer(random_state=42)
```

#### Methods

**`train(X, y)`**

Train model on data.

Parameters:
- `X` (np.ndarray): Feature matrix (n_samples, n_features)
- `y` (np.ndarray): Target values (n_samples,)

Returns:
- `TrainedModel`: Trained model object

Example:
```python
trainer.train(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)
print(f"R²: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.1f}")
```

**`predict(X)`**

Make predictions.

Parameters:
- `X` (np.ndarray): Feature matrix

Returns:
- `np.ndarray`: Predictions

**`predict_with_uncertainty(X)`**

Make predictions with uncertainty quantification.

Parameters:
- `X` (np.ndarray): Feature matrix

Returns:
- `PredictionWithUncertainty`: Predictions with confidence intervals

**`get_feature_importance()`**

Get feature importance scores.

Returns:
- `np.ndarray`: Feature importance values

**`save_model(filepath)`**, **`load_model(filepath)`**

Save/load trained model.

---

### FeatureEngineeringPipeline

Feature engineering with Tier 1-2 validation.

```python
from ceramic_discovery.ml import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()
```

#### Methods

**`extract_features(material)`**

Extract features from material data.

Parameters:
- `material` (Dict): Material properties

Returns:
- `Dict`: Extracted features

**`create_feature_matrix(materials)`**

Create feature matrix from multiple materials.

Parameters:
- `materials` (List[Dict] or Dict): Material data

Returns:
- `np.ndarray`: Feature matrix

**`validate_features(feature_names)`**

Validate that features are Tier 1-2 only.

Parameters:
- `feature_names` (List[str]): Feature names to validate

Returns:
- `Tuple[bool, List[str]]`: (is_valid, violations)

---

### UncertaintyQuantifier

Quantify prediction uncertainty.

```python
from ceramic_discovery.ml import UncertaintyQuantifier

uq = UncertaintyQuantifier()
```

#### Methods

**`bootstrap_prediction_intervals(model, X_train, y_train, X_test, confidence=0.95, n_bootstrap=1000)`**

Generate prediction intervals using bootstrap sampling.

Parameters:
- `model`: Trained model
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training targets
- `X_test` (np.ndarray): Test features
- `confidence` (float): Confidence level (default: 0.95)
- `n_bootstrap` (int): Number of bootstrap samples

Returns:
- `np.ndarray`: Prediction intervals (n_samples, 2) [lower, upper]

Example:
```python
intervals = uq.bootstrap_prediction_intervals(
    model=trainer.model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    confidence=0.95
)

for i, (lower, upper) in enumerate(intervals):
    print(f"Sample {i}: [{lower:.0f}, {upper:.0f}]")
```

---

### CompositionDescriptorCalculator

Calculate composition-based material descriptors.

```python
from ceramic_discovery.ml import CompositionDescriptorCalculator

calc = CompositionDescriptorCalculator()
```

#### Methods

**`calculate_descriptors(formula)`**

Calculate all composition descriptors for a formula.

Parameters:
- `formula` (str): Chemical formula

Returns:
- `Dict[str, float]`: Composition descriptors

Example:
```python
descriptors = calc.calculate_descriptors("SiC")

print(f"Number of elements: {descriptors['n_elements']}")
print(f"Composition entropy: {descriptors['composition_entropy']:.3f}")
print(f"Mean electronegativity: {descriptors['mean_electronegativity']:.3f}")
```

**`calculate_composition_entropy(composition)`**

Calculate Shannon entropy of composition.

Parameters:
- `composition` (Composition): Pymatgen Composition object

Returns:
- `float`: Composition entropy

**`calculate_electronegativity_features(composition)`**

Calculate electronegativity-based features.

Parameters:
- `composition` (Composition): Pymatgen Composition object

Returns:
- `Dict[str, float]`: Electronegativity features (mean, std, delta)

---

### StructureDescriptorCalculator

Calculate structure-derived descriptors.

```python
from ceramic_discovery.ml import StructureDescriptorCalculator

calc = StructureDescriptorCalculator()
```

#### Methods

**`calculate_descriptors(properties)`**

Calculate all structure descriptors from properties.

Parameters:
- `properties` (Dict[str, float]): Material properties

Returns:
- `Dict[str, float]`: Structure descriptors

Example:
```python
properties = {
    "bulk_modulus": 220.0,
    "shear_modulus": 190.0,
    "formation_energy_per_atom": -0.65,
    "density": 3.21
}

descriptors = calc.calculate_descriptors(properties)

print(f"Pugh ratio: {descriptors['pugh_ratio']:.3f}")
print(f"Young's modulus: {descriptors['youngs_modulus']:.1f} GPa")
print(f"Poisson ratio: {descriptors['poisson_ratio']:.3f}")
```

**`calculate_pugh_ratio(bulk_modulus, shear_modulus)`**

Calculate Pugh ratio (B/G) for ductility.

Parameters:
- `bulk_modulus` (float): Bulk modulus in GPa
- `shear_modulus` (float): Shear modulus in GPa

Returns:
- `float`: Pugh ratio

**`calculate_elastic_properties(bulk_modulus, shear_modulus)`**

Calculate Young's modulus and Poisson ratio.

Parameters:
- `bulk_modulus` (float): Bulk modulus in GPa
- `shear_modulus` (float): Shear modulus in GPa

Returns:
- `Dict[str, float]`: Elastic properties

---

## Ballistics Module

### BallisticPredictor

Predict ballistic performance (V₅₀).

```python
from ceramic_discovery.ballistics import BallisticPredictor

predictor = BallisticPredictor()
predictor.load_model("./models/v50_predictor_v1.pkl")
```

#### Methods

**`predict_v50(properties)`**

Predict V₅₀ with uncertainty.

Parameters:
- `properties` (Dict): Material properties

Returns:
- `Dict`: Prediction with keys:
  - `v50` (float): Predicted V₅₀ (m/s)
  - `confidence_interval` (Tuple[float, float]): 95% CI
  - `reliability_score` (float): Reliability (0-1)

Example:
```python
properties = {
    "hardness": 28.0,
    "fracture_toughness": 4.5,
    "density": 3.21,
    "thermal_conductivity_1000C": 45.0,
}

prediction = predictor.predict_v50(properties)
print(f"V₅₀: {prediction['v50']:.0f} m/s")
print(f"95% CI: [{prediction['confidence_interval'][0]:.0f}, {prediction['confidence_interval'][1]:.0f}]")
print(f"Reliability: {prediction['reliability_score']:.2f}")
```

**`load_model(filepath)`**, **`save_model(filepath)`**

Load/save predictor model.

---

### MechanismAnalyzer

Analyze physical mechanisms.

```python
from ceramic_discovery.ballistics import MechanismAnalyzer

analyzer = MechanismAnalyzer()
```

#### Methods

**`analyze_thermal_conductivity_impact(materials, v50_values)`**

Analyze thermal conductivity impact on performance.

Parameters:
- `materials` (List[Dict]): Material properties
- `v50_values` (List[float]): Corresponding V₅₀ values

Returns:
- `Dict`: Analysis results with correlation and impact magnitude

**`analyze_property_contributions(materials, v50_values)`**

Analyze property contributions to performance.

Parameters:
- `materials` (List[Dict]): Material properties
- `v50_values` (List[float]): Corresponding V₅₀ values

Returns:
- `Dict[str, float]`: Property contributions

---

## Screening Module

### ScreeningEngine

High-throughput materials screening.

```python
from ceramic_discovery.screening import ScreeningEngine
from ceramic_discovery.dft import StabilityAnalyzer

analyzer = StabilityAnalyzer()
engine = ScreeningEngine(stability_analyzer=analyzer)
```

#### Methods

**`screen_materials(materials, stability_threshold=0.1, min_hardness=None, min_fracture_toughness=None)`**

Screen materials by criteria.

Parameters:
- `materials` (List[Dict]): Materials to screen
- `stability_threshold` (float): Max ΔE_hull (eV/atom)
- `min_hardness` (float, optional): Minimum hardness (GPa)
- `min_fracture_toughness` (float, optional): Minimum K_IC (MPa·m^0.5)

Returns:
- `List[Dict]`: Filtered and ranked materials

Example:
```python
results = engine.screen_materials(
    materials=candidate_materials,
    stability_threshold=0.1,
    min_hardness=25.0,
    min_fracture_toughness=4.0
)

print(f"Viable materials: {len(results)}")
for i, material in enumerate(results[:5], 1):
    print(f"{i}. {material['formula']}: V₅₀ = {material['predicted_v50']:.0f} m/s")
```

---

### WorkflowOrchestrator

Orchestrate complex workflows.

```python
from ceramic_discovery.screening import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator(output_dir="./results")
```

#### Methods

**`execute_workflow(workflow_id, workflow_function, parameters)`**

Execute a workflow.

Parameters:
- `workflow_id` (str): Unique workflow identifier
- `workflow_function` (Callable): Function to execute
- `parameters` (Dict): Workflow parameters

Returns:
- `Dict`: Workflow results

**`get_workflow_status(workflow_id=None)`**

Get workflow status.

Parameters:
- `workflow_id` (str, optional): Specific workflow ID

Returns:
- `Dict`: Status information

---

### ApplicationRanker

Rank materials for specific applications.

```python
from ceramic_discovery.screening import ApplicationRanker

ranker = ApplicationRanker()
```

#### Methods

**`rank_materials(materials, application)`**

Rank materials for specific application.

Parameters:
- `materials` (List[Dict]): Materials to rank
- `application` (str): Application name

Returns:
- `List[RankedMaterial]`: Ranked materials

Example:
```python
aerospace_ranking = ranker.rank_materials(
    materials=materials,
    application="aerospace_hypersonic"
)

for ranked in aerospace_ranking[:5]:
    print(f"{ranked.formula}: Score {ranked.overall_score:.3f}")
```

**`rank_for_all_applications(materials)`**

Rank materials for all defined applications.

Parameters:
- `materials` (List[Dict]): Materials to rank

Returns:
- `pd.DataFrame`: Rankings for all applications

Example:
```python
all_rankings = ranker.rank_for_all_applications(materials)

# Find top material for each application
top_by_app = all_rankings.loc[all_rankings.groupby('application')['overall_score'].idxmax()]
print(top_by_app[['application', 'formula', 'overall_score']])
```

**`list_applications()`**

List available applications.

Returns:
- `List[str]`: Application names

**`get_application_spec(application)`**

Get specification for an application.

Parameters:
- `application` (str): Application name

Returns:
- `ApplicationSpec`: Application specification

---

### SiCAlloyDesignerPipeline

Complete pipeline orchestrating all stages.

```python
from ceramic_discovery.screening import SiCAlloyDesignerPipeline

pipeline = SiCAlloyDesignerPipeline(
    jarvis_file="./data/jdft_3d-7-7-2018.json",
    output_dir="./results/sic_pipeline"
)
```

#### Methods

**`run_full_pipeline()`**

Execute complete pipeline from data loading to reporting.

Returns:
- `Dict`: Pipeline results

Example:
```python
results = pipeline.run_full_pipeline()

print(f"Materials loaded: {results['n_materials_loaded']}")
print(f"Top candidates: {results['n_top_candidates']}")
print(f"Output directory: {results['output_dir']}")
```

**`run_stage(stage_name)`**

Execute specific pipeline stage.

Parameters:
- `stage_name` (str): Stage name ('data_loading', 'feature_engineering', etc.)

Returns:
- `Dict`: Stage results

---

## Validation Module

### PhysicalPlausibilityValidator

Validate physical plausibility of properties.

```python
from ceramic_discovery.validation import PhysicalPlausibilityValidator

validator = PhysicalPlausibilityValidator()
```

#### Methods

**`validate_material(material_id, properties)`**

Validate material properties.

Parameters:
- `material_id` (str): Material identifier
- `properties` (Dict): Properties to validate

Returns:
- `ValidationReport`: Validation results

Example:
```python
properties = {
    "hardness": 28.0,
    "fracture_toughness": 4.5,
    "density": 3.21,
    "thermal_conductivity_25C": 120.0,
}

report = validator.validate_material("SiC", properties)

if report.is_valid:
    print("✓ All properties valid")
else:
    print("⚠ Validation issues:")
    for violation in report.violations:
        print(f"  - {violation.property_name}: {violation.message}")
```

**`validate_property(property_name, value)`**

Validate single property.

Parameters:
- `property_name` (str): Property name
- `value` (float): Property value

Returns:
- `ValidationViolation` or `None`: Violation if invalid

**`add_literature_reference(material_id, property_name, value, uncertainty, source)`**

Add literature reference for validation.

**`cross_reference_literature(material_id, property_name, predicted_value, predicted_uncertainty)`**

Cross-reference prediction with literature.

---

### ReproducibilityFramework

Ensure reproducibility of research workflows.

```python
from ceramic_discovery.validation import ReproducibilityFramework

framework = ReproducibilityFramework(
    experiment_dir="./results/experiments",
    master_seed=42
)
```

#### Methods

**`start_experiment(experiment_id, experiment_type, parameters, random_seed=None)`**

Start a new experiment.

Parameters:
- `experiment_id` (str): Unique experiment identifier
- `experiment_type` (str): Type of experiment
- `parameters` (Dict): Experiment parameters
- `random_seed` (int, optional): Random seed (uses master_seed if None)

Returns:
- `ExperimentSnapshot`: Experiment snapshot

**`record_transformation(description, data)`**

Record a data transformation step.

Parameters:
- `description` (str): Description of transformation
- `data` (Dict): Transformation data

**`finalize_experiment(results)`**

Finalize experiment and save snapshot.

Parameters:
- `results` (Dict): Experiment results

Returns:
- `ExperimentSnapshot`: Final snapshot

**`export_reproducibility_package(experiment_id, output_dir)`**

Export complete reproducibility package.

Parameters:
- `experiment_id` (str): Experiment to export
- `output_dir` (Path): Output directory

Example:
```python
# Start experiment
framework.start_experiment(
    experiment_id="sic_screening_001",
    experiment_type="dopant_screening",
    parameters={"base_system": "SiC", "n_dopants": 5}
)

# Record steps
framework.record_transformation("DFT collection", {"n_materials": 25})
framework.record_transformation("Stability filter", {"n_viable": 18})

# Finalize
results = {"top_candidates": 10}
snapshot = framework.finalize_experiment(results)

# Export package
framework.export_reproducibility_package(
    "sic_screening_001",
    "./results/reproducibility_packages/sic_screening_001"
)
```

---

### RandomSeedManager

Manage random seeds for reproducibility.

```python
from ceramic_discovery.validation import RandomSeedManager

manager = RandomSeedManager(master_seed=42)
```

#### Methods

**`set_global_seed()`**

Set global random seed for all libraries.

**`get_component_seed(component_name)`**

Get deterministic seed for a component.

Parameters:
- `component_name` (str): Component identifier

Returns:
- `int`: Component-specific seed

**`create_rng(component_name)`**

Create component-specific random number generator.

Parameters:
- `component_name` (str): Component identifier

Returns:
- `np.random.Generator`: Random number generator

---

### ExperimentalPlanner

Generate experimental design recommendations.

```python
from ceramic_discovery.validation import ExperimentalPlanner

planner = ExperimentalPlanner()
```

#### Methods

**`design_synthesis_protocol(formula, properties)`**

Recommend synthesis methods for a material.

Parameters:
- `formula` (str): Chemical formula
- `properties` (Dict[str, float]): Material properties

Returns:
- `List[SynthesisMethod]`: Recommended synthesis methods

Example:
```python
methods = planner.design_synthesis_protocol(
    formula="HfC",
    properties={"melting_point": 3900, "hardness": 28.0}
)

for method in methods:
    print(f"{method.method}: {method.temperature_K}K, {method.pressure_MPa}MPa")
```

**`design_characterization_plan(formula, target_properties)`**

Recommend characterization techniques.

Parameters:
- `formula` (str): Chemical formula
- `target_properties` (List[str]): Properties to characterize

Returns:
- `List[CharacterizationTechnique]`: Recommended techniques

Example:
```python
techniques = planner.design_characterization_plan(
    formula="HfC",
    target_properties=["hardness", "thermal_conductivity", "phase_composition"]
)

for tech in techniques:
    print(f"{tech.technique}: ${tech.estimated_cost_dollars:.0f}, {tech.estimated_time_hours:.1f}h")
```

**`estimate_resources(synthesis_methods, characterization_plan)`**

Estimate time and cost for experimental work.

Parameters:
- `synthesis_methods` (List[SynthesisMethod]): Synthesis methods
- `characterization_plan` (List[CharacterizationTechnique]): Characterization techniques

Returns:
- `ResourceEstimate`: Resource estimates

Example:
```python
estimate = planner.estimate_resources(
    synthesis_methods=methods,
    characterization_plan=techniques
)

print(f"Timeline: {estimate.timeline_months:.1f} months")
print(f"Cost: ${estimate.estimated_cost_k_dollars:.1f}K")
print(f"Confidence: {estimate.confidence_level:.2f}")
```

---

## Analysis Module

### PropertyAnalyzer

Analyze material properties.

```python
from ceramic_discovery.analysis import PropertyAnalyzer

analyzer = PropertyAnalyzer()
```

#### Methods

**`analyze_distribution(properties, property_name)`**

Analyze property distribution.

**`correlation_analysis(materials, properties)`**

Analyze correlations between properties.

**`identify_outliers(materials, property_name, threshold=3.0)`**

Identify outliers using z-score.

---

### Visualizer

Create visualizations.

```python
from ceramic_discovery.analysis import Visualizer

viz = Visualizer()
```

#### Methods

**`parallel_coordinates(materials, properties, output=None)`**

Create parallel coordinates plot.

Parameters:
- `materials` (List[Dict]): Materials to plot
- `properties` (List[str]): Properties to include
- `output` (str, optional): Output file path

**`stability_performance_plot(materials, output=None)`**

Plot stability vs. performance.

**`property_performance_plot(x, y, x_label, y_label, title, show_trend=True, show_confidence=True, output=None)`**

Create property-performance scatter plot.

---

## Reporting Module

### ReportGenerator

Generate comprehensive analysis reports.

```python
from ceramic_discovery.reporting import ReportGenerator

generator = ReportGenerator()
```

#### Methods

**`generate_summary_report(dataset, ml_results=None, top_candidates=None)`**

Generate summary report with dataset statistics and results.

Parameters:
- `dataset` (pd.DataFrame): Materials dataset
- `ml_results` (Dict, optional): ML model results
- `top_candidates` (List[Dict], optional): Top material candidates

Returns:
- `str`: Report text

Example:
```python
report = generator.generate_summary_report(
    dataset=combined_data,
    ml_results={"r2": 0.72, "mae": 45.2, "rmse": 62.1},
    top_candidates=top_10_materials
)

print(report)

# Save to file
with open("./results/summary_report.txt", "w") as f:
    f.write(report)
```

**`generate_experimental_protocol_report(candidates, synthesis_plans, characterization_plans, resource_estimates, output_dir)`**

Generate experimental protocol report.

Parameters:
- `candidates` (List[RankedMaterial]): Material candidates
- `synthesis_plans` (List[Dict]): Synthesis plans
- `characterization_plans` (List[Dict]): Characterization plans
- `resource_estimates` (List[Dict]): Resource estimates
- `output_dir` (str): Output directory

Returns:
- `str`: Report file path

Example:
```python
report_path = generator.generate_experimental_protocol_report(
    candidates=top_candidates,
    synthesis_plans=synthesis_plans,
    characterization_plans=char_plans,
    resource_estimates=estimates,
    output_dir="./results/protocols"
)

print(f"Protocol report saved to: {report_path}")
```

---

## Utilities

### Configuration

```python
from ceramic_discovery.config import Config

config = Config.load()
print(f"Database URL: {config.database_url}")
print(f"Max workers: {config.max_workers}")
```

### Logging

```python
from ceramic_discovery.utils.logging import setup_logging

logger = setup_logging(
    name="my_analysis",
    log_file="./logs/my_analysis.log",
    level="INFO"
)

logger.info("Starting analysis")
logger.warning("Unusual value detected")
logger.error("Processing failed")
```

### Data Conversion Utilities

Safe conversion utilities for heterogeneous data formats.

```python
from ceramic_discovery.utils.data_conversion import (
    safe_float,
    safe_int,
    extract_numeric_from_dict,
    extract_numeric_from_list
)
```

**`safe_float(value)`**

Safely convert any value to float.

Parameters:
- `value` (Any): Value to convert

Returns:
- `Optional[float]`: Float value or None

Example:
```python
# Handles various types
safe_float(42)           # 42.0
safe_float("3.14")       # 3.14
safe_float(None)         # None
safe_float({"value": 5}) # 5.0
safe_float([1, 2, 3])    # 2.0 (mean)
```

**`safe_int(value)`**

Safely convert any value to int.

Parameters:
- `value` (Any): Value to convert

Returns:
- `Optional[int]`: Integer value or None

**`extract_numeric_from_dict(data, keys=None)`**

Extract numeric value from dictionary.

Parameters:
- `data` (Dict): Dictionary to extract from
- `keys` (List[str], optional): Keys to try (default: ["value", "magnitude", "mean"])

Returns:
- `Optional[float]`: Extracted value or None

**`extract_numeric_from_list(data)`**

Extract numeric value from list (computes mean).

Parameters:
- `data` (List): List to extract from

Returns:
- `Optional[float]`: Mean of numeric elements or None

---

## Data Types

### StabilityResult

```python
@dataclass
class StabilityResult:
    material_id: str
    formula: str
    classification: StabilityClassification  # STABLE, METASTABLE, UNSTABLE
    energy_above_hull: float
    formation_energy_per_atom: float
    is_viable: bool
    confidence_score: float
```

### ValidationReport

```python
@dataclass
class ValidationReport:
    material_id: str
    is_valid: bool
    violations: List[ValidationViolation]
    
    def has_critical_violations(self) -> bool
    def get_violations_by_severity(self, severity: str) -> List[ValidationViolation]
    def summary(self) -> str
```

### PredictionWithUncertainty

```python
@dataclass
class PredictionWithUncertainty:
    value: float
    lower_bound: float
    upper_bound: float
    uncertainty: float
    reliability_score: float
```

---

## Error Handling

### Common Exceptions

```python
from ceramic_discovery.exceptions import (
    StabilityError,
    ValidationError,
    ModelNotFoundError,
    DatabaseError,
    APIError
)

try:
    result = analyzer.analyze_stability(...)
except StabilityError as e:
    logger.error(f"Stability analysis failed: {e}")
except ValidationError as e:
    logger.warning(f"Validation failed: {e}")
```

---

## Examples

### Complete Workflow Example

```python
from ceramic_discovery.dft import StabilityAnalyzer, MaterialsProjectClient
from ceramic_discovery.ml import ModelTrainer, FeatureEngineeringPipeline
from ceramic_discovery.ballistics import BallisticPredictor
from ceramic_discovery.validation import PhysicalPlausibilityValidator, ReproducibilityFramework

# Initialize components
stability_analyzer = StabilityAnalyzer()
validator = PhysicalPlausibilityValidator()
predictor = BallisticPredictor()
framework = ReproducibilityFramework(experiment_dir="./results", master_seed=42)

# Start experiment
framework.start_experiment(
    experiment_id="complete_workflow_001",
    experiment_type="screening",
    parameters={"base_system": "SiC"}
)

# Step 1: Collect DFT data
client = MaterialsProjectClient(api_key="your_key")
materials = client.search_materials("SiC")
framework.record_transformation("DFT collection", {"n_materials": len(materials)})

# Step 2: Stability screening
stability_results = stability_analyzer.batch_analyze(materials)
viable_materials = [m for m, r in zip(materials, stability_results) if r.is_viable]
framework.record_transformation("Stability filter", {"n_viable": len(viable_materials)})

# Step 3: Physical validation
validated_materials = []
for material in viable_materials:
    report = validator.validate_material(material["material_id"], material)
    if not report.has_critical_violations():
        validated_materials.append(material)
framework.record_transformation("Validation", {"n_validated": len(validated_materials)})

# Step 4: Ballistic prediction
predictor.load_model("./models/v50_predictor_v1.pkl")
predictions = []
for material in validated_materials:
    pred = predictor.predict_v50(material)
    predictions.append({
        "material_id": material["material_id"],
        "formula": material["formula"],
        "v50": pred["v50"],
        "confidence_interval": pred["confidence_interval"],
        "reliability": pred["reliability_score"]
    })

# Sort by performance
predictions.sort(key=lambda x: x["v50"], reverse=True)

# Finalize experiment
results = {
    "top_candidates": len(predictions[:10]),
    "best_v50": predictions[0]["v50"] if predictions else None
}
snapshot = framework.finalize_experiment(results)

print(f"Workflow complete. Top candidate: {predictions[0]['formula']}")
print(f"Predicted V₅₀: {predictions[0]['v50']:.0f} m/s")
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-23  
**For Questions:** See documentation or contact project team

