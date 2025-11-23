# User Guide - Research Workflows

## Ceramic Armor Discovery Framework

**Version:** 1.0  
**Date:** 2025-11-23  
**Requirements:** 7.3, 7.4, 7.5

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Workflow 1: Dopant Screening](#workflow-1-dopant-screening)
3. [Workflow 2: Property Prediction](#workflow-2-property-prediction)
4. [Workflow 3: Ballistic Performance Analysis](#workflow-3-ballistic-performance-analysis)
5. [Workflow 4: Mechanism Investigation](#workflow-4-mechanism-investigation)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

Before starting, ensure you have:
- ✅ Installed the framework (see README.md)
- ✅ Configured environment variables (.env file)
- ✅ Materials Project API key
- ✅ Database initialized
- ✅ Test data available

### Quick Health Check

```bash
# Check system health
ceramic-discovery health

# Expected output:
# ✓ Database connection: OK
# ✓ API credentials: OK
# ✓ Dependencies: OK
# ✓ Data directories: OK
```

### Configuration

Edit `.env` file with your settings:

```bash
# Materials Project API
MATERIALS_PROJECT_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ceramic_discovery

# Computational Resources
MAX_WORKERS=4
MEMORY_LIMIT_GB=16

# Output Directories
DATA_DIR=./data
RESULTS_DIR=./results
```

---

## Workflow 1: Dopant Screening

### Objective
Screen dopant candidates for a ceramic system and identify promising materials for synthesis.

### Step 1: Define Screening Parameters

Create a configuration file `screening_config.yaml`:

```yaml
# Screening Configuration
base_system: "SiC"
dopant_elements:
  - "B"
  - "Al"
  - "Ti"
  - "Zr"
  - "Hf"

dopant_concentrations:
  - 0.01  # 1%
  - 0.02  # 2%
  - 0.03  # 3%
  - 0.05  # 5%
  - 0.10  # 10%

stability_threshold: 0.1  # eV/atom
min_hardness: 25.0  # GPa
min_fracture_toughness: 4.0  # MPa·m^0.5

output_dir: "./results/sic_dopant_screening"
```

### Step 2: Run Screening

```bash
# Command-line interface
ceramic-discovery screen --config screening_config.yaml

# Or use Python API
```

```python
from ceramic_discovery.screening import ScreeningEngine, ScreeningConfig
from ceramic_discovery.dft import StabilityAnalyzer

# Initialize components
config = ScreeningConfig.from_yaml("screening_config.yaml")
analyzer = StabilityAnalyzer()
engine = ScreeningEngine(stability_analyzer=analyzer)

# Run screening
results = engine.run_screening(config)

# Save results
results.save("./results/sic_dopant_screening/results.json")
```

### Step 3: Analyze Results

```python
from ceramic_discovery.analysis import PropertyAnalyzer

# Load results
results = ScreeningResults.load("./results/sic_dopant_screening/results.json")

# Get top candidates
top_10 = results.get_top_candidates(n=10)

# Print summary
for i, candidate in enumerate(top_10, 1):
    print(f"{i}. {candidate.formula}")
    print(f"   ΔE_hull: {candidate.energy_above_hull:.3f} eV/atom")
    print(f"   Hardness: {candidate.hardness:.1f} GPa")
    print(f"   K_IC: {candidate.fracture_toughness:.1f} MPa·m^0.5")
    print(f"   Confidence: {candidate.confidence_score:.2f}")
    print()
```

### Step 4: Visualize Results

```python
from ceramic_discovery.analysis import Visualizer

viz = Visualizer()

# Create parallel coordinates plot
viz.parallel_coordinates(
    results.viable_materials,
    properties=["hardness", "fracture_toughness", "density", "thermal_conductivity"],
    output="./results/sic_dopant_screening/parallel_coords.png"
)

# Create stability vs. performance plot
viz.stability_performance_plot(
    results.all_materials,
    output="./results/sic_dopant_screening/stability_performance.png"
)
```

### Expected Output

```
Screening Results Summary
=========================
Total materials screened: 25
Viable materials (ΔE_hull ≤ 0.1): 18
Top candidates identified: 10

Top 3 Candidates:
1. Si0.98B0.02C - ΔE_hull: 0.02 eV/atom, Hardness: 29.5 GPa
2. Si0.95Al0.05C - ΔE_hull: 0.05 eV/atom, Hardness: 28.8 GPa
3. Si0.97Ti0.03C - ΔE_hull: 0.08 eV/atom, Hardness: 29.2 GPa

Results saved to: ./results/sic_dopant_screening/
```

---

## Workflow 2: Property Prediction

### Objective
Predict properties for a new material composition using ML models.

### Step 1: Prepare Input Data

```python
from ceramic_discovery.ceramics import CeramicSystemFactory

# Create material
material = {
    "formula": "Si0.98B0.02C",
    "energy_above_hull": 0.02,
    "formation_energy_per_atom": -0.63,
    "band_gap": 2.4,
    "density": 3.19,
}
```

### Step 2: Load ML Model

```python
from ceramic_discovery.ml import ModelTrainer

# Load pre-trained model
trainer = ModelTrainer()
trainer.load_model("./models/v50_predictor_v1.pkl")
```

### Step 3: Make Predictions

```python
from ceramic_discovery.ml import FeatureEngineeringPipeline

# Extract features
engineer = FeatureEngineeringPipeline()
features = engineer.extract_features(material)

# Predict with uncertainty
prediction = trainer.predict_with_uncertainty(features)

print(f"Predicted V₅₀: {prediction.value:.0f} m/s")
print(f"95% CI: [{prediction.lower_bound:.0f}, {prediction.upper_bound:.0f}] m/s")
print(f"Reliability: {prediction.reliability_score:.2f}")
```

### Step 3: Validate Prediction

```python
from ceramic_discovery.validation import PhysicalPlausibilityValidator

# Validate physical plausibility
validator = PhysicalPlausibilityValidator()
report = validator.validate_material(material["formula"], material)

if report.is_valid:
    print("✓ Prediction passes physical plausibility checks")
else:
    print("⚠ Warning: Prediction may be unreliable")
    for violation in report.violations:
        print(f"  - {violation.message}")
```

### Expected Output

```
Predicted V₅₀: 875 m/s
95% CI: [805, 945] m/s
Reliability: 0.87

✓ Prediction passes physical plausibility checks

Feature Importance:
1. Hardness: 0.35
2. Fracture Toughness: 0.28
3. Thermal Conductivity: 0.18
4. Density: 0.12
5. Young's Modulus: 0.07
```

---

## Workflow 3: Ballistic Performance Analysis

### Objective
Analyze ballistic performance predictions and identify key performance drivers.

### Step 1: Load Materials Database

```python
from ceramic_discovery.dft import DatabaseManager

# Connect to database
db = DatabaseManager()

# Query materials
materials = db.query_materials(
    base_system="SiC",
    stability_threshold=0.1,
    min_hardness=25.0
)

print(f"Loaded {len(materials)} materials from database")
```

### Step 2: Predict Ballistic Performance

```python
from ceramic_discovery.ballistics import BallisticPredictor

predictor = BallisticPredictor()
predictor.load_model("./models/v50_predictor_v1.pkl")

# Batch prediction
predictions = []
for material in materials:
    pred = predictor.predict_v50(material.properties)
    predictions.append({
        "formula": material.formula,
        "v50": pred["v50"],
        "confidence_interval": pred["confidence_interval"],
        "reliability": pred["reliability_score"]
    })

# Sort by predicted performance
predictions.sort(key=lambda x: x["v50"], reverse=True)
```

### Step 3: Analyze Mechanisms

```python
from ceramic_discovery.ballistics import MechanismAnalyzer

analyzer = MechanismAnalyzer()

# Analyze thermal conductivity impact
thermal_analysis = analyzer.analyze_thermal_conductivity_impact(
    materials=[m.properties for m in materials],
    v50_values=[p["v50"] for p in predictions]
)

print(f"Thermal conductivity correlation: {thermal_analysis['correlation']:.3f}")
print(f"Impact on V₅₀: {thermal_analysis['impact_magnitude']:.1f} m/s per W/(m·K)")

# Analyze property contributions
contributions = analyzer.analyze_property_contributions(
    materials=[m.properties for m in materials],
    v50_values=[p["v50"] for p in predictions]
)

print("\nProperty Contributions to V₅₀:")
for prop, contribution in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {prop}: {contribution:.2f}")
```

### Step 4: Generate Report

```python
from ceramic_discovery.reporting import ReportGenerator

generator = ReportGenerator()

report = generator.generate_ballistic_analysis_report(
    materials=materials,
    predictions=predictions,
    mechanism_analysis=thermal_analysis,
    output_dir="./results/ballistic_analysis"
)

print(f"Report generated: {report.filepath}")
```

### Expected Output

```
Ballistic Performance Analysis
==============================

Top 5 Materials by Predicted V₅₀:
1. Si0.98B0.02C: 895 ± 65 m/s (reliability: 0.89)
2. Si0.95Al0.05C: 880 ± 70 m/s (reliability: 0.85)
3. Si0.97Ti0.03C: 870 ± 75 m/s (reliability: 0.82)
4. SiC (baseline): 850 ± 60 m/s (reliability: 0.92)
5. Si0.90Zr0.10C: 845 ± 80 m/s (reliability: 0.78)

Thermal conductivity correlation: 0.456
Impact on V₅₀: 2.3 m/s per W/(m·K)

Property Contributions to V₅₀:
  hardness: 0.35
  fracture_toughness: 0.28
  thermal_conductivity_1000C: 0.18
  density: 0.12
  youngs_modulus: 0.07

Report saved to: ./results/ballistic_analysis/report.pdf
```

---

## Workflow 4: Mechanism Investigation

### Objective
Investigate physical mechanisms underlying property-performance relationships.

### Step 1: Hypothesis Definition

```python
# Hypothesis: Higher thermal conductivity improves ballistic performance
# by dissipating impact energy more effectively

hypothesis = {
    "name": "Thermal Dissipation Mechanism",
    "independent_variable": "thermal_conductivity_1000C",
    "dependent_variable": "v50",
    "expected_correlation": "positive",
    "physical_basis": "Energy dissipation during impact"
}
```

### Step 2: Data Collection

```python
from ceramic_discovery.dft import PropertyExtractor

# Extract relevant properties
extractor = PropertyExtractor()

data = []
for material in materials:
    data.append({
        "formula": material.formula,
        "thermal_conductivity": material.thermal.thermal_conductivity_1000C,
        "hardness": material.mechanical.hardness,
        "fracture_toughness": material.mechanical.fracture_toughness,
        "v50": predictor.predict_v50(material.properties)["v50"]
    })
```

### Step 3: Statistical Analysis

```python
from ceramic_discovery.ballistics import MechanismAnalyzer
import numpy as np
from scipy import stats

analyzer = MechanismAnalyzer()

# Correlation analysis
k_values = [d["thermal_conductivity"] for d in data]
v50_values = [d["v50"] for d in data]

correlation, p_value = stats.pearsonr(k_values, v50_values)

print(f"Correlation coefficient: {correlation:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

# Partial correlation (controlling for hardness)
hardness_values = [d["hardness"] for d in data]
partial_corr = analyzer.partial_correlation(
    x=k_values,
    y=v50_values,
    control=hardness_values
)

print(f"Partial correlation (controlling for hardness): {partial_corr:.3f}")
```

### Step 4: Visualization

```python
from ceramic_discovery.analysis import Visualizer

viz = Visualizer()

# Scatter plot with trend line
viz.property_performance_plot(
    x=k_values,
    y=v50_values,
    x_label="Thermal Conductivity at 1000°C (W/(m·K))",
    y_label="V₅₀ (m/s)",
    title="Thermal Conductivity vs. Ballistic Performance",
    show_trend=True,
    show_confidence=True,
    output="./results/mechanism_analysis/thermal_k_vs_v50.png"
)

# Multi-property correlation matrix
viz.correlation_matrix(
    data=data,
    properties=["thermal_conductivity", "hardness", "fracture_toughness", "v50"],
    output="./results/mechanism_analysis/correlation_matrix.png"
)
```

### Expected Output

```
Mechanism Investigation: Thermal Dissipation
============================================

Correlation Analysis:
  Pearson correlation: 0.456
  P-value: 0.0023
  Significant: Yes

Partial Correlation (controlling for hardness):
  Partial correlation: 0.312
  Interpretation: Thermal conductivity has independent effect

Conclusion:
  ✓ Hypothesis supported by data
  ✓ Thermal conductivity positively correlates with V₅₀
  ✓ Effect persists after controlling for hardness
  ✓ Estimated impact: 2.3 m/s per W/(m·K)

Recommendation:
  Prioritize dopants that maintain or enhance thermal conductivity
```

---

## Advanced Topics

### Reproducibility and Provenance

```python
from ceramic_discovery.validation import ReproducibilityFramework

# Initialize framework
framework = ReproducibilityFramework(
    experiment_dir="./results/experiments",
    master_seed=42
)

# Start experiment
framework.start_experiment(
    experiment_id="sic_screening_001",
    experiment_type="dopant_screening",
    parameters={
        "base_system": "SiC",
        "stability_threshold": 0.1,
        "n_dopants": 5
    }
)

# Record workflow steps
framework.record_transformation("DFT data collection", {"n_materials": 25})
framework.record_transformation("Stability filtering", {"n_viable": 18})
framework.record_transformation("ML prediction", {"model_r2": 0.72})

# Finalize with results
results = {"top_candidates": 10, "avg_v50": 875}
snapshot = framework.finalize_experiment(results)

# Export reproducibility package
framework.export_reproducibility_package(
    "sic_screening_001",
    "./results/reproducibility_packages/sic_screening_001"
)
```

### Batch Processing

```python
from ceramic_discovery.screening import WorkflowOrchestrator

# Initialize orchestrator
orchestrator = WorkflowOrchestrator(output_dir="./results/batch_screening")

# Define batch workflow
materials_batch = [
    {"base": "SiC", "dopants": ["B", "Al", "Ti"]},
    {"base": "B4C", "dopants": ["Si", "Al", "Zr"]},
    {"base": "WC", "dopants": ["Ti", "Ta", "Nb"]},
]

# Run batch
for config in materials_batch:
    workflow_id = f"{config['base']}_screening"
    orchestrator.submit_workflow(
        workflow_id=workflow_id,
        workflow_function=run_screening,
        parameters=config
    )

# Monitor progress
status = orchestrator.get_workflow_status()
print(f"Completed: {status['completed']}/{status['total']}")
```

### HPC Integration

```bash
# Submit to SLURM cluster
ceramic-discovery submit-slurm \
    --config screening_config.yaml \
    --nodes 4 \
    --time 24:00:00 \
    --output ./results/hpc_screening

# Monitor job
ceramic-discovery job-status --job-id 12345

# Retrieve results
ceramic-discovery retrieve-results --job-id 12345
```

---

## Troubleshooting

### Common Issues

#### Issue: API Rate Limiting

**Symptom:** `RateLimitError: Materials Project API rate limit exceeded`

**Solution:**
```python
from ceramic_discovery.dft import MaterialsProjectClient

# Configure rate limiting
client = MaterialsProjectClient(
    api_key="your_key",
    requests_per_second=5,  # Reduce rate
    retry_on_rate_limit=True
)
```

#### Issue: Memory Errors

**Symptom:** `MemoryError: Unable to allocate array`

**Solution:**
```python
from ceramic_discovery.performance import MemoryManager

# Enable memory management
manager = MemoryManager(max_memory_gb=8)
manager.enable_chunking()

# Process in batches
for batch in manager.batch_iterator(materials, batch_size=100):
    results = process_batch(batch)
```

#### Issue: Model Not Found

**Symptom:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Download pre-trained models
ceramic-discovery download-models --version latest

# Or train new model
ceramic-discovery train-model \
    --data ./data/training_data.csv \
    --output ./models/custom_model.pkl
```

#### Issue: Database Connection

**Symptom:** `DatabaseError: Could not connect to database`

**Solution:**
```bash
# Check database status
ceramic-discovery db-status

# Reinitialize database
ceramic-discovery db-init --reset

# Test connection
ceramic-discovery db-test
```

### Getting Help

- **Documentation:** `docs/` directory
- **Examples:** `examples/` directory
- **Notebooks:** `notebooks/` directory
- **Issues:** GitHub Issues
- **Email:** [project-email]

---

## Best Practices

### 1. Always Validate Inputs

```python
from ceramic_discovery.validation import PhysicalPlausibilityValidator

validator = PhysicalPlausibilityValidator()
report = validator.validate_material(material_id, properties)

if not report.is_valid:
    print("⚠ Warning: Input validation failed")
    for violation in report.violations:
        print(f"  - {violation.message}")
```

### 2. Use Reproducibility Framework

```python
# Always use reproducibility framework for research
framework = ReproducibilityFramework(experiment_dir="./results", master_seed=42)
framework.start_experiment(...)
# ... your workflow ...
framework.finalize_experiment(results)
```

### 3. Check Uncertainty

```python
# Always report predictions with uncertainty
prediction = predictor.predict_with_uncertainty(features)
print(f"V₅₀: {prediction.value:.0f} ± {prediction.uncertainty:.0f} m/s")
print(f"Reliability: {prediction.reliability_score:.2f}")
```

### 4. Save Intermediate Results

```python
# Save checkpoints during long workflows
checkpoint_manager = CheckpointManager("./results/checkpoints")
checkpoint_manager.save("step_1_dft_collection", dft_results)
checkpoint_manager.save("step_2_stability_filter", viable_materials)
checkpoint_manager.save("step_3_ml_predictions", predictions)
```

### 5. Document Assumptions

```python
# Document assumptions in your analysis
assumptions = {
    "temperature": "Room temperature (25°C)",
    "threat": "7.62mm AP",
    "microstructure": "Fully dense, fine-grained",
    "processing": "Hot-pressed",
}

# Include in report
report.add_assumptions(assumptions)
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-23  
**For Questions:** See documentation or contact project team

