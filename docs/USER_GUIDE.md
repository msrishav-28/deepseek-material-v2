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
6. [Workflow 5: Multi-Source Data Integration](#workflow-5-multi-source-data-integration)
7. [Workflow 6: Application-Specific Material Ranking](#workflow-6-application-specific-material-ranking)
8. [Workflow 7: Experimental Planning](#workflow-7-experimental-planning)
9. [Advanced Topics](#advanced-topics)
10. [Troubleshooting](#troubleshooting)

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

## Workflow 5: Multi-Source Data Integration

### Objective
Load and combine data from multiple sources (Materials Project, JARVIS-DFT, NIST-JANAF, literature) to create a comprehensive materials database.

### Step 1: Load JARVIS-DFT Data

```python
from ceramic_discovery.dft import JarvisClient

# Initialize JARVIS client
client = JarvisClient(jarvis_file_path="./data/jdft_3d-7-7-2018.json")

# Load carbide materials
metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "W", "B"}
carbides = client.load_carbides(metal_elements=metal_elements)

print(f"Loaded {len(carbides)} carbide materials from JARVIS")

# Display sample
for material in carbides[:3]:
    print(f"  {material.formula}: E_f = {material.formation_energy_per_atom:.3f} eV/atom")
```

### Step 2: Load NIST-JANAF Data

```python
from ceramic_discovery.dft import NISTClient

# Initialize NIST client
nist_client = NISTClient()

# Load thermochemical data for SiC
sic_thermo = nist_client.load_thermochemical_data(
    filepath="./data/nist/SiC_thermochemical.txt",
    material_name="SiC"
)

# Interpolate property at specific temperature
cp_1000K = nist_client.interpolate_property(
    data=sic_thermo,
    temperature=1000.0,
    property_name="Cp"
)

print(f"SiC heat capacity at 1000K: {cp_1000K:.2f} J/(mol·K)")
```

### Step 3: Access Literature Database

```python
from ceramic_discovery.dft import LiteratureDatabase

# Initialize literature database
lit_db = LiteratureDatabase()

# Get available materials
available = lit_db.list_available_materials()
print(f"Literature data available for: {', '.join(available)}")

# Get property with uncertainty
hardness, uncertainty = lit_db.get_property_with_uncertainty(
    formula="SiC",
    property_name="hardness_vickers"
)

print(f"SiC hardness: {hardness:.0f} ± {uncertainty:.0f} HV")
```

### Step 4: Combine All Data Sources

```python
from ceramic_discovery.dft import DataCombiner, MaterialsProjectClient

# Load Materials Project data
mp_client = MaterialsProjectClient(api_key="your_api_key")
mp_data = mp_client.search_materials("SiC")

# Combine all sources
combiner = DataCombiner()
combined_data = combiner.combine_sources(
    mp_data=mp_data,
    jarvis_data=carbides,
    nist_data={"SiC": sic_thermo},
    literature_data=lit_db.get_all_data()
)

print(f"Combined dataset: {len(combined_data)} materials")
print(f"Average properties per material: {combined_data['property_count'].mean():.1f}")

# Save combined data
combined_data.to_csv("./results/combined_materials_database.csv", index=False)
```

### Expected Output

```
Multi-Source Data Integration Results
======================================

Data Sources Loaded:
  Materials Project: 150 materials
  JARVIS-DFT: 425 carbides
  NIST-JANAF: 8 materials with temperature-dependent data
  Literature: 12 materials with experimental validation

Combined Dataset:
  Total unique materials: 485
  Materials with multiple sources: 95
  Average data completeness: 78%

Top Materials by Data Completeness:
1. SiC: 95% (4 sources)
2. TiC: 88% (3 sources)
3. ZrC: 85% (3 sources)

Data saved to: ./results/combined_materials_database.csv
```

---

## Workflow 6: Application-Specific Material Ranking

### Objective
Rank materials for specific applications (aerospace, cutting tools, thermal barriers, etc.) based on property requirements.

### Step 1: Initialize Application Ranker

```python
from ceramic_discovery.screening import ApplicationRanker

# Initialize with default applications
ranker = ApplicationRanker()

# View available applications
applications = ranker.list_applications()
for app in applications:
    print(f"  - {app}")
```

### Step 2: Rank Materials for Single Application

```python
# Load materials
materials = combined_data.to_dict('records')

# Rank for aerospace hypersonic application
aerospace_ranking = ranker.rank_materials(
    materials=materials,
    application="aerospace_hypersonic"
)

# Display top candidates
print("\nTop 5 Candidates for Aerospace Hypersonic:")
for i, ranked in enumerate(aerospace_ranking[:5], 1):
    print(f"{i}. {ranked.formula}")
    print(f"   Overall Score: {ranked.overall_score:.3f}")
    print(f"   Hardness Score: {ranked.component_scores.get('hardness', 0):.3f}")
    print(f"   Thermal Cond Score: {ranked.component_scores.get('thermal_cond', 0):.3f}")
    print(f"   Confidence: {ranked.confidence:.2f}")
    print()
```

### Step 3: Rank for All Applications

```python
# Batch ranking for all applications
all_rankings = ranker.rank_for_all_applications(materials=materials)

# Display summary
print("\nApplication Ranking Summary:")
print(all_rankings.groupby('application')['overall_score'].describe())

# Find materials that rank highly across multiple applications
multi_app_leaders = all_rankings[all_rankings['rank'] <= 10].groupby('formula').size()
multi_app_leaders = multi_app_leaders.sort_values(ascending=False)

print("\nMaterials ranking in top 10 for multiple applications:")
for formula, count in multi_app_leaders.head(5).items():
    print(f"  {formula}: Top 10 in {count} applications")
```

### Step 4: Visualize Rankings

```python
from ceramic_discovery.analysis import Visualizer

viz = Visualizer()

# Create ranking comparison plot
viz.application_ranking_plot(
    rankings=all_rankings,
    top_n=10,
    output="./results/application_rankings.png"
)

# Create radar chart for top material
top_material = aerospace_ranking[0]
viz.property_radar_chart(
    material=top_material,
    applications=["aerospace_hypersonic", "cutting_tools", "thermal_barriers"],
    output=f"./results/{top_material.formula}_radar.png"
)
```

### Expected Output

```
Application-Specific Ranking Results
====================================

Top Candidates by Application:

Aerospace Hypersonic:
1. HfC: Score 0.892 (Confidence: 0.95)
2. ZrC: Score 0.875 (Confidence: 0.93)
3. TiC: Score 0.848 (Confidence: 0.91)

Cutting Tools:
1. TiC: Score 0.915 (Confidence: 0.94)
2. WC: Score 0.901 (Confidence: 0.96)
3. B4C: Score 0.887 (Confidence: 0.92)

Thermal Barriers:
1. ZrC: Score 0.856 (Confidence: 0.89)
2. HfC: Score 0.842 (Confidence: 0.91)
3. SiC: Score 0.825 (Confidence: 0.95)

Multi-Application Leaders:
  HfC: Top 10 in 4 applications
  ZrC: Top 10 in 4 applications
  TiC: Top 10 in 3 applications
```

---

## Workflow 7: Experimental Planning

### Objective
Generate experimental design recommendations for synthesis and characterization of top material candidates.

### Step 1: Select Candidates for Experimental Validation

```python
from ceramic_discovery.validation import ExperimentalPlanner

# Initialize planner
planner = ExperimentalPlanner()

# Select top candidates from ranking
top_candidates = aerospace_ranking[:5]

print(f"Planning experiments for {len(top_candidates)} candidates")
```

### Step 2: Design Synthesis Protocols

```python
# Generate synthesis recommendations for each candidate
synthesis_plans = []

for candidate in top_candidates:
    # Design synthesis protocol
    methods = planner.design_synthesis_protocol(
        formula=candidate.formula,
        properties=candidate.properties
    )
    
    synthesis_plans.append({
        'formula': candidate.formula,
        'methods': methods
    })
    
    print(f"\nSynthesis Methods for {candidate.formula}:")
    for method in methods:
        print(f"  - {method.method}")
        print(f"    Temperature: {method.temperature_K:.0f} K")
        print(f"    Pressure: {method.pressure_MPa:.0f} MPa")
        print(f"    Time: {method.time_hours:.1f} hours")
        print(f"    Difficulty: {method.difficulty}")
        print(f"    Cost Factor: {method.cost_factor:.1f}x")
```

### Step 3: Design Characterization Plans

```python
# Design characterization plan
target_properties = [
    "hardness",
    "fracture_toughness",
    "thermal_conductivity",
    "density",
    "phase_composition"
]

characterization_plans = []

for candidate in top_candidates:
    techniques = planner.design_characterization_plan(
        formula=candidate.formula,
        target_properties=target_properties
    )
    
    characterization_plans.append({
        'formula': candidate.formula,
        'techniques': techniques
    })
    
    print(f"\nCharacterization Plan for {candidate.formula}:")
    for tech in techniques:
        print(f"  - {tech.technique}: {tech.purpose}")
        print(f"    Time: {tech.estimated_time_hours:.1f} hours")
        print(f"    Cost: ${tech.estimated_cost_dollars:.0f}")
```

### Step 4: Estimate Resources

```python
# Estimate resources for each candidate
resource_estimates = []

for i, candidate in enumerate(top_candidates):
    estimate = planner.estimate_resources(
        synthesis_methods=synthesis_plans[i]['methods'],
        characterization_plan=characterization_plans[i]['techniques']
    )
    
    resource_estimates.append({
        'formula': candidate.formula,
        'estimate': estimate
    })
    
    print(f"\nResource Estimate for {candidate.formula}:")
    print(f"  Timeline: {estimate.timeline_months:.1f} months")
    print(f"  Estimated Cost: ${estimate.estimated_cost_k_dollars:.1f}K")
    print(f"  Required Equipment: {', '.join(estimate.required_equipment[:3])}")
    print(f"  Required Expertise: {', '.join(estimate.required_expertise)}")
    print(f"  Confidence: {estimate.confidence_level:.2f}")
```

### Step 5: Generate Experimental Protocol Document

```python
from ceramic_discovery.reporting import ReportGenerator

# Generate comprehensive experimental protocol
generator = ReportGenerator()

protocol_report = generator.generate_experimental_protocol_report(
    candidates=top_candidates,
    synthesis_plans=synthesis_plans,
    characterization_plans=characterization_plans,
    resource_estimates=resource_estimates,
    output_dir="./results/experimental_protocols"
)

print(f"\nExperimental protocol report generated: {protocol_report}")
```

### Expected Output

```
Experimental Planning Results
=============================

Candidate: HfC

Synthesis Methods:
1. Hot Pressing
   - Temperature: 2273 K (2000°C)
   - Pressure: 30 MPa
   - Time: 2.0 hours
   - Difficulty: Medium
   - Cost Factor: 1.5x

2. Spark Plasma Sintering
   - Temperature: 2073 K (1800°C)
   - Pressure: 50 MPa
   - Time: 0.5 hours
   - Difficulty: Hard
   - Cost Factor: 2.5x

Characterization Plan:
- XRD: Phase identification and purity (4 hours, $500)
- SEM: Microstructure analysis (6 hours, $800)
- Hardness Testing: Vickers hardness (2 hours, $200)
- Thermal Conductivity: Laser flash method (8 hours, $1,500)
- Density: Archimedes method (1 hour, $100)

Resource Estimate:
- Timeline: 4.5 months
- Estimated Cost: $18.5K
- Required Equipment: Hot press, XRD, SEM, Hardness tester
- Required Expertise: Ceramic processing, Materials characterization
- Risk Factors: High temperature processing, Hafnium precursor cost
- Confidence: 0.85

Total Program Estimate (5 candidates):
- Timeline: 6-8 months (parallel processing)
- Total Cost: $75-90K
- Success Probability: 80%

Protocol documents saved to: ./results/experimental_protocols/
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

