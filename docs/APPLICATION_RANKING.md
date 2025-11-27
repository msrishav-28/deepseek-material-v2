# Application-Specific Material Ranking Guide

## Overview

This guide explains how to use the application-specific material ranking system in the Ceramic Armor Discovery Framework. The ranking system evaluates materials against specific use-case requirements for applications like aerospace, cutting tools, thermal barriers, and more.

**Version:** 1.0  
**Date:** 2025-11-27  
**Requirements:** 3.1, 3.2, 3.3, 3.4, 3.5, 9.4

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Predefined Applications](#predefined-applications)
4. [Ranking Materials](#ranking-materials)
5. [Understanding Scores](#understanding-scores)
6. [Custom Applications](#custom-applications)
7. [Advanced Usage](#advanced-usage)
8. [Best Practices](#best-practices)

---

## Introduction

### What is Application-Specific Ranking?

Application-specific ranking evaluates materials based on how well their properties match the requirements of a specific use case. Unlike general property screening, this approach:

- **Weights properties** based on application importance
- **Scores within target ranges** rather than simple thresholds
- **Handles missing data** gracefully with partial scoring
- **Provides confidence metrics** based on data completeness

### When to Use Application Ranking

Use application ranking when you need to:

- Identify optimal materials for a specific application
- Compare materials across multiple applications
- Prioritize candidates for experimental validation
- Understand property trade-offs for different use cases

---

## Getting Started

### Basic Usage

```python
from ceramic_discovery.screening import ApplicationRanker

# Initialize ranker with default applications
ranker = ApplicationRanker()

# Load your materials
materials = [
    {
        'material_id': 'mat1',
        'formula': 'SiC',
        'hardness': 28.0,
        'thermal_conductivity_1000C': 45.0,
        'melting_point': 3100,
        'formation_energy_per_atom': -0.65,
        'bulk_modulus': 220.0,
        'band_gap': 2.4,
        'density': 3.21
    },
    # ... more materials
]

# Rank for aerospace application
aerospace_ranking = ranker.rank_materials(
    materials=materials,
    application="aerospace_hypersonic"
)

# Display top 5
for i, ranked in enumerate(aerospace_ranking[:5], 1):
    print(f"{i}. {ranked.formula}: Score {ranked.overall_score:.3f}")
```

### Quick Example

```python
from ceramic_discovery.dft import JarvisClient, DataCombiner
from ceramic_discovery.screening import ApplicationRanker

# Load materials
client = JarvisClient("./data/jdft_3d-7-7-2018.json")
carbides = client.load_carbides(metal_elements={"Si", "Ti", "Zr", "Hf"})

# Convert to dict format
materials = [
    {
        'material_id': m.material_id,
        'formula': m.formula,
        'hardness': m.hardness,
        'thermal_conductivity_1000C': m.thermal_conductivity,
        'melting_point': m.melting_point,
        'formation_energy_per_atom': m.formation_energy_per_atom,
        'bulk_modulus': m.bulk_modulus,
        'band_gap': m.band_gap,
        'density': m.density
    }
    for m in carbides
]

# Rank
ranker = ApplicationRanker()
results = ranker.rank_materials(materials, "aerospace_hypersonic")

print(f"Top candidate: {results[0].formula} (Score: {results[0].overall_score:.3f})")
```

---

## Predefined Applications

The framework includes five predefined applications with optimized property requirements:

### 1. Aerospace Hypersonic

**Target Use:** Hypersonic vehicle leading edges, turbine components

**Key Requirements:**
- Hardness: 28-35 GPa (weight: 0.30)
- Thermal Conductivity: 50-150 W/(m·K) (weight: 0.25)
- Melting Point: 2500-4000 K (weight: 0.25)
- Formation Energy: -1.0 to -0.3 eV/atom (weight: 0.20)

**Rationale:** Requires extreme hardness for erosion resistance, high thermal conductivity for heat dissipation, and high melting point for thermal stability.

```python
aerospace_ranking = ranker.rank_materials(materials, "aerospace_hypersonic")
```

### 2. Cutting Tools

**Target Use:** High-speed machining, wear-resistant tooling

**Key Requirements:**
- Hardness: 30-40 GPa (weight: 0.50)
- Formation Energy: -4.0 to -2.0 eV/atom (weight: 0.30)
- Bulk Modulus: 300-500 GPa (weight: 0.20)

**Rationale:** Prioritizes extreme hardness for cutting performance and stability for long tool life.

```python
cutting_ranking = ranker.rank_materials(materials, "cutting_tools")
```

### 3. Thermal Barriers

**Target Use:** Thermal protection systems, insulation

**Key Requirements:**
- Thermal Conductivity: 20-60 W/(m·K) (weight: 0.40) - Lower is better
- Density: 1-6 g/cm³ (weight: 0.30) - Lower is better
- Melting Point: 2000-3500 K (weight: 0.30)

**Rationale:** Requires low thermal conductivity for insulation, low density for weight savings, and high melting point for thermal stability.

```python
thermal_ranking = ranker.rank_materials(materials, "thermal_barriers")
```

### 4. Wear-Resistant Coatings

**Target Use:** Protective coatings, tribological applications

**Key Requirements:**
- Hardness: 25-35 GPa (weight: 0.40)
- Formation Energy: -2.0 to -0.5 eV/atom (weight: 0.30)
- Density: 2-8 g/cm³ (weight: 0.30)

**Rationale:** Balances hardness for wear resistance with stability and reasonable density.

```python
wear_ranking = ranker.rank_materials(materials, "wear_resistant_coatings")
```

### 5. Electronic/Semiconductor

**Target Use:** High-temperature electronics, power devices

**Key Requirements:**
- Band Gap: 2.0-4.0 eV (weight: 0.40)
- Thermal Conductivity: 100-300 W/(m·K) (weight: 0.35)
- Formation Energy: -2.0 to -0.5 eV/atom (weight: 0.25)

**Rationale:** Requires appropriate band gap for semiconducting behavior and high thermal conductivity for heat management.

```python
electronic_ranking = ranker.rank_materials(materials, "electronic_semiconductor")
```

---

## Ranking Materials

### Single Application Ranking

```python
# Rank for one application
results = ranker.rank_materials(
    materials=materials,
    application="aerospace_hypersonic"
)

# Access ranking details
for ranked in results[:10]:
    print(f"\n{ranked.rank}. {ranked.formula}")
    print(f"   Overall Score: {ranked.overall_score:.3f}")
    print(f"   Confidence: {ranked.confidence:.2f}")
    print(f"   Component Scores:")
    for prop, score in ranked.component_scores.items():
        print(f"     {prop}: {score:.3f}")
```

### Multi-Application Ranking

```python
# Rank for all applications at once
all_rankings = ranker.rank_for_all_applications(materials=materials)

# Result is a DataFrame
print(all_rankings.head())

# Find best material for each application
best_by_app = all_rankings.loc[
    all_rankings.groupby('application')['overall_score'].idxmax()
]

print("\nBest Material by Application:")
for _, row in best_by_app.iterrows():
    print(f"{row['application']}: {row['formula']} (Score: {row['overall_score']:.3f})")
```

### Filtering Results

```python
# Get only high-confidence results
high_confidence = [
    r for r in results
    if r.confidence >= 0.8
]

# Get materials scoring above threshold
high_scoring = [
    r for r in results
    if r.overall_score >= 0.7
]

# Get top N for each application
all_rankings = ranker.rank_for_all_applications(materials)
top_10_per_app = all_rankings[all_rankings['rank'] <= 10]
```

---

## Understanding Scores

### Overall Score Calculation

The overall score is a weighted average of component scores:

```
overall_score = Σ(weight_i × component_score_i) / Σ(weight_i)
```

Where:
- `weight_i` is the importance weight for property i
- `component_score_i` is the score for property i (0-1 range)

### Component Score Calculation

For each property, the component score is calculated based on target range:

**Inside Target Range:**
```
score = 1.0
```

**Outside Target Range:**
```
score = max(0, 1.0 - distance_from_range / threshold)
```

### Confidence Score

Confidence indicates data completeness:

```
confidence = (available_properties / total_required_properties)
```

Example:
- Application requires 5 properties
- Material has 4 properties available
- Confidence = 4/5 = 0.80

### Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Excellent match - ideal candidate |
| 0.75 - 0.89 | Good match - strong candidate |
| 0.60 - 0.74 | Moderate match - consider with caution |
| 0.40 - 0.59 | Poor match - likely unsuitable |
| 0.00 - 0.39 | Very poor match - unsuitable |

| Confidence | Interpretation |
|------------|----------------|
| 0.90 - 1.00 | High confidence - complete data |
| 0.70 - 0.89 | Moderate confidence - mostly complete |
| 0.50 - 0.69 | Low confidence - significant gaps |
| < 0.50 | Very low confidence - unreliable |

---

## Custom Applications

### Defining Custom Applications

```python
from ceramic_discovery.screening import ApplicationRanker, ApplicationSpec

# Define custom application
custom_app = ApplicationSpec(
    name="nuclear_fuel_cladding",
    description="Nuclear fuel cladding material",
    
    # Target ranges (min, max)
    target_hardness=(20.0, 30.0),
    target_thermal_cond=(50.0, 150.0),
    target_melting_point=(2500.0, 3500.0),
    target_density=(4.0, 8.0),
    
    # Weights (must sum to reasonable total)
    weight_hardness=0.25,
    weight_thermal_cond=0.30,
    weight_melting_point=0.30,
    weight_density=0.15
)

# Create ranker with custom application
applications = {
    "nuclear_fuel_cladding": custom_app
}

ranker = ApplicationRanker(applications=applications)

# Use custom application
results = ranker.rank_materials(materials, "nuclear_fuel_cladding")
```

### Modifying Existing Applications

```python
# Get default applications
ranker = ApplicationRanker()
default_apps = ranker.applications.copy()

# Modify aerospace application
aerospace_spec = default_apps["aerospace_hypersonic"]
aerospace_spec.weight_hardness = 0.40  # Increase hardness importance
aerospace_spec.weight_thermal_cond = 0.20  # Decrease thermal conductivity importance

# Create new ranker with modified specs
custom_ranker = ApplicationRanker(applications=default_apps)
```

---

## Advanced Usage

### Batch Processing

```python
from concurrent.futures import ProcessPoolExecutor

def rank_batch(materials_batch, application):
    """Rank a batch of materials."""
    ranker = ApplicationRanker()
    return ranker.rank_materials(materials_batch, application)

# Split materials into batches
batch_size = 1000
batches = [materials[i:i+batch_size] for i in range(0, len(materials), batch_size)]

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(rank_batch, batch, "aerospace_hypersonic")
        for batch in batches
    ]
    
    results = []
    for future in futures:
        results.extend(future.result())

# Sort combined results
results.sort(key=lambda x: x.overall_score, reverse=True)
```

### Sensitivity Analysis

```python
# Analyze sensitivity to property weights
import numpy as np

base_ranker = ApplicationRanker()
base_results = base_ranker.rank_materials(materials, "aerospace_hypersonic")
base_top_10 = [r.formula for r in base_results[:10]]

# Vary hardness weight
hardness_weights = np.linspace(0.1, 0.5, 5)
sensitivity = []

for weight in hardness_weights:
    # Create modified application
    app_spec = base_ranker.applications["aerospace_hypersonic"]
    app_spec.weight_hardness = weight
    
    # Rank with modified weight
    ranker = ApplicationRanker(applications={"aerospace_hypersonic": app_spec})
    results = ranker.rank_materials(materials, "aerospace_hypersonic")
    top_10 = [r.formula for r in results[:10]]
    
    # Calculate overlap with base ranking
    overlap = len(set(base_top_10) & set(top_10))
    sensitivity.append({
        'hardness_weight': weight,
        'top_10_overlap': overlap
    })

print("Sensitivity to Hardness Weight:")
for s in sensitivity:
    print(f"  Weight {s['hardness_weight']:.2f}: {s['top_10_overlap']}/10 overlap")
```

### Multi-Objective Optimization

```python
# Find materials that rank highly across multiple applications
all_rankings = ranker.rank_for_all_applications(materials)

# Calculate average rank across applications
avg_ranks = all_rankings.groupby('formula')['rank'].mean().sort_values()

print("Materials with Best Average Rank Across All Applications:")
for formula, avg_rank in avg_ranks.head(10).items():
    print(f"  {formula}: Average rank {avg_rank:.1f}")

# Find materials in top 20 for multiple applications
top_20_counts = all_rankings[all_rankings['rank'] <= 20].groupby('formula').size()
top_20_counts = top_20_counts.sort_values(ascending=False)

print("\nMaterials in Top 20 for Multiple Applications:")
for formula, count in top_20_counts.head(10).items():
    print(f"  {formula}: Top 20 in {count} applications")
```

### Visualization

```python
from ceramic_discovery.analysis import Visualizer
import matplotlib.pyplot as plt

viz = Visualizer()

# Create ranking comparison plot
all_rankings = ranker.rank_for_all_applications(materials)

# Plot top 10 for each application
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
applications = all_rankings['application'].unique()

for ax, app in zip(axes.flat, applications):
    app_data = all_rankings[all_rankings['application'] == app].head(10)
    ax.barh(app_data['formula'], app_data['overall_score'])
    ax.set_xlabel('Score')
    ax.set_title(app.replace('_', ' ').title())
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('./results/application_rankings.png', dpi=300)

# Create radar chart for top material
top_material = results[0]
properties = ['hardness', 'thermal_cond', 'melting_point', 'formation_energy']
scores = [top_material.component_scores.get(p, 0) for p in properties]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False).tolist()
scores += scores[:1]
angles += angles[:1]

ax.plot(angles, scores, 'o-', linewidth=2)
ax.fill(angles, scores, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(properties)
ax.set_ylim(0, 1)
ax.set_title(f"{top_material.formula} - Property Scores")

plt.savefig(f'./results/{top_material.formula}_radar.png', dpi=300)
```

---

## Best Practices

### 1. Validate Input Data

```python
# Check for required properties
required_props = ['hardness', 'thermal_conductivity_1000C', 'melting_point']

valid_materials = []
for material in materials:
    if all(material.get(prop) is not None for prop in required_props):
        valid_materials.append(material)
    else:
        missing = [p for p in required_props if material.get(p) is None]
        print(f"Warning: {material['formula']} missing {missing}")

print(f"Valid materials: {len(valid_materials)}/{len(materials)}")
```

### 2. Consider Confidence Scores

```python
# Filter by confidence threshold
high_confidence_results = [
    r for r in results
    if r.confidence >= 0.75
]

# Report confidence distribution
confidences = [r.confidence for r in results]
print(f"Mean confidence: {np.mean(confidences):.2f}")
print(f"Median confidence: {np.median(confidences):.2f}")
print(f"High confidence (≥0.75): {sum(c >= 0.75 for c in confidences)}/{len(confidences)}")
```

### 3. Cross-Reference with Literature

```python
from ceramic_discovery.dft import LiteratureDatabase

lit_db = LiteratureDatabase()

# Check if top candidates have literature validation
for ranked in results[:10]:
    lit_data = lit_db.get_material_data(ranked.formula)
    if lit_data:
        print(f"✓ {ranked.formula}: Literature data available")
    else:
        print(f"⚠ {ranked.formula}: No literature validation")
```

### 4. Document Assumptions

```python
# Document ranking assumptions
assumptions = {
    'application': 'aerospace_hypersonic',
    'property_weights': {
        'hardness': 0.30,
        'thermal_conductivity': 0.25,
        'melting_point': 0.25,
        'formation_energy': 0.20
    },
    'target_ranges': {
        'hardness': (28, 35),
        'thermal_conductivity': (50, 150),
        'melting_point': (2500, 4000),
        'formation_energy': (-1.0, -0.3)
    },
    'data_sources': ['JARVIS-DFT', 'Materials Project'],
    'date': '2025-11-27'
}

# Save with results
import json
with open('./results/ranking_assumptions.json', 'w') as f:
    json.dump(assumptions, f, indent=2)
```

---

## References

- **Materials Selection**: Ashby, M. F. "Materials Selection in Mechanical Design" (2011)
- **Multi-Objective Optimization**: Deb, K. "Multi-Objective Optimization using Evolutionary Algorithms" (2001)
- **Ceramic Properties**: Munro, R. G. "Material Properties of Titanium Diboride" NIST (2000)

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-27  
**For Questions:** See main documentation or contact project team
