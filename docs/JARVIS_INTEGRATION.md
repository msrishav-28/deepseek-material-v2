# JARVIS-DFT Integration Guide

## Overview

This guide explains how to use the JARVIS-DFT (Joint Automated Repository for Various Integrated Simulations) integration in the Ceramic Armor Discovery Framework. JARVIS-DFT provides access to over 25,000 DFT-calculated materials, including 400+ carbide compounds.

**Version:** 1.0  
**Date:** 2025-11-27  
**Requirements:** 1.1, 1.4, 9.1

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Acquisition](#data-acquisition)
3. [Loading JARVIS Data](#loading-jarvis-data)
4. [Filtering Carbides](#filtering-carbides)
5. [Property Extraction](#property-extraction)
6. [Integration with Other Sources](#integration-with-other-sources)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Ceramic Armor Discovery Framework installed
- JARVIS-DFT database file (JSON format)
- Python 3.11+
- 4 GB RAM minimum for loading full database

### Quick Start

```python
from ceramic_discovery.dft import JarvisClient

# Initialize client
client = JarvisClient(jarvis_file_path="./data/jdft_3d-7-7-2018.json")

# Load all carbides
carbides = client.load_carbides()

print(f"Loaded {len(carbides)} carbide materials")
```

---

## Data Acquisition

### Downloading JARVIS Database

The JARVIS-DFT database can be downloaded from the official JARVIS repository:

**Option 1: Direct Download**
```bash
# Download JARVIS 3D materials database
wget https://www.ctcms.nist.gov/~knc6/JVASP.json -O ./data/jdft_3d-7-7-2018.json
```

**Option 2: Using JARVIS-Tools**
```bash
# Install jarvis-tools
pip install jarvis-tools

# Download using Python
python -c "from jarvis.db.figshare import data; d = data('dft_3d'); import json; json.dump(d, open('./data/jdft_3d-7-7-2018.json', 'w'))"
```

### Database Structure

The JARVIS database is a JSON file containing an array of material entries:

```json
[
  {
    "jid": "JVASP-1002",
    "formula": "SiC",
    "atoms": {...},
    "formation_energy_peratom": -0.6523,
    "optb88vdw_bandgap": 2.416,
    "bulk_modulus_kv": 220.5,
    "shear_modulus_gv": 189.3,
    ...
  },
  ...
]
```

---

## Loading JARVIS Data

### Basic Loading

```python
from ceramic_discovery.dft import JarvisClient

# Initialize client with database file
client = JarvisClient(jarvis_file_path="./data/jdft_3d-7-7-2018.json")

# Check how many materials were loaded
print(f"Total materials in database: {len(client.data)}")
```

### Error Handling

```python
from ceramic_discovery.dft import JarvisClient

try:
    client = JarvisClient(jarvis_file_path="./data/jdft_3d-7-7-2018.json")
except FileNotFoundError:
    print("JARVIS database file not found. Please download it first.")
except ValueError as e:
    print(f"Invalid database file: {e}")
except Exception as e:
    print(f"Error loading JARVIS data: {e}")
```

---

## Filtering Carbides

### Load All Carbides

```python
# Load all carbide materials (materials containing carbon)
all_carbides = client.load_carbides()

print(f"Found {len(all_carbides)} carbide materials")

# Display first few
for material in all_carbides[:5]:
    print(f"  {material.formula}: E_f = {material.formation_energy_per_atom:.3f} eV/atom")
```

### Filter by Metal Elements

```python
# Load carbides with specific metal elements
metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "W"}

filtered_carbides = client.load_carbides(metal_elements=metal_elements)

print(f"Found {len(filtered_carbides)} carbides with specified metals")

# Group by metal
from collections import defaultdict
by_metal = defaultdict(list)

for material in filtered_carbides:
    formula = material.formula
    for metal in metal_elements:
        if metal in formula:
            by_metal[metal].append(formula)

for metal, formulas in sorted(by_metal.items()):
    print(f"{metal}: {len(formulas)} carbides")
```

### Filter by Properties

```python
# Load carbides and filter by properties
carbides = client.load_carbides()

# Filter by formation energy (stable materials)
stable_carbides = [
    m for m in carbides
    if m.formation_energy_per_atom is not None
    and m.formation_energy_per_atom < -0.5
]

print(f"Stable carbides (E_f < -0.5 eV/atom): {len(stable_carbides)}")

# Filter by band gap (semiconductors)
semiconductor_carbides = [
    m for m in carbides
    if m.band_gap is not None
    and 0.5 < m.band_gap < 3.0
]

print(f"Semiconductor carbides (0.5 < E_g < 3.0 eV): {len(semiconductor_carbides)}")
```

---

## Property Extraction

### Available Properties

JARVIS-DFT provides the following properties:

- **Energetic**: Formation energy, energy above hull
- **Electronic**: Band gap, Fermi energy
- **Mechanical**: Bulk modulus, shear modulus
- **Structural**: Crystal structure, space group, density

### Extracting Properties

```python
# Load carbides
carbides = client.load_carbides(metal_elements={"Si", "Ti"})

# Extract specific properties
for material in carbides[:10]:
    print(f"\n{material.formula}:")
    print(f"  Formation Energy: {material.formation_energy_per_atom:.3f} eV/atom")
    print(f"  Band Gap: {material.band_gap:.2f} eV" if material.band_gap else "  Band Gap: N/A")
    print(f"  Bulk Modulus: {material.bulk_modulus:.1f} GPa" if material.bulk_modulus else "  Bulk Modulus: N/A")
    print(f"  Shear Modulus: {material.shear_modulus:.1f} GPa" if material.shear_modulus else "  Shear Modulus: N/A")
```

### Converting to DataFrame

```python
import pandas as pd

# Load carbides
carbides = client.load_carbides()

# Convert to DataFrame
data = []
for material in carbides:
    data.append({
        'material_id': material.material_id,
        'formula': material.formula,
        'formation_energy': material.formation_energy_per_atom,
        'band_gap': material.band_gap,
        'bulk_modulus': material.bulk_modulus,
        'shear_modulus': material.shear_modulus,
        'density': material.density
    })

df = pd.DataFrame(data)

# Display statistics
print(df.describe())

# Save to CSV
df.to_csv("./results/jarvis_carbides.csv", index=False)
```

---

## Integration with Other Sources

### Combining with Materials Project

```python
from ceramic_discovery.dft import JarvisClient, MaterialsProjectClient, DataCombiner

# Load JARVIS data
jarvis_client = JarvisClient("./data/jdft_3d-7-7-2018.json")
jarvis_carbides = jarvis_client.load_carbides(metal_elements={"Si", "Ti", "Zr"})

# Load Materials Project data
mp_client = MaterialsProjectClient(api_key="your_api_key")
mp_materials = []
for formula in ["SiC", "TiC", "ZrC"]:
    mp_materials.extend(mp_client.search_materials(formula))

# Combine sources
combiner = DataCombiner()
combined = combiner.combine_sources(
    mp_data=mp_materials,
    jarvis_data=jarvis_carbides,
    nist_data={},
    literature_data={}
)

print(f"Combined dataset: {len(combined)} materials")
print(f"Materials from JARVIS: {(combined['source'] == 'JARVIS').sum()}")
print(f"Materials from MP: {(combined['source'] == 'MP').sum()}")
print(f"Materials from both: {(combined['source'] == 'MP,JARVIS').sum()}")
```

### Combining with NIST and Literature

```python
from ceramic_discovery.dft import (
    JarvisClient,
    NISTClient,
    LiteratureDatabase,
    DataCombiner
)

# Load all sources
jarvis_client = JarvisClient("./data/jdft_3d-7-7-2018.json")
nist_client = NISTClient()
lit_db = LiteratureDatabase()

# Get data
jarvis_carbides = jarvis_client.load_carbides()
nist_data = {
    "SiC": nist_client.load_thermochemical_data("./data/nist/SiC.txt", "SiC")
}
literature_data = lit_db.get_all_data()

# Combine
combiner = DataCombiner()
combined = combiner.combine_sources(
    mp_data=[],
    jarvis_data=jarvis_carbides,
    nist_data=nist_data,
    literature_data=literature_data
)

# Analyze data completeness
print("\nData Completeness:")
for col in combined.columns:
    completeness = (1 - combined[col].isna().sum() / len(combined)) * 100
    print(f"  {col}: {completeness:.1f}%")
```

---

## Advanced Usage

### Custom Property Extraction

```python
# Access raw JARVIS entry data
client = JarvisClient("./data/jdft_3d-7-7-2018.json")

# Extract custom properties
for entry in client.data[:5]:
    jid = entry.get("jid", "")
    formula = entry.get("formula", "")
    
    # Extract custom properties not in standard MaterialData
    space_group = entry.get("spg_number", None)
    n_atoms = entry.get("natoms", None)
    
    print(f"{formula} ({jid}):")
    print(f"  Space Group: {space_group}")
    print(f"  Number of Atoms: {n_atoms}")
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def process_carbide_batch(carbides_batch):
    """Process a batch of carbides."""
    results = []
    for material in carbides_batch:
        # Perform analysis
        if material.bulk_modulus and material.shear_modulus:
            pugh_ratio = material.bulk_modulus / material.shear_modulus
            results.append({
                'formula': material.formula,
                'pugh_ratio': pugh_ratio,
                'ductile': pugh_ratio > 1.75
            })
    return results

# Load carbides
client = JarvisClient("./data/jdft_3d-7-7-2018.json")
carbides = client.load_carbides()

# Split into batches
batch_size = 100
batches = [carbides[i:i+batch_size] for i in range(0, len(carbides), batch_size)]

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_carbide_batch, batches))

# Flatten results
all_results = [item for batch in results for item in batch]

print(f"Processed {len(all_results)} carbides")
print(f"Ductile materials: {sum(1 for r in all_results if r['ductile'])}")
```

### Caching for Performance

```python
import pickle
from pathlib import Path

def load_jarvis_with_cache(jarvis_file, cache_file="./data/jarvis_cache.pkl"):
    """Load JARVIS data with caching."""
    cache_path = Path(cache_file)
    
    # Check if cache exists and is newer than source
    if cache_path.exists():
        cache_time = cache_path.stat().st_mtime
        source_time = Path(jarvis_file).stat().st_mtime
        
        if cache_time > source_time:
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Load from source
    print("Loading from JARVIS file...")
    client = JarvisClient(jarvis_file)
    carbides = client.load_carbides()
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(carbides, f)
    
    return carbides

# Use cached loading
carbides = load_jarvis_with_cache("./data/jdft_3d-7-7-2018.json")
```

---

## Troubleshooting

### Issue: File Not Found

**Symptom:** `FileNotFoundError: JARVIS file not found`

**Solution:**
```bash
# Download JARVIS database
wget https://www.ctcms.nist.gov/~knc6/JVASP.json -O ./data/jdft_3d-7-7-2018.json

# Or check file path
ls -lh ./data/jdft_3d-7-7-2018.json
```

### Issue: Memory Error

**Symptom:** `MemoryError: Unable to allocate array`

**Solution:**
```python
# Load only specific materials
client = JarvisClient("./data/jdft_3d-7-7-2018.json")

# Filter early to reduce memory
metal_elements = {"Si"}  # Smaller set
carbides = client.load_carbides(metal_elements=metal_elements)

# Or process in batches
batch_size = 1000
for i in range(0, len(client.data), batch_size):
    batch = client.data[i:i+batch_size]
    # Process batch
```

### Issue: Invalid JSON

**Symptom:** `ValueError: Invalid JSON in JARVIS file`

**Solution:**
```bash
# Validate JSON file
python -c "import json; json.load(open('./data/jdft_3d-7-7-2018.json'))"

# Re-download if corrupted
rm ./data/jdft_3d-7-7-2018.json
wget https://www.ctcms.nist.gov/~knc6/JVASP.json -O ./data/jdft_3d-7-7-2018.json
```

### Issue: Missing Properties

**Symptom:** Many materials have `None` for properties

**Solution:**
```python
# Filter materials with required properties
carbides = client.load_carbides()

# Filter for materials with complete data
complete_carbides = [
    m for m in carbides
    if all([
        m.formation_energy_per_atom is not None,
        m.bulk_modulus is not None,
        m.shear_modulus is not None
    ])
]

print(f"Materials with complete data: {len(complete_carbides)}/{len(carbides)}")
```

---

## Best Practices

### 1. Cache Loaded Data

```python
# Load once, use many times
client = JarvisClient("./data/jdft_3d-7-7-2018.json")
carbides = client.load_carbides()

# Save for later use
import pickle
with open("./data/carbides_cache.pkl", "wb") as f:
    pickle.dump(carbides, f)
```

### 2. Filter Early

```python
# Filter at load time for better performance
carbides = client.load_carbides(metal_elements={"Si", "Ti"})

# Rather than loading all then filtering
# all_carbides = client.load_carbides()
# filtered = [c for c in all_carbides if "Si" in c.formula or "Ti" in c.formula]
```

### 3. Handle Missing Data

```python
# Always check for None values
for material in carbides:
    if material.bulk_modulus is not None:
        # Use property
        print(f"{material.formula}: B = {material.bulk_modulus:.1f} GPa")
    else:
        # Handle missing data
        print(f"{material.formula}: Bulk modulus not available")
```

### 4. Validate Extracted Data

```python
from ceramic_discovery.validation import PhysicalPlausibilityValidator

validator = PhysicalPlausibilityValidator()

for material in carbides[:10]:
    properties = {
        'formation_energy_per_atom': material.formation_energy_per_atom,
        'bulk_modulus': material.bulk_modulus,
        'density': material.density
    }
    
    report = validator.validate_material(material.material_id, properties)
    if not report.is_valid:
        print(f"Warning: {material.formula} has invalid properties")
```

---

## References

- **JARVIS-DFT Database**: https://jarvis.nist.gov/
- **JARVIS Publication**: Choudhary et al., npj Computational Materials (2020)
- **JARVIS-Tools**: https://github.com/usnistgov/jarvis

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-27  
**For Questions:** See main documentation or contact project team
