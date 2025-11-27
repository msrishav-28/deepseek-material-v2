"""Script to create all integration example files."""

from pathlib import Path


# Example 1: JARVIS Data Loading
jarvis_example = '''"""Example of loading and processing JARVIS-DFT data for carbide materials."""

from pathlib import Path
from ceramic_discovery.dft.jarvis_client import JarvisClient
import json
import statistics


def main():
    """Demonstrate JARVIS-DFT data loading and carbide extraction."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: JARVIS-DFT Data Loading Example")
    print("=" * 80)
    
    # Step 1: Initialize JARVIS client
    print("\\n1. Initializing JARVIS client...")
    
    jarvis_file = Path("data/jdft_3d-7-7-2018.json")
    
    if not jarvis_file.exists():
        print(f"   [INFO] JARVIS file not found at: {jarvis_file}")
        print("   Creating sample data for demonstration...")
        
        jarvis_file.parent.mkdir(parents=True, exist_ok=True)
        
        sample_data = [
            {"jid": "JVASP-1234", "formula": "SiC", "elements": ["Si", "C"],
             "formation_energy_peratom": -1.234, "optb88vdw_bandgap": 2.3,
             "bulk_modulus_kv": 220.5, "shear_modulus_gv": 180.2},
            {"jid": "JVASP-5678", "formula": "TiC", "elements": ["Ti", "C"],
             "formation_energy_peratom": -1.456, "optb88vdw_bandgap": 0.0,
             "bulk_modulus_kv": 240.8, "shear_modulus_gv": 175.3},
        ]
        
        with open(jarvis_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"   [OK] Created sample JARVIS data at: {jarvis_file}")
    
    client = JarvisClient(str(jarvis_file))
    print(f"   [OK] JARVIS client initialized")
    
    # Step 2: Load carbide materials
    print("\\n2. Loading carbide materials...")
    
    metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "Nb", "V", "Cr", "W", "Mo"}
    carbides = client.load_carbides(metal_elements)
    
    print(f"   [OK] Found {len(carbides)} carbide materials")
    
    # Step 3: Export data
    print("\\n3. Exporting carbide data...")
    
    import pandas as pd
    
    df_data = []
    for material in carbides:
        df_data.append({
            'formula': material.get('formula'),
            'formation_energy_peratom': material.get('formation_energy_peratom'),
            'band_gap': material.get('band_gap'),
            'bulk_modulus_kv': material.get('bulk_modulus_kv'),
        })
    
    df = pd.DataFrame(df_data)
    
    output_file = Path("results/jarvis_carbides.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"   [OK] Exported {len(df)} carbides to: {output_file}")
    
    print("\\n" + "=" * 80)
    print("JARVIS data loading completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
'''

# Example 2: Multi-source Data Combination
multi_source_example = '''"""Example of combining data from multiple sources."""

from pathlib import Path
import pandas as pd
from ceramic_discovery.dft.jarvis_client import JarvisClient
from ceramic_discovery.dft.literature_database import LiteratureDatabase
from ceramic_discovery.dft.data_combiner import DataCombiner
import json


def main():
    """Demonstrate multi-source data combination."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: Multi-Source Data Combination Example")
    print("=" * 80)
    
    # Step 1: Load JARVIS data
    print("\\n1. Loading JARVIS-DFT data...")
    
    jarvis_file = Path("data/jdft_3d-7-7-2018.json")
    
    if not jarvis_file.exists():
        jarvis_file.parent.mkdir(parents=True, exist_ok=True)
        sample_jarvis = [
            {"jid": "JVASP-1234", "formula": "SiC", "elements": ["Si", "C"],
             "formation_energy_peratom": -1.234, "optb88vdw_bandgap": 2.3,
             "bulk_modulus_kv": 220.5, "shear_modulus_gv": 180.2},
        ]
        with open(jarvis_file, 'w') as f:
            json.dump(sample_jarvis, f, indent=2)
    
    jarvis_client = JarvisClient(str(jarvis_file))
    metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta"}
    jarvis_data = jarvis_client.load_carbides(metal_elements)
    
    print(f"   [OK] Loaded {len(jarvis_data)} materials from JARVIS")
    
    # Step 2: Load literature data
    print("\\n2. Loading literature database...")
    
    lit_db = LiteratureDatabase()
    available_materials = lit_db.list_available_materials()
    
    print(f"   [OK] Literature database contains {len(available_materials)} materials")
    
    literature_data = {}
    for material in available_materials:
        lit_data = lit_db.get_material_data(material)
        if lit_data:
            literature_data[material] = lit_data
    
    # Step 3: Combine all data sources
    print("\\n3. Combining data from all sources...")
    
    combiner = DataCombiner()
    
    mp_data = [
        {'formula': 'SiC', 'formation_energy_peratom': -1.25,
         'band_gap': 2.4, 'density': 3.21},
    ]
    
    combined_df = combiner.combine_sources(
        mp_data=mp_data,
        jarvis_data=jarvis_data,
        nist_data={},
        literature_data=literature_data
    )
    
    print(f"   [OK] Combined dataset created")
    print(f"   Total materials: {len(combined_df)}")
    
    # Step 4: Export combined data
    print("\\n4. Exporting combined dataset...")
    
    output_file = Path("results/combined_materials_data.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"   [OK] Exported combined data to: {output_file}")
    
    print("\\n" + "=" * 80)
    print("Multi-source data combination completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
'''

# Example 3: Application Ranking
app_ranking_example = '''"""Example of ranking materials for specific applications."""

from pathlib import Path
import pandas as pd
from ceramic_discovery.screening.application_ranker import ApplicationRanker


def main():
    """Demonstrate application-specific material ranking."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: Application-Specific Ranking Example")
    print("=" * 80)
    
    # Step 1: Initialize application ranker
    print("\\n1. Initializing application ranker...")
    
    ranker = ApplicationRanker()
    
    print(f"   [OK] Application ranker initialized")
    print(f"   Available applications: {len(ranker.applications)}")
    
    # Step 2: Create sample material dataset
    print("\\n2. Creating sample material dataset...")
    
    materials = [
        {'formula': 'SiC', 'formation_energy_peratom': -1.234, 'band_gap': 2.3,
         'bulk_modulus_kv': 220.5, 'hardness_vickers': 2800,
         'thermal_conductivity': 120, 'melting_point': 3103, 'density': 3.21},
        {'formula': 'TiC', 'formation_energy_peratom': -1.456, 'band_gap': 0.0,
         'bulk_modulus_kv': 240.8, 'hardness_vickers': 3200,
         'thermal_conductivity': 30, 'melting_point': 3340, 'density': 4.93},
        {'formula': 'B4C', 'formation_energy_peratom': -0.567, 'band_gap': 2.1,
         'bulk_modulus_kv': 247.0, 'hardness_vickers': 3800,
         'thermal_conductivity': 90, 'melting_point': 2723, 'density': 2.52},
    ]
    
    materials_df = pd.DataFrame(materials)
    
    print(f"   [OK] Created dataset with {len(materials_df)} materials")
    
    # Step 3: Rank for all applications
    print("\\n3. Ranking for all applications...")
    
    all_rankings = ranker.rank_for_all_applications(materials_df)
    
    print(f"   [OK] Ranked {len(all_rankings)} materials across all applications")
    
    # Step 4: Export rankings
    print("\\n4. Exporting ranking results...")
    
    output_file = Path("results/application_rankings.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_rankings.to_csv(output_file, index=False)
    
    print(f"   [OK] Exported rankings to: {output_file}")
    
    print("\\n" + "=" * 80)
    print("Application ranking completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
'''

# Example 4: Experimental Planning
exp_planning_example = '''"""Example of generating experimental validation plans."""

from pathlib import Path
import pandas as pd
from ceramic_discovery.validation.experimental_planner import ExperimentalPlanner


def main():
    """Demonstrate experimental planning."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: Experimental Planning Example")
    print("=" * 80)
    
    # Step 1: Initialize experimental planner
    print("\\n1. Initializing experimental planner...")
    
    planner = ExperimentalPlanner()
    
    print(f"   [OK] Experimental planner initialized")
    
    # Step 2: Define top material candidates
    print("\\n2. Defining top material candidates...")
    
    candidates = [
        {'formula': 'SiC', 'formation_energy_peratom': -1.234,
         'hardness_vickers': 2800, 'thermal_conductivity': 120,
         'melting_point': 3103, 'application_score': 0.85},
        {'formula': 'B4C', 'formation_energy_peratom': -0.567,
         'hardness_vickers': 3800, 'thermal_conductivity': 90,
         'melting_point': 2723, 'application_score': 0.92},
    ]
    
    print(f"   [OK] Defined {len(candidates)} candidate materials")
    
    # Step 3: Generate experimental protocols
    print("\\n3. Generating experimental protocols...")
    
    comparison_data = []
    
    for candidate in candidates:
        formula = candidate['formula']
        properties = candidate
        
        synthesis_methods = planner.design_synthesis_protocol(formula, properties)
        char_plan = planner.design_characterization_plan(
            formula, ['hardness', 'thermal_conductivity']
        )
        resource_estimate = planner.estimate_resources(synthesis_methods, char_plan)
        
        comparison_data.append({
            'formula': formula,
            'application_score': candidate['application_score'],
            'timeline_months': resource_estimate.timeline_months,
            'cost_k_dollars': resource_estimate.estimated_cost_k_dollars,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Step 4: Export experimental protocols
    print("\\n4. Exporting experimental protocols...")
    
    output_dir = Path("results/experimental_protocols")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "candidate_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"   [OK] Exported comparison data to: {comparison_file}")
    
    print("\\n" + "=" * 80)
    print("Experimental planning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
'''

# Example 5: Notebook Creation Script
notebook_script = '''"""Script to create the SiC Alloy Integration workflow notebook."""

import json
from pathlib import Path


def create_notebook():
    """Create a comprehensive Jupyter notebook."""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SiC Alloy Designer Integration Workflow\\n",
                    "\\n",
                    "Complete workflow demonstration."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\\n",
                    "print('[OK] Imports successful')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    output_path = Path("notebooks/06_sic_alloy_integration_workflow.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"[OK] Created notebook: {output_path}")


if __name__ == "__main__":
    create_notebook()
'''

# Write all example files
examples = {
    'examples/jarvis_data_loading_example.py': jarvis_example,
    'examples/multi_source_data_combination_example.py': multi_source_example,
    'examples/application_ranking_example.py': app_ranking_example,
    'examples/experimental_planning_example.py': exp_planning_example,
    'examples/create_workflow_notebook.py': notebook_script,
}

for filepath, content in examples.items():
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\nAll example files created successfully!")
'''

with open('create_examples.py', 'w', encoding='utf-8') as f:
    f.write(create_examples_script)

print("Created: create_examples.py")
