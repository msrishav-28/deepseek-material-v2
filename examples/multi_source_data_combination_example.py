"""Example of combining data from multiple sources."""

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
    print("\n1. Loading JARVIS-DFT data...")
    
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
    print("\n2. Loading literature database...")
    
    lit_db = LiteratureDatabase()
    available_materials = lit_db.list_available_materials()
    
    print(f"   [OK] Literature database contains {len(available_materials)} materials")
    
    literature_data = {}
    for material in available_materials:
        lit_data = lit_db.get_material_data(material)
        if lit_data:
            literature_data[material] = lit_data
    
    # Step 3: Combine all data sources
    print("\n3. Combining data from all sources...")
    
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
    print("\n4. Exporting combined dataset...")
    
    output_file = Path("results/combined_materials_data.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"   [OK] Exported combined data to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Multi-source data combination completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
