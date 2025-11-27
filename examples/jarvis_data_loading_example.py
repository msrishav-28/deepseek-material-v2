"""Example of loading and processing JARVIS-DFT data for carbide materials."""

from pathlib import Path
from ceramic_discovery.dft.jarvis_client import JarvisClient
import json
import pandas as pd


def main():
    """Demonstrate JARVIS-DFT data loading and carbide extraction."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: JARVIS-DFT Data Loading Example")
    print("=" * 80)
    
    # Step 1: Initialize JARVIS client
    print("\n1. Initializing JARVIS client...")
    
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
    print("\n2. Loading carbide materials...")
    
    metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "Nb", "V", "Cr", "W", "Mo"}
    carbides = client.load_carbides(metal_elements)
    
    print(f"   [OK] Found {len(carbides)} carbide materials")
    
    # Step 3: Export data
    print("\n3. Exporting carbide data...")
    
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
    
    print("\n" + "=" * 80)
    print("JARVIS data loading completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
