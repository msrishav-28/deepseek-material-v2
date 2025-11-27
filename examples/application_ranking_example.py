"""Example of ranking materials for specific applications."""

from pathlib import Path
import pandas as pd
from ceramic_discovery.screening.application_ranker import ApplicationRanker


def main():
    """Demonstrate application-specific material ranking."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: Application-Specific Ranking Example")
    print("=" * 80)
    
    # Step 1: Initialize application ranker
    print("\n1. Initializing application ranker...")
    
    ranker = ApplicationRanker()
    
    print(f"   [OK] Application ranker initialized")
    print(f"   Available applications: {len(ranker.applications)}")
    
    # Step 2: Create sample material dataset
    print("\n2. Creating sample material dataset...")
    
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
    print("\n3. Ranking for all applications...")
    
    all_rankings = ranker.rank_for_all_applications(materials_df)
    
    print(f"   [OK] Ranked {len(all_rankings)} materials across all applications")
    
    # Step 4: Export rankings
    print("\n4. Exporting ranking results...")
    
    output_file = Path("results/application_rankings.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_rankings.to_csv(output_file, index=False)
    
    print(f"   [OK] Exported rankings to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Application ranking completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
