"""Example of generating experimental validation plans."""

from pathlib import Path
import pandas as pd
from ceramic_discovery.validation.experimental_planner import ExperimentalPlanner


def main():
    """Demonstrate experimental planning."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: Experimental Planning Example")
    print("=" * 80)
    
    # Step 1: Initialize experimental planner
    print("\n1. Initializing experimental planner...")
    
    planner = ExperimentalPlanner()
    
    print(f"   [OK] Experimental planner initialized")
    
    # Step 2: Define top material candidates
    print("\n2. Defining top material candidates...")
    
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
    print("\n3. Generating experimental protocols...")
    
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
    print("\n4. Exporting experimental protocols...")
    
    output_dir = Path("results/experimental_protocols")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "candidate_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"   [OK] Exported comparison data to: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("Experimental planning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
