"""Example usage of the validation and reproducibility framework."""

from pathlib import Path

import numpy as np

from ceramic_discovery.ceramics.ceramic_system import CeramicSystemFactory
from ceramic_discovery.validation import (
    PhysicalPlausibilityValidator,
    SensitivityAnalyzer,
    ReproducibilityFramework,
)


def example_physical_validation():
    """Example: Validate material properties for physical plausibility."""
    print("=" * 60)
    print("Physical Plausibility Validation Example")
    print("=" * 60)
    
    # Create validator
    validator = PhysicalPlausibilityValidator()
    
    # Load SiC material
    sic = CeramicSystemFactory.create_sic()
    
    # Extract properties for validation
    properties = {
        "hardness": sic.mechanical.hardness,
        "fracture_toughness": sic.mechanical.fracture_toughness,
        "youngs_modulus": sic.mechanical.youngs_modulus,
        "bulk_modulus": sic.mechanical.bulk_modulus,
        "shear_modulus": sic.mechanical.shear_modulus,
        "poisson_ratio": sic.mechanical.poisson_ratio,
        "density": sic.structural.density,
        "thermal_conductivity_25C": sic.thermal.thermal_conductivity_25C,
        "thermal_conductivity_1000C": sic.thermal.thermal_conductivity_1000C,
    }
    
    # Validate material
    report = validator.validate_material("SiC", properties)
    
    # Print results
    print(f"\n{report.summary()}\n")
    
    if report.violations:
        print("Violations found:")
        for violation in report.violations:
            print(f"  - {violation.severity.value.upper()}: {violation.message}")
    else:
        print("No violations found. Material properties are physically plausible.")
    
    # Add literature reference for cross-validation
    validator.add_literature_reference(
        material_id="SiC",
        property_name="hardness",
        value=28.0,
        uncertainty=2.0,
        source="Munro, 1997"
    )
    
    # Cross-reference with literature
    lit_violation = validator.cross_reference_literature(
        material_id="SiC",
        property_name="hardness",
        predicted_value=sic.mechanical.hardness.value,
        predicted_uncertainty=sic.mechanical.hardness.uncertainty
    )
    
    if lit_violation:
        print(f"\nLiterature cross-reference: {lit_violation.message}")
    else:
        print("\nPredicted hardness is consistent with literature data.")


def example_sensitivity_analysis():
    """Example: Analyze sensitivity of performance to property changes."""
    print("\n" + "=" * 60)
    print("Sensitivity Analysis Example")
    print("=" * 60)
    
    analyzer = SensitivityAnalyzer()
    
    # Define a simple ballistic performance metric
    # (This is a simplified example; real V50 prediction would use ML models)
    def ballistic_performance(props):
        """Simplified ballistic performance metric."""
        return (
            props.get("hardness", 0) * 2.0 +
            props.get("fracture_toughness", 0) * 3.0 +
            props.get("density", 0) * 1.5 +
            props.get("thermal_conductivity_1000C", 0) * 0.5
        )
    
    # Base properties for SiC
    base_properties = {
        "hardness": 28.0,
        "fracture_toughness": 4.6,
        "density": 3.21,
        "thermal_conductivity_1000C": 45.0,
        "youngs_modulus": 410.0,
    }
    
    # Analyze sensitivity
    sensitivities = analyzer.analyze_property_sensitivity(
        base_properties,
        ballistic_performance,
        perturbation_fraction=0.1
    )
    
    # Rank performance drivers
    print("\nTop Performance Drivers:")
    ranked = analyzer.rank_performance_drivers(sensitivities, top_n=5)
    
    for i, (prop_name, sensitivity) in enumerate(ranked, 1):
        direction = "increases" if sensitivity > 0 else "decreases"
        print(f"{i}. {prop_name}: {sensitivity:.3f}")
        print(f"   A 10% increase in {prop_name} {direction} performance by {abs(sensitivity * 10):.1f}%")
    
    # Generate full report
    property_units = {
        "hardness": "GPa",
        "fracture_toughness": "MPa·m^0.5",
        "density": "g/cm³",
        "thermal_conductivity_1000C": "W/m·K",
        "youngs_modulus": "GPa",
    }
    
    print("\n" + analyzer.generate_sensitivity_report(sensitivities, property_units))


def example_reproducibility():
    """Example: Use reproducibility framework for experiments."""
    print("\n" + "=" * 60)
    print("Reproducibility Framework Example")
    print("=" * 60)
    
    # Create framework
    framework = ReproducibilityFramework(
        experiment_dir=Path("./results/experiments"),
        master_seed=42
    )
    
    # Start experiment
    print("\nStarting experiment...")
    snapshot = framework.start_experiment(
        experiment_id="sic_screening_001",
        experiment_type="dopant_screening",
        parameters={
            "base_ceramic": "SiC",
            "dopant_elements": ["Ti", "Zr", "Hf"],
            "concentrations": [0.01, 0.02, 0.05],
            "stability_threshold": 0.1
        },
        random_seed=42
    )
    
    print(f"Experiment ID: {snapshot.experiment_id}")
    print(f"Random seed: {snapshot.parameters.random_seed}")
    
    # Simulate data transformations
    print("\nRecording data transformations...")
    
    framework.record_transformation(
        description="Load DFT data from Materials Project",
        data={"materials_count": 150}
    )
    
    framework.record_transformation(
        description="Filter by thermodynamic stability (ΔE_hull ≤ 0.1 eV/atom)",
        data={"viable_materials": 45}
    )
    
    framework.record_transformation(
        description="Calculate composite properties",
        data={"composites_evaluated": 135}
    )
    
    # Simulate experiment results
    np.random.seed(42)  # Deterministic results
    results = {
        "top_candidates": [
            {"composition": "SiC:Ti(2%)", "predicted_v50": 850.5},
            {"composition": "SiC:Zr(5%)", "predicted_v50": 875.2},
            {"composition": "SiC:Hf(2%)", "predicted_v50": 865.8},
        ],
        "total_screened": 150,
        "viable_count": 45,
        "avg_confidence": 0.82,
    }
    
    # Finalize experiment
    print("\nFinalizing experiment...")
    final_snapshot = framework.finalize_experiment(results)
    
    print(f"Experiment saved to: {framework.experiment_dir / f'{snapshot.experiment_id}.json'}")
    print(f"Provenance hash: {final_snapshot.provenance.final_hash[:16]}...")
    
    # Export reproducibility package
    print("\nExporting reproducibility package...")
    output_dir = Path("./results/reproducibility_packages") / snapshot.experiment_id
    framework.export_reproducibility_package(snapshot.experiment_id, output_dir)
    
    print(f"Package exported to: {output_dir}")
    print("Package includes:")
    print("  - experiment_snapshot.json")
    print("  - requirements.txt")
    print("  - README.md")


def example_batch_validation():
    """Example: Validate multiple materials in batch."""
    print("\n" + "=" * 60)
    print("Batch Validation Example")
    print("=" * 60)
    
    validator = PhysicalPlausibilityValidator()
    
    # Create multiple ceramic systems
    materials = []
    
    for ceramic_type in ["SiC", "B4C", "WC", "TiC", "Al2O3"]:
        factory_method = getattr(CeramicSystemFactory, f"create_{ceramic_type.lower()}")
        ceramic = factory_method()
        
        materials.append({
            "id": ceramic_type,
            "hardness": ceramic.mechanical.hardness,
            "fracture_toughness": ceramic.mechanical.fracture_toughness,
            "density": ceramic.structural.density,
            "thermal_conductivity_25C": ceramic.thermal.thermal_conductivity_25C,
        })
    
    # Validate all materials
    reports = validator.validate_batch(materials)
    
    # Get statistics
    stats = validator.get_validation_statistics(reports)
    
    print(f"\nValidation Statistics:")
    print(f"  Total materials: {stats['total_materials']}")
    print(f"  Valid materials: {stats['valid_materials']}")
    print(f"  Invalid materials: {stats['invalid_materials']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"\nViolations by severity:")
    for severity, count in stats['violations_by_severity'].items():
        if count > 0:
            print(f"  {severity.capitalize()}: {count}")
    
    if stats['most_common_violations']:
        print(f"\nMost common violations:")
        for prop_name, count in stats['most_common_violations']:
            print(f"  {prop_name}: {count}")


if __name__ == "__main__":
    # Run all examples
    example_physical_validation()
    example_sensitivity_analysis()
    example_reproducibility()
    example_batch_validation()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
