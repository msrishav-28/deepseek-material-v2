"""
Example: Configuration Management

This example demonstrates how to use and customize the configuration
system for the SiC Alloy Designer Integration.
"""

from ceramic_discovery import Config
from pathlib import Path


def example_basic_usage():
    """Example 1: Basic configuration usage."""
    print("=" * 60)
    print("Example 1: Basic Configuration Usage")
    print("=" * 60)
    
    # Get the global configuration instance
    from ceramic_discovery.config import config
    
    # Display data source settings
    print("\nData Sources Configuration:")
    print(f"  JARVIS enabled: {config.data_sources.jarvis.enabled}")
    print(f"  JARVIS file: {config.data_sources.jarvis.file_path}")
    print(f"  NIST enabled: {config.data_sources.nist.enabled}")
    print(f"  Literature enabled: {config.data_sources.literature.enabled}")
    
    # Display feature engineering settings
    print("\nFeature Engineering Configuration:")
    print(f"  Composition descriptors: {config.feature_engineering.composition_descriptors.enabled}")
    print(f"  Structure descriptors: {config.feature_engineering.structure_descriptors.enabled}")
    
    # Display application ranking settings
    print("\nApplication Ranking Configuration:")
    print(f"  Enabled: {config.application_ranking.enabled}")
    print(f"  Applications: {', '.join(config.application_ranking.applications)}")
    
    print()


def example_custom_configuration():
    """Example 2: Creating a custom configuration."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create a new configuration instance
    config = Config()
    
    # Customize data sources
    config.data_sources.jarvis.enabled = False
    print("\nDisabled JARVIS data source")
    
    # Customize applications for ranking
    config.application_ranking.applications = [
        "aerospace_hypersonic",
        "cutting_tools"
    ]
    print(f"Limited applications to: {config.application_ranking.applications}")
    
    # Customize feature engineering
    config.feature_engineering.structure_descriptors.enabled = False
    print("Disabled structure descriptors")
    
    # Adjust experimental planning costs
    config.experimental_planning.synthesis.cost_multiplier = 1.5
    config.experimental_planning.characterization.cost_multiplier = 1.3
    print("\nAdjusted cost multipliers:")
    print(f"  Synthesis: {config.experimental_planning.synthesis.cost_multiplier}x")
    print(f"  Characterization: {config.experimental_planning.characterization.cost_multiplier}x")
    
    print()


def example_validation():
    """Example 3: Configuration validation."""
    print("=" * 60)
    print("Example 3: Configuration Validation")
    print("=" * 60)
    
    # Create a configuration
    config = Config()
    
    # Try to validate with invalid settings
    print("\nTesting validation with invalid correlation threshold...")
    config.feature_engineering.correlation_threshold = 1.5
    
    try:
        config.validate()
        print("Validation passed")
    except ValueError as e:
        print(f"Validation failed (expected): {e}")
    
    # Fix the issue
    config.feature_engineering.correlation_threshold = 0.95
    print("\nFixed correlation threshold to 0.95")
    
    # Test with empty applications
    print("\nTesting validation with empty applications...")
    config.application_ranking.applications = []
    
    try:
        config.validate()
        print("Validation passed")
    except ValueError as e:
        print(f"Validation failed (expected): {e}")
    
    # Fix the issue
    config.application_ranking.applications = ["aerospace_hypersonic"]
    print("\nFixed applications list")
    
    print()


def example_environment_based():
    """Example 4: Environment-based configuration."""
    print("=" * 60)
    print("Example 4: Environment-Based Configuration")
    print("=" * 60)
    
    print("\nConfiguration can be controlled via environment variables:")
    print("  JARVIS_ENABLED=false")
    print("  JARVIS_FILE_PATH=/custom/path/jarvis.json")
    print("  NIST_ENABLED=true")
    print("  NIST_DATA_DIR=/custom/path/nist")
    print("  LITERATURE_ENABLED=true")
    
    print("\nSet these in a .env file or export them in your shell.")
    print("The configuration will automatically load them on initialization.")
    
    print()


def example_application_specific():
    """Example 5: Application-specific configuration."""
    print("=" * 60)
    print("Example 5: Application-Specific Configuration")
    print("=" * 60)
    
    config = Config()
    
    # Configure for aerospace-only analysis
    print("\nConfiguring for aerospace-only analysis:")
    config.application_ranking.applications = ["aerospace_hypersonic"]
    config.application_ranking.min_properties_for_ranking = 3
    config.application_ranking.confidence_threshold = 0.7
    
    print(f"  Applications: {config.application_ranking.applications}")
    print(f"  Min properties: {config.application_ranking.min_properties_for_ranking}")
    print(f"  Confidence threshold: {config.application_ranking.confidence_threshold}")
    
    # Configure for cutting tools analysis
    print("\nConfiguring for cutting tools analysis:")
    config.application_ranking.applications = ["cutting_tools"]
    config.feature_engineering.composition_descriptors.calculate_entropy = True
    config.feature_engineering.structure_descriptors.calculate_pugh_ratio = True
    
    print(f"  Applications: {config.application_ranking.applications}")
    print(f"  Calculate entropy: {config.feature_engineering.composition_descriptors.calculate_entropy}")
    print(f"  Calculate Pugh ratio: {config.feature_engineering.structure_descriptors.calculate_pugh_ratio}")
    
    print()


def example_performance_tuning():
    """Example 6: Performance tuning configuration."""
    print("=" * 60)
    print("Example 6: Performance Tuning")
    print("=" * 60)
    
    config = Config()
    
    print("\nFor large datasets, optimize by:")
    
    # Disable unused data sources
    config.data_sources.nist.enabled = False
    print("  1. Disabling unused data sources (NIST)")
    
    # Disable unused descriptors
    config.feature_engineering.structure_descriptors.calculate_energy_density = False
    print("  2. Disabling unused descriptors (energy density)")
    
    # Reduce applications
    config.application_ranking.applications = ["aerospace_hypersonic", "cutting_tools"]
    print(f"  3. Limiting applications to: {len(config.application_ranking.applications)}")
    
    # Increase minimum properties
    config.application_ranking.min_properties_for_ranking = 4
    print(f"  4. Requiring more properties: {config.application_ranking.min_properties_for_ranking}")
    
    # Enable feature correlation removal
    config.feature_engineering.remove_correlated_features = True
    config.feature_engineering.correlation_threshold = 0.90
    print(f"  5. Removing correlated features (threshold: {config.feature_engineering.correlation_threshold})")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Configuration Management Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_custom_configuration()
    example_validation()
    example_environment_based()
    example_application_specific()
    example_performance_tuning()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
