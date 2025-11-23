"""Example: Ballistic Performance Prediction and Mechanism Analysis

This example demonstrates how to use the ballistic performance prediction
system to predict V50 velocities and analyze the mechanisms driving performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ceramic_discovery.ballistics import (
    BallisticPredictor,
    MechanismAnalyzer,
)
from ceramic_discovery.ml.model_trainer import ModelTrainer


def generate_sample_data():
    """Generate sample training data for demonstration."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic material properties
    data = {
        'hardness': np.random.uniform(20, 35, n_samples),
        'fracture_toughness': np.random.uniform(3, 6, n_samples),
        'density': np.random.uniform(2.5, 4.0, n_samples),
        'thermal_conductivity_1000C': np.random.uniform(30, 60, n_samples),
        'youngs_modulus': np.random.uniform(350, 450, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Generate synthetic V50 values (correlated with properties)
    v50 = (
        20 * X['hardness'] +
        50 * X['fracture_toughness'] +
        100 * X['density'] +
        2 * X['thermal_conductivity_1000C'] +
        0.5 * X['youngs_modulus'] +
        np.random.normal(0, 50, n_samples)
    )
    
    y = pd.Series(v50, name='v50')
    
    return X, y


def main():
    """Run ballistic prediction example."""
    print("=" * 80)
    print("Ballistic Performance Prediction Example")
    print("=" * 80)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample training data...")
    X, y = generate_sample_data()
    print(f"  Generated {len(X)} material samples")
    print()
    
    # Step 2: Split data
    print("Step 2: Splitting data into train/test sets...")
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()
    
    # Step 3: Train ML model
    print("Step 3: Training ML model for V50 prediction...")
    trainer = ModelTrainer(random_seed=42)
    model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    print(f"  Model R² score: {model.metrics.r2_score:.3f}")
    print(f"  Model RMSE: {model.metrics.rmse:.1f} m/s")
    print()
    
    # Step 4: Create ballistic predictor
    print("Step 4: Creating ballistic predictor...")
    predictor = BallisticPredictor(model=model, random_seed=42)
    predictor.set_training_data(X_train, y_train)
    print("  Predictor initialized with trained model")
    print()
    
    # Step 5: Make predictions with uncertainty
    print("Step 5: Making predictions with uncertainty quantification...")
    print()
    
    # Predict for first 3 test samples
    for i in range(min(3, len(X_test))):
        material = X_test.iloc[[i]]
        prediction = predictor.predict_v50(material)
        
        print(f"  Material {i+1}:")
        print(f"    Properties:")
        for prop, value in material.iloc[0].items():
            print(f"      {prop}: {value:.2f}")
        print(f"    {prediction.format_prediction()}")
        print(f"    True V50: {y_test.iloc[i]:.1f} m/s")
        print(f"    Physical bounds satisfied: {prediction.physical_bounds_satisfied}")
        print()
    
    # Step 6: Validate prediction accuracy
    print("Step 6: Validating prediction accuracy...")
    accuracy_metrics = predictor.validate_prediction_accuracy(X_test, y_test)
    print(f"  R² score: {accuracy_metrics['r2_score']:.3f}")
    print(f"  RMSE: {accuracy_metrics['rmse']:.1f} m/s")
    print(f"  MAE: {accuracy_metrics['mae']:.1f} m/s")
    print(f"  Coverage (95% CI): {accuracy_metrics['coverage']:.1%}")
    print(f"  Mean reliability: {accuracy_metrics['mean_reliability']:.3f}")
    print()
    
    # Step 7: Analyze mechanisms
    print("Step 7: Analyzing ballistic performance mechanisms...")
    analyzer = MechanismAnalyzer(random_seed=42)
    
    # Thermal conductivity analysis
    thermal_analysis = analyzer.analyze_thermal_conductivity_impact(X_test, y_test)
    print(f"  Thermal Conductivity Analysis:")
    print(f"    Correlation with V50: {thermal_analysis.correlation_with_v50:.3f}")
    print(f"    P-value: {thermal_analysis.p_value:.4f}")
    print(f"    Contribution score: {thermal_analysis.contribution_score:.3f}")
    print(f"    Significant: {thermal_analysis.is_significant()}")
    print()
    
    # Property contributions
    print("  Property Contributions:")
    contributions = analyzer.analyze_property_contributions(
        X_test, y_test,
        model.feature_importances,
        model.feature_names
    )
    print(contributions[['property', 'pearson_correlation', 'effect_size', 'significant']].head())
    print()
    
    # Property interactions
    print("  Top Property Interactions:")
    interactions = analyzer.analyze_property_interactions(X_test, y_test, top_n=3)
    for i, interaction in enumerate(interactions, 1):
        print(f"    {i}. {interaction.property1} × {interaction.property2}")
        print(f"       Interaction strength: {interaction.interaction_strength:.3f}")
        print(f"       Synergy score: {interaction.synergy_score:.3f}")
        print(f"       Synergistic: {interaction.is_synergistic()}")
    print()
    
    # Hypothesis tests
    print("  Hypothesis Tests:")
    hypotheses = analyzer.test_performance_hypotheses(X_test, y_test)
    for hypothesis in hypotheses:
        print(f"    • {hypothesis.hypothesis}")
        print(f"      {hypothesis.conclusion}")
        print(f"      Significant: {hypothesis.is_significant()}")
        print()
    
    # Step 8: Generate comprehensive report
    print("Step 8: Generating comprehensive mechanism report...")
    report = analyzer.generate_mechanism_report(
        X_test, y_test,
        model.feature_importances,
        model.feature_names
    )
    print(f"  Report generated with {len(report)} sections")
    print(f"  Summary statistics:")
    for key, value in report['summary'].items():
        print(f"    {key}: {value}")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
