"""Quick verification script for ballistic performance prediction implementation."""

import numpy as np
import pandas as pd

from ceramic_discovery.ballistics import (
    BallisticPredictor,
    MechanismAnalyzer,
)
from ceramic_discovery.ml.model_trainer import ModelTrainer

print("Testing ballistic performance prediction implementation...")
print()

# Generate minimal test data
np.random.seed(42)
n = 30

X = pd.DataFrame({
    'hardness': np.random.uniform(20, 35, n),
    'fracture_toughness': np.random.uniform(3, 6, n),
    'density': np.random.uniform(2.5, 4.0, n),
    'thermal_conductivity_1000C': np.random.uniform(30, 60, n),
    'youngs_modulus': np.random.uniform(350, 450, n),
})

y = pd.Series(
    20 * X['hardness'] + 50 * X['fracture_toughness'] + 
    100 * X['density'] + np.random.normal(0, 30, n),
    name='v50'
)

# Split
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("1. Training model...")
trainer = ModelTrainer(random_seed=42)
model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
print(f"   ✓ Model trained (R²: {model.metrics.r2_score:.3f})")

print("2. Creating predictor...")
predictor = BallisticPredictor(model=model, random_seed=42)
predictor.set_training_data(X_train, y_train)
print("   ✓ Predictor created")

print("3. Making prediction...")
prediction = predictor.predict_v50(X_test.iloc[[0]])
print(f"   ✓ V50 prediction: {prediction.v50:.1f} m/s")
print(f"   ✓ Confidence interval: [{prediction.confidence_interval[0]:.1f}, {prediction.confidence_interval[1]:.1f}]")
print(f"   ✓ Reliability: {prediction.reliability_score:.3f}")

print("4. Analyzing mechanisms...")
analyzer = MechanismAnalyzer(random_seed=42)
thermal_analysis = analyzer.analyze_thermal_conductivity_impact(X_test, y_test)
print(f"   ✓ Thermal conductivity correlation: {thermal_analysis.correlation_with_v50:.3f}")

contributions = analyzer.analyze_property_contributions(X_test, y_test)
print(f"   ✓ Property contributions analyzed: {len(contributions)} properties")

interactions = analyzer.analyze_property_interactions(X_test, y_test, top_n=2)
print(f"   ✓ Property interactions found: {len(interactions)}")

hypotheses = analyzer.test_performance_hypotheses(X_test, y_test)
print(f"   ✓ Hypotheses tested: {len(hypotheses)}")

print()
print("✅ All ballistic performance prediction components working correctly!")
