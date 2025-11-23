# Ballistic Performance Prediction Module

This module provides comprehensive ballistic performance prediction and mechanism analysis for ceramic armor materials.

## Overview

The ballistic performance prediction system combines machine learning models with rigorous uncertainty quantification to predict V50 ballistic limit velocities and analyze the physical mechanisms driving performance.

## Components

### 1. BallisticPredictor

The main prediction engine that integrates ML models with uncertainty quantification.

**Key Features:**
- V50 prediction with confidence intervals
- Physical bounds enforcement (200-2000 m/s)
- Reliability scoring based on data quality and prediction uncertainty
- Batch prediction capabilities
- Prediction accuracy validation

**Example Usage:**
```python
from ceramic_discovery.ballistics import BallisticPredictor
from ceramic_discovery.ml.model_trainer import ModelTrainer

# Train model
trainer = ModelTrainer(random_seed=42)
model = trainer.train_random_forest(X_train, y_train, X_test, y_test)

# Create predictor
predictor = BallisticPredictor(model=model)
predictor.set_training_data(X_train, y_train)

# Make prediction
prediction = predictor.predict_v50(material_properties)
print(prediction.format_prediction())
# Output: V50: 800.0 m/s (95% CI: [750.0, 850.0] m/s, reliability: 0.85)
```

### 2. MechanismAnalyzer

Advanced mechanism analysis system for understanding performance drivers.

**Key Features:**
- Thermal conductivity impact analysis
- Property contribution decomposition
- Property interaction analysis (synergistic/antagonistic effects)
- Statistical hypothesis testing
- Comprehensive mechanism reporting

**Example Usage:**
```python
from ceramic_discovery.ballistics import MechanismAnalyzer

analyzer = MechanismAnalyzer(random_seed=42)

# Analyze thermal conductivity impact
thermal_analysis = analyzer.analyze_thermal_conductivity_impact(materials, v50_values)
print(f"Correlation: {thermal_analysis.correlation_with_v50:.3f}")
print(f"Mechanism: {thermal_analysis.mechanism_description}")

# Analyze property contributions
contributions = analyzer.analyze_property_contributions(materials, v50_values)
print(contributions[['property', 'effect_size', 'significant']])

# Test hypotheses
hypotheses = analyzer.test_performance_hypotheses(materials, v50_values)
for h in hypotheses:
    print(f"{h.hypothesis}: {h.conclusion}")
```

## Data Classes

### BallisticPrediction
Contains V50 prediction with uncertainty estimates:
- `v50`: Predicted V50 velocity (m/s)
- `confidence_interval`: 95% confidence interval
- `uncertainty`: Standard deviation of prediction
- `reliability_score`: 0-1 reliability indicator
- `physical_bounds_satisfied`: Bounds check result

### ThermalConductivityAnalysis
Specific analysis of thermal conductivity impact:
- `correlation_with_v50`: Correlation coefficient
- `p_value`: Statistical significance
- `contribution_score`: Relative contribution (0-1)
- `mechanism_description`: Physical mechanism explanation

### PropertyInteraction
Analysis of property interactions:
- `property1`, `property2`: Interacting properties
- `interaction_strength`: Correlation between properties
- `combined_effect_on_v50`: Combined contribution
- `synergy_score`: Synergistic (+) or antagonistic (-) effect

### HypothesisTest
Statistical hypothesis test result:
- `hypothesis`: Hypothesis statement
- `test_statistic`: Statistical test value
- `p_value`: Significance level
- `conclusion`: Human-readable conclusion

## Requirements Addressed

This implementation addresses the following requirements from the design document:

**Requirement 3.2**: Machine learning model integration for V50 prediction
- ✓ Integrated with trained ML models
- ✓ Realistic R² targets (0.65-0.75)
- ✓ Proper cross-validation

**Requirement 3.3**: Uncertainty estimates through bootstrap sampling
- ✓ Bootstrap-based confidence intervals
- ✓ Prediction uncertainty quantification
- ✓ Reliability scoring

**Requirement 4.3**: Confidence intervals and reliability scores for predictions
- ✓ 95% confidence intervals for all predictions
- ✓ Multi-factor reliability scoring
- ✓ Physical bounds enforcement

**Requirement 4.4**: Property contribution analysis
- ✓ Feature importance integration
- ✓ Correlation analysis
- ✓ Effect size calculation
- ✓ Statistical significance testing

**Requirement 4.5**: Mechanism analysis
- ✓ Thermal conductivity impact analysis
- ✓ Property interaction analysis
- ✓ Synergy detection

**Requirement 8.2**: Hypothesis testing for performance mechanisms
- ✓ Multiple hypothesis tests
- ✓ Statistical significance evaluation
- ✓ Comprehensive reporting

## Physical Bounds

The predictor enforces realistic physical bounds on V50 predictions:
- **Minimum V50**: 200 m/s (lower bound for ceramic armor)
- **Maximum V50**: 2000 m/s (upper bound for ceramic armor)

Predictions outside these bounds are flagged and optionally clipped to valid ranges.

## Uncertainty Quantification

The system uses bootstrap sampling (1000 iterations by default) to generate prediction intervals:
1. Resample training data with replacement
2. Train model on bootstrap sample
3. Make prediction on test data
4. Repeat N times
5. Calculate percentiles for confidence intervals

## Reliability Scoring

Reliability scores (0-1) are calculated based on:
- **Distance to training data** (40%): Closer = more reliable
- **Prediction uncertainty** (40%): Lower uncertainty = more reliable
- **Data quality** (20%): Complete properties = more reliable

## Testing

Comprehensive test coverage includes:
- Unit tests for all data classes
- Integration tests for prediction pipeline
- Mechanism analysis validation
- Physical bounds enforcement
- Hypothesis testing verification

Run tests with:
```bash
pytest tests/test_ballistic_predictor.py tests/test_mechanism_analyzer.py -v
```

## Performance

The system is designed for efficient batch processing:
- Vectorized operations where possible
- Cached training data for uncertainty quantification
- Parallel-ready architecture (future enhancement)

## Future Enhancements

Potential improvements:
1. GPU acceleration for large-scale predictions
2. Active learning for model improvement
3. Multi-fidelity modeling (DFT + experiments)
4. Bayesian optimization for material design
5. Real-time prediction API

## References

- Design Document: `.kiro/specs/design.md`
- Requirements: `.kiro/specs/requirements.md`
- Examples: `examples/ballistic_prediction_example.py`
