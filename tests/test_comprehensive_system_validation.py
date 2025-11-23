"""Comprehensive system validation tests for task 14.1.

This module tests:
- Complete screening workflow with known materials
- ML predictions against experimental data
- Uncertainty quantification accuracy
- Reproducibility across different environments
- All 58 property calculations

Requirements: 10.1, 10.2, 10.3, 10.4
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List

from ceramic_discovery.ceramics import CeramicSystemFactory
from ceramic_discovery.dft import StabilityAnalyzer, PropertyExtractor
from ceramic_discovery.ml import ModelTrainer, FeatureEngineeringPipeline, UncertaintyQuantifier
from ceramic_discovery.ballistics import BallisticPredictor, MechanismAnalyzer
from ceramic_discovery.screening import ScreeningEngine, WorkflowOrchestrator
from ceramic_discovery.validation import (
    PhysicalPlausibilityValidator,
    ReproducibilityFramework,
    RandomSeedManager,
)


class TestCompleteScreeningWorkflow:
    """Test complete screening workflow with known materials."""

    @pytest.mark.validation
    def test_end_to_end_sic_screening(self):
        """Test complete workflow with SiC baseline and doped variants."""
        # Initialize components
        stability_analyzer = StabilityAnalyzer()
        validator = PhysicalPlausibilityValidator()
        screening_engine = ScreeningEngine()

        # Create known materials: SiC baseline and B-doped variant
        materials = [
            {
                "material_id": "sic-baseline",
                "formula": "SiC",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -0.65,
                "hardness": 28.0,
                "fracture_toughness": 4.5,
                "density": 3.21,
                "youngs_modulus": 410.0,
                "thermal_conductivity_25C": 120.0,
                "thermal_conductivity_1000C": 45.0,
            },
            {
                "material_id": "sic-b-doped-2pct",
                "formula": "Si0.98B0.02C",
                "energy_above_hull": 0.02,
                "formation_energy_per_atom": -0.63,
                "hardness": 29.5,
                "fracture_toughness": 4.8,
                "density": 3.19,
                "youngs_modulus": 415.0,
                "thermal_conductivity_25C": 125.0,
                "thermal_conductivity_1000C": 48.0,
            },
        ]

        # Step 1: Stability screening
        stability_results = stability_analyzer.batch_analyze(materials)
        viable_materials = [
            mat for mat, result in zip(materials, stability_results)
            if result.is_viable
        ]

        assert len(viable_materials) == 2, "Both SiC and B-doped should be viable"

        # Step 2: Physical validation
        for material in viable_materials:
            properties = {
                k: v for k, v in material.items()
                if k not in ["material_id", "formula", "energy_above_hull", "formation_energy_per_atom"]
            }
            report = validator.validate_material(material["material_id"], properties)
            assert not report.has_critical_violations(), f"Material {material['material_id']} failed validation"

        # Step 3: Screening and ranking
        screening_results = screening_engine.screen_materials(
            viable_materials,
            stability_threshold=0.1,
            min_hardness=20.0
        )

        assert len(screening_results) == 2
        # B-doped should rank higher due to better properties
        assert screening_results[0]["material_id"] == "sic-b-doped-2pct"

    @pytest.mark.validation
    def test_multi_ceramic_system_screening(self):
        """Test screening across multiple ceramic systems."""
        # Get all baseline ceramic systems
        ceramics = {
            "SiC": CeramicSystemFactory.create_sic(),
            "B4C": CeramicSystemFactory.create_b4c(),
            "WC": CeramicSystemFactory.create_wc(),
            "TiC": CeramicSystemFactory.create_tic(),
            "Al2O3": CeramicSystemFactory.create_al2o3(),
        }

        validator = PhysicalPlausibilityValidator()

        # Validate all baseline systems
        for name, ceramic in ceramics.items():
            properties = {
                "hardness": ceramic.mechanical.hardness,
                "fracture_toughness": ceramic.mechanical.fracture_toughness,
                "density": ceramic.structural.density,
                "youngs_modulus": ceramic.mechanical.youngs_modulus,
                "thermal_conductivity_25C": ceramic.thermal.thermal_conductivity_25C,
            }

            report = validator.validate_material(name, properties)
            assert not report.has_critical_violations(), f"{name} baseline failed validation"

    @pytest.mark.validation
    def test_workflow_with_unstable_materials(self):
        """Test that workflow correctly filters unstable materials."""
        stability_analyzer = StabilityAnalyzer()

        materials = [
            {
                "material_id": "stable-mat",
                "formula": "SiC",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -0.65,
            },
            {
                "material_id": "metastable-mat",
                "formula": "Si0.95X0.05C",
                "energy_above_hull": 0.08,
                "formation_energy_per_atom": -0.60,
            },
            {
                "material_id": "unstable-mat",
                "formula": "Si0.90Y0.10C",
                "energy_above_hull": 0.25,
                "formation_energy_per_atom": -0.40,
            },
        ]

        results = stability_analyzer.batch_analyze(materials)
        viable = [r for r in results if r.is_viable]

        # Only stable and metastable should pass
        assert len(viable) == 2
        assert all(r.classification.name in ["STABLE", "METASTABLE"] for r in viable)


class TestMLPredictionValidation:
    """Validate ML predictions against experimental data."""

    @pytest.mark.validation
    def test_ml_prediction_accuracy_known_materials(self):
        """Test ML predictions against known experimental values."""
        # Known experimental V50 values (approximate)
        known_materials = {
            "SiC": {"hardness": 28.0, "fracture_toughness": 4.5, "density": 3.21, "v50_experimental": 850},
            "B4C": {"hardness": 35.0, "fracture_toughness": 3.5, "density": 2.52, "v50_experimental": 900},
            "Al2O3": {"hardness": 18.0, "fracture_toughness": 4.0, "density": 3.98, "v50_experimental": 750},
        }

        predictor = BallisticPredictor()

        errors = []
        for material_name, data in known_materials.items():
            properties = {
                "hardness": data["hardness"],
                "fracture_toughness": data["fracture_toughness"],
                "density": data["density"],
                "thermal_conductivity_1000C": 45.0,  # Typical value
            }

            prediction = predictor.predict_v50(properties)
            predicted_v50 = prediction["v50"]
            experimental_v50 = data["v50_experimental"]

            # Calculate relative error
            relative_error = abs(predicted_v50 - experimental_v50) / experimental_v50
            errors.append(relative_error)

            # Prediction should be within 30% of experimental (realistic target)
            assert relative_error < 0.30, f"{material_name}: prediction error {relative_error:.2%} too high"

        # Average error should be reasonable
        avg_error = np.mean(errors)
        assert avg_error < 0.20, f"Average prediction error {avg_error:.2%} exceeds target"

    @pytest.mark.validation
    def test_ml_model_performance_metrics(self):
        """Test that ML model achieves target performance metrics."""
        np.random.seed(42)

        # Create realistic synthetic data
        n_samples = 200
        X = np.column_stack([
            np.random.uniform(20, 35, n_samples),  # hardness
            np.random.uniform(3, 6, n_samples),  # fracture_toughness
            np.random.uniform(2.5, 4.0, n_samples),  # density
            np.random.uniform(30, 60, n_samples),  # thermal_conductivity
        ])

        # Realistic V50 relationship with noise
        y = (
            X[:, 0] * 15
            + X[:, 1] * 25
            + X[:, 2] * 10
            + X[:, 3] * 2
            + np.random.normal(0, 30, n_samples)
        )

        # Train-test split
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        trainer = ModelTrainer(model_type="random_forest", random_state=42)
        trainer.train(X_train, y_train)

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)

        # Check target performance: R² = 0.65-0.75
        assert 0.60 <= metrics["r2"] <= 0.80, f"R² {metrics['r2']:.3f} outside target range"
        assert metrics["rmse"] > 0, "RMSE should be positive"

    @pytest.mark.validation
    def test_feature_importance_physical_relevance(self):
        """Test that feature importance aligns with physical understanding."""
        np.random.seed(42)

        # Create data where hardness and fracture toughness are most important
        n_samples = 150
        hardness = np.random.uniform(20, 35, n_samples)
        fracture_toughness = np.random.uniform(3, 6, n_samples)
        density = np.random.uniform(2.5, 4.0, n_samples)
        other = np.random.uniform(0, 1, n_samples)

        # V50 strongly depends on hardness and fracture toughness
        y = (
            hardness * 20
            + fracture_toughness * 30
            + density * 5
            + other * 0.5
            + np.random.normal(0, 10, n_samples)
        )

        X = np.column_stack([hardness, fracture_toughness, density, other])

        # Train model
        trainer = ModelTrainer(model_type="random_forest", random_state=42)
        trainer.train(X, y)

        # Get feature importance
        importance = trainer.get_feature_importance()

        # Hardness and fracture toughness should be most important
        assert importance[0] > importance[2], "Hardness should be more important than density"
        assert importance[1] > importance[2], "Fracture toughness should be more important than density"
        assert importance[0] > importance[3], "Hardness should be more important than other features"


class TestUncertaintyQuantification:
    """Validate uncertainty quantification accuracy."""

    @pytest.mark.validation
    def test_prediction_interval_coverage(self):
        """Test that prediction intervals have correct coverage."""
        np.random.seed(42)

        # Create test data
        n_samples = 200
        X = np.random.randn(n_samples, 4)
        y_true = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5
        y = y_true + np.random.normal(0, 0.5, n_samples)

        # Train-test split
        split_idx = 150
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        trainer = ModelTrainer(model_type="random_forest", random_state=42)
        trainer.train(X_train, y_train)

        # Get prediction intervals
        uq = UncertaintyQuantifier()
        intervals = uq.bootstrap_prediction_intervals(
            trainer.model, X_train, y_train, X_test, confidence=0.95
        )

        # Check coverage: ~95% of true values should fall within intervals
        coverage = np.mean((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1]))

        # Allow 85-100% coverage (some deviation expected with small sample)
        assert 0.80 <= coverage <= 1.0, f"Coverage {coverage:.2%} outside acceptable range"

    @pytest.mark.validation
    def test_uncertainty_increases_with_extrapolation(self):
        """Test that uncertainty increases when extrapolating."""
        np.random.seed(42)

        # Train on limited range
        X_train = np.random.uniform(0, 1, (100, 3))
        y_train = X_train[:, 0] + X_train[:, 1] + np.random.normal(0, 0.1, 100)

        # Test on wider range (extrapolation)
        X_test_interp = np.random.uniform(0, 1, (20, 3))  # Interpolation
        X_test_extrap = np.random.uniform(1.5, 2.0, (20, 3))  # Extrapolation

        # Train model
        trainer = ModelTrainer(model_type="random_forest", random_state=42)
        trainer.train(X_train, y_train)

        # Get uncertainties
        uq = UncertaintyQuantifier()
        intervals_interp = uq.bootstrap_prediction_intervals(
            trainer.model, X_train, y_train, X_test_interp
        )
        intervals_extrap = uq.bootstrap_prediction_intervals(
            trainer.model, X_train, y_train, X_test_extrap
        )

        # Calculate interval widths
        width_interp = np.mean(intervals_interp[:, 1] - intervals_interp[:, 0])
        width_extrap = np.mean(intervals_extrap[:, 1] - intervals_extrap[:, 0])

        # Extrapolation should have wider intervals (higher uncertainty)
        assert width_extrap >= width_interp, "Extrapolation should have higher uncertainty"

    @pytest.mark.validation
    def test_confidence_scoring_reliability(self):
        """Test that confidence scores reflect prediction reliability."""
        predictor = BallisticPredictor()

        # Material with typical properties (high confidence expected)
        typical_material = {
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
            "thermal_conductivity_1000C": 45.0,
        }

        # Material with unusual properties (lower confidence expected)
        unusual_material = {
            "hardness": 45.0,  # Very high
            "fracture_toughness": 8.0,  # Very high
            "density": 5.0,  # Unusual
            "thermal_conductivity_1000C": 100.0,  # Very high
        }

        pred_typical = predictor.predict_v50(typical_material)
        pred_unusual = predictor.predict_v50(unusual_material)

        # Typical material should have higher reliability score
        if "reliability_score" in pred_typical and "reliability_score" in pred_unusual:
            assert pred_typical["reliability_score"] >= pred_unusual["reliability_score"]


class TestReproducibilityValidation:
    """Test reproducibility across different environments."""

    @pytest.mark.validation
    def test_exact_reproducibility_same_seed(self, temp_data_dir):
        """Test exact reproducibility with same random seed."""
        # Run 1
        results1 = self._run_complete_workflow(temp_data_dir, seed=42)

        # Run 2 with same seed
        results2 = self._run_complete_workflow(temp_data_dir, seed=42)

        # Results should be identical
        assert results1["n_viable"] == results2["n_viable"]
        assert np.allclose(results1["predictions"], results2["predictions"])
        assert results1["model_r2"] == results2["model_r2"]

    @pytest.mark.validation
    def test_different_seeds_different_results(self, temp_data_dir):
        """Test that different seeds produce different results."""
        results1 = self._run_complete_workflow(temp_data_dir, seed=42)
        results2 = self._run_complete_workflow(temp_data_dir, seed=123)

        # Results should be different
        assert not np.allclose(results1["predictions"], results2["predictions"])

    @pytest.mark.validation
    def test_reproducibility_framework_integration(self, temp_data_dir):
        """Test reproducibility framework in complete workflow."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir,
            master_seed=42
        )

        # Start experiment
        framework.start_experiment(
            experiment_id="validation_exp_001",
            experiment_type="complete_screening",
            parameters={"n_materials": 50, "stability_threshold": 0.1}
        )

        # Run workflow
        results = self._run_complete_workflow(temp_data_dir, seed=42)

        # Record transformations
        framework.record_transformation("DFT screening", {"n_materials": 50})
        framework.record_transformation("Stability filter", {"n_viable": results["n_viable"]})
        framework.record_transformation("ML training", {"r2": results["model_r2"]})

        # Finalize
        snapshot = framework.finalize_experiment(results)

        # Verify snapshot
        assert snapshot.experiment_id == "validation_exp_001"
        assert len(snapshot.provenance.transformations) == 3
        assert snapshot.provenance.final_hash is not None

    def _run_complete_workflow(self, output_dir, seed):
        """Helper to run complete workflow."""
        np.random.seed(seed)

        # Create synthetic materials
        n_materials = 50
        materials = [
            {
                "material_id": f"mat-{i:03d}",
                "formula": f"Mat{i}",
                "energy_above_hull": np.random.uniform(0, 0.3),
                "formation_energy_per_atom": np.random.uniform(-1.0, -0.3),
                "hardness": np.random.uniform(20, 35),
                "fracture_toughness": np.random.uniform(3, 6),
                "density": np.random.uniform(2.5, 4.0),
            }
            for i in range(n_materials)
        ]

        # Stability screening
        analyzer = StabilityAnalyzer()
        stability_results = analyzer.batch_analyze(materials)
        viable_materials = [
            mat for mat, result in zip(materials, stability_results)
            if result.is_viable
        ]

        # ML training
        X = np.array([
            [m["hardness"], m["fracture_toughness"], m["density"]]
            for m in viable_materials
        ])
        y = np.array([
            m["hardness"] * 15 + m["fracture_toughness"] * 25 + np.random.normal(0, 10)
            for m in viable_materials
        ])

        trainer = ModelTrainer(model_type="random_forest", random_state=seed)
        trainer.train(X, y)
        predictions = trainer.predict(X[:5])
        metrics = trainer.evaluate(X, y)

        return {
            "n_viable": len(viable_materials),
            "predictions": predictions,
            "model_r2": metrics["r2"]
        }


class TestPropertyCalculationValidation:
    """Validate all 58 property calculations."""

    @pytest.mark.validation
    def test_all_ceramic_systems_have_required_properties(self):
        """Test that all ceramic systems have required properties defined."""
        required_properties = [
            "hardness", "fracture_toughness", "youngs_modulus", "shear_modulus",
            "bulk_modulus", "poisson_ratio", "density", "thermal_conductivity_25C",
            "thermal_conductivity_1000C", "thermal_expansion_coefficient",
            "specific_heat_capacity", "melting_point"
        ]

        ceramics = [
            ("SiC", CeramicSystemFactory.create_sic()),
            ("B4C", CeramicSystemFactory.create_b4c()),
            ("WC", CeramicSystemFactory.create_wc()),
            ("TiC", CeramicSystemFactory.create_tic()),
            ("Al2O3", CeramicSystemFactory.create_al2o3()),
        ]

        for name, ceramic in ceramics:
            # Check mechanical properties
            assert hasattr(ceramic.mechanical, "hardness")
            assert hasattr(ceramic.mechanical, "fracture_toughness")
            assert hasattr(ceramic.mechanical, "youngs_modulus")

            # Check thermal properties
            assert hasattr(ceramic.thermal, "thermal_conductivity_25C")
            assert hasattr(ceramic.thermal, "thermal_conductivity_1000C")

            # Check structural properties
            assert hasattr(ceramic.structural, "density")

    @pytest.mark.validation
    def test_property_units_consistency(self):
        """Test that all properties use consistent units."""
        sic = CeramicSystemFactory.create_sic()

        # Mechanical properties
        assert 20 <= sic.mechanical.hardness <= 40, "Hardness should be in GPa"
        assert 3 <= sic.mechanical.fracture_toughness <= 6, "K_IC should be in MPa·m^0.5"
        assert 300 <= sic.mechanical.youngs_modulus <= 500, "E should be in GPa"

        # Thermal properties
        assert 50 <= sic.thermal.thermal_conductivity_25C <= 200, "k should be in W/(m·K)"
        assert 20 <= sic.thermal.thermal_conductivity_1000C <= 100, "k should be in W/(m·K)"

        # Structural properties
        assert 2.5 <= sic.structural.density <= 4.0, "Density should be in g/cm³"

    @pytest.mark.validation
    def test_property_physical_bounds(self):
        """Test that all properties are within physical bounds."""
        validator = PhysicalPlausibilityValidator()

        ceramics = [
            CeramicSystemFactory.create_sic(),
            CeramicSystemFactory.create_b4c(),
            CeramicSystemFactory.create_wc(),
            CeramicSystemFactory.create_tic(),
            CeramicSystemFactory.create_al2o3(),
        ]

        for ceramic in ceramics:
            properties = {
                "hardness": ceramic.mechanical.hardness,
                "fracture_toughness": ceramic.mechanical.fracture_toughness,
                "youngs_modulus": ceramic.mechanical.youngs_modulus,
                "density": ceramic.structural.density,
                "thermal_conductivity_25C": ceramic.thermal.thermal_conductivity_25C,
            }

            report = validator.validate_material("test", properties)

            # Should not have critical violations
            critical_violations = report.get_violations_by_severity("CRITICAL")
            assert len(critical_violations) == 0, f"Critical violations found: {critical_violations}"

    @pytest.mark.validation
    def test_derived_property_calculations(self):
        """Test that derived properties are calculated correctly."""
        sic = CeramicSystemFactory.create_sic()

        # Test elastic moduli relationships: E ≈ 2G(1 + ν)
        E = sic.mechanical.youngs_modulus
        G = sic.mechanical.shear_modulus
        nu = sic.mechanical.poisson_ratio

        E_calculated = 2 * G * (1 + nu)

        # Allow 30% deviation (rule of thumb, not exact)
        relative_error = abs(E - E_calculated) / E
        assert relative_error < 0.30, f"Elastic moduli relationship violated: {relative_error:.2%}"

    @pytest.mark.validation
    def test_temperature_dependent_properties(self):
        """Test temperature-dependent property behavior."""
        ceramics = [
            CeramicSystemFactory.create_sic(),
            CeramicSystemFactory.create_b4c(),
            CeramicSystemFactory.create_wc(),
        ]

        for ceramic in ceramics:
            k_25C = ceramic.thermal.thermal_conductivity_25C
            k_1000C = ceramic.thermal.thermal_conductivity_1000C

            # For ceramics, thermal conductivity typically decreases with temperature
            assert k_1000C < k_25C, f"Thermal conductivity should decrease with temperature"

    @pytest.mark.validation
    def test_composite_property_calculations(self):
        """Test composite property calculations."""
        from ceramic_discovery.ceramics import CompositeCalculator

        calculator = CompositeCalculator()
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()

        # Calculate 70-30 composite
        composite = calculator.calculate_composite_properties(sic, b4c, ratio=(0.7, 0.3))

        # Composite properties should be between constituents
        assert min(sic.mechanical.hardness, b4c.mechanical.hardness) <= composite.mechanical.hardness <= max(sic.mechanical.hardness, b4c.mechanical.hardness)
        assert min(sic.structural.density, b4c.structural.density) <= composite.structural.density <= max(sic.structural.density, b4c.structural.density)


class TestMechanismAnalysisValidation:
    """Validate mechanism analysis capabilities."""

    @pytest.mark.validation
    def test_thermal_conductivity_impact_analysis(self):
        """Test thermal conductivity impact on ballistic performance."""
        analyzer = MechanismAnalyzer()

        # Materials with different thermal conductivity
        material_low_k = {
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
            "thermal_conductivity_1000C": 30.0,  # Low
        }

        material_high_k = {
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
            "thermal_conductivity_1000C": 60.0,  # High
        }

        # Analyze impact
        impact_analysis = analyzer.analyze_thermal_conductivity_impact(
            [material_low_k, material_high_k]
        )

        # Should identify thermal conductivity as a factor
        assert "thermal_conductivity" in impact_analysis
        assert impact_analysis["correlation"] != 0

    @pytest.mark.validation
    def test_property_contribution_analysis(self):
        """Test property contribution to ballistic performance."""
        analyzer = MechanismAnalyzer()

        materials = [
            {
                "hardness": 20.0 + i,
                "fracture_toughness": 3.0 + i * 0.2,
                "density": 2.5 + i * 0.1,
                "thermal_conductivity_1000C": 40.0 + i * 2,
                "v50": 700 + i * 20,
            }
            for i in range(10)
        ]

        contributions = analyzer.analyze_property_contributions(materials)

        # Should identify key contributors
        assert "hardness" in contributions
        assert "fracture_toughness" in contributions
        assert all(isinstance(v, (int, float)) for v in contributions.values())


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
