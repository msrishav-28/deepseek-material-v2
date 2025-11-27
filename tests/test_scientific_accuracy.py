"""Scientific accuracy and correctness tests."""

import pytest
import numpy as np

from ceramic_discovery.dft import StabilityAnalyzer, UnitConverter
from ceramic_discovery.ceramics import (
    CeramicSystemFactory,
    CompositeCalculator,
)
from ceramic_discovery.ml import FeatureEngineeringPipeline, ModelTrainer
from ceramic_discovery.ballistics import BallisticPredictor
from ceramic_discovery.validation import PhysicalPlausibilityValidator


class TestDFTCriteriaCorrectness:
    """Test correctness of DFT stability criteria."""

    @pytest.mark.scientific
    def test_correct_stability_threshold(self):
        """CRITICAL: Verify correct 0.1 eV/atom threshold is used."""
        analyzer = StabilityAnalyzer()

        # The CORRECT threshold is 0.1 eV/atom for metastable materials
        assert analyzer.metastable_threshold == 0.1

        # Test boundary cases
        assert analyzer.is_viable(0.0) is True  # Stable
        assert analyzer.is_viable(0.05) is True  # Metastable, viable
        assert analyzer.is_viable(0.1) is True  # At threshold, viable
        assert analyzer.is_viable(0.101) is False  # Above threshold, not viable
        assert analyzer.is_viable(0.2) is False  # Unstable

    @pytest.mark.scientific
    def test_stability_classification_accuracy(self):
        """Test accuracy of stability classification."""
        analyzer = StabilityAnalyzer()

        # Test cases with known classifications
        test_cases = [
            # (energy_above_hull, expected_viable, expected_classification)
            (0.0, True, "STABLE"),
            (0.001, True, "METASTABLE"),
            (0.05, True, "METASTABLE"),
            (0.099, True, "METASTABLE"),
            (0.1, True, "METASTABLE"),
            (0.11, False, "UNSTABLE"),
            (0.5, False, "UNSTABLE"),
        ]

        for e_hull, expected_viable, expected_class in test_cases:
            result = analyzer.analyze_stability(
                material_id=f"test-{e_hull}",
                formula="Test",
                energy_above_hull=e_hull,
                formation_energy_per_atom=-0.5,
            )

            assert result.is_viable == expected_viable, (
                f"Failed viability check for ΔE_hull={e_hull}"
            )
            assert result.classification.name == expected_class, (
                f"Failed classification for ΔE_hull={e_hull}"
            )

    @pytest.mark.scientific
    def test_formation_energy_influence(self):
        """Test that formation energy influences confidence correctly."""
        analyzer = StabilityAnalyzer()

        # More negative formation energy should increase confidence
        result_weak = analyzer.analyze_stability(
            "test1", "Test", 0.05, -0.3
        )
        result_strong = analyzer.analyze_stability(
            "test2", "Test", 0.05, -1.5
        )

        assert result_strong.confidence_score > result_weak.confidence_score


class TestUnitConsistency:
    """Test unit consistency and conversions."""

    @pytest.mark.scientific
    def test_fracture_toughness_units(self):
        """CRITICAL: Verify fracture toughness uses correct units (MPa·m^0.5)."""
        sic = CeramicSystemFactory.create_sic()

        # SiC fracture toughness should be in MPa·m^0.5
        k_ic = sic.mechanical.fracture_toughness

        # Typical range for SiC: 3-6 MPa·m^0.5
        assert 3.0 <= k_ic <= 6.0, (
            f"SiC fracture toughness {k_ic} outside expected range"
        )

    @pytest.mark.scientific
    def test_unit_converter_accuracy(self):
        """Test unit converter accuracy."""
        converter = UnitConverter()

        # Test energy conversions
        ev_to_j = converter.convert(1.0, "eV", "J")
        assert abs(ev_to_j - 1.602176634e-19) < 1e-25

        # Test pressure conversions
        gpa_to_mpa = converter.convert(1.0, "GPa", "MPa")
        assert abs(gpa_to_mpa - 1000.0) < 1e-10

        # Test temperature conversions
        c_to_k = converter.convert(25.0, "C", "K")
        assert abs(c_to_k - 298.15) < 1e-10

    @pytest.mark.scientific
    def test_property_unit_consistency(self):
        """Test that all ceramic systems use consistent units."""
        systems = [
            CeramicSystemFactory.create_sic(),
            CeramicSystemFactory.create_b4c(),
            CeramicSystemFactory.create_wc(),
            CeramicSystemFactory.create_tic(),
            CeramicSystemFactory.create_al2o3(),
        ]

        for system in systems:
            # Hardness in GPa
            assert 10 <= system.mechanical.hardness <= 50

            # Fracture toughness in MPa·m^0.5
            assert 1 <= system.mechanical.fracture_toughness <= 15

            # Density in g/cm³
            assert 2 <= system.structural.density <= 20

            # Young's modulus in GPa
            assert 200 <= system.mechanical.youngs_modulus <= 700


class TestCompositeCalculationAccuracy:
    """Test composite property calculation accuracy."""

    @pytest.mark.scientific
    def test_rule_of_mixtures_bounds(self):
        """Test rule of mixtures produces values within bounds."""
        calculator = CompositeCalculator()

        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()

        # 70% SiC, 30% B4C
        composite = calculator.calculate_composite_properties(
            sic, b4c, ratio=(0.7, 0.3)
        )

        # Composite properties should be between constituent properties
        sic_hardness = sic.mechanical.hardness
        b4c_hardness = b4c.mechanical.hardness

        min_hardness = min(sic_hardness, b4c_hardness)
        max_hardness = max(sic_hardness, b4c_hardness)

        assert min_hardness <= composite.mechanical.hardness <= max_hardness

    @pytest.mark.scientific
    def test_composite_density_accuracy(self):
        """Test composite density calculation accuracy."""
        calculator = CompositeCalculator()

        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()

        ratio = (0.6, 0.4)
        composite = calculator.calculate_composite_properties(sic, b4c, ratio)

        # Density should follow rule of mixtures closely
        expected_density = (
            ratio[0] * sic.structural.density +
            ratio[1] * b4c.structural.density
        )

        # Allow 5% deviation
        assert abs(composite.structural.density - expected_density) < expected_density * 0.05

    @pytest.mark.scientific
    def test_composite_limitations_warning(self):
        """Test that composite calculations include limitation warnings."""
        calculator = CompositeCalculator()

        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()

        # Should warn about limitations
        with pytest.warns(UserWarning, match="rule-of-mixtures"):
            composite = calculator.calculate_composite_properties(
                sic, b4c, ratio=(0.5, 0.5)
            )


class TestMLModelAccuracy:
    """Test ML model accuracy and performance."""

    @pytest.mark.scientific
    def test_realistic_performance_targets(self):
        """Test that ML models achieve realistic R² targets (0.65-0.75)."""
        np.random.seed(42)

        # Create synthetic data with realistic noise
        n_samples = 200
        X = np.random.randn(n_samples, 5)

        # True relationship with noise
        y_true = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5
        noise = np.random.normal(0, 0.5, n_samples)
        y = y_true + noise

        # Train model
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X, y)

        # Evaluate
        score = trainer.evaluate(X, y)

        # Should achieve reasonable R² (not perfect due to noise)
        assert 0.5 <= score["r2"] <= 0.95

    @pytest.mark.scientific
    def test_feature_importance_physical_relevance(self):
        """Test that feature importance reflects physical relevance."""
        np.random.seed(42)

        # Create data where hardness is most important
        n_samples = 150
        hardness = np.random.uniform(20, 35, n_samples)
        density = np.random.uniform(2.5, 4.0, n_samples)
        other = np.random.uniform(0, 1, n_samples)

        # V50 strongly depends on hardness
        y = hardness * 20 + density * 2 + other * 0.1 + np.random.normal(0, 5, n_samples)

        X = np.column_stack([hardness, density, other])

        # Train model
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X, y)

        # Get feature importance
        importance = trainer.get_feature_importance()

        # Hardness should be most important
        assert importance[0] > importance[1]
        assert importance[0] > importance[2]

    @pytest.mark.scientific
    def test_uncertainty_quantification_coverage(self):
        """Test that uncertainty intervals have correct coverage."""
        np.random.seed(42)

        # Create test data
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.5, n_samples)

        # Train model
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X[:80], y[:80])

        # Get predictions with uncertainty
        from ceramic_discovery.ml import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        intervals = uq.bootstrap_prediction_intervals(
            trainer.model, X[:80], y[:80], X[80:], confidence=0.95
        )

        # Check coverage: ~95% of true values should fall within intervals
        y_test = y[80:]
        coverage = np.mean(
            (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
        )

        # Allow some deviation (85-100% coverage)
        assert 0.85 <= coverage <= 1.0


class TestFeatureEngineeringCorrectness:
    """Test feature engineering correctness."""

    @pytest.mark.scientific
    def test_tier_separation(self):
        """Test that only Tier 1-2 fundamental properties are used."""
        engineer = FeatureEngineeringPipeline()

        # Get allowed features
        tier1_features = engineer.get_tier1_features()
        tier2_features = engineer.get_tier2_features()

        # Tier 1: Fundamental DFT properties
        assert "formation_energy" in tier1_features
        assert "band_gap" in tier1_features
        assert "density" in tier1_features

        # Tier 2: Directly measurable properties
        assert "hardness" in tier2_features
        assert "youngs_modulus" in tier2_features
        assert "thermal_conductivity" in tier2_features

        # Should NOT include derived properties like V50
        all_features = tier1_features + tier2_features
        assert "v50" not in all_features
        assert "ballistic_performance" not in all_features

    @pytest.mark.scientific
    def test_no_circular_dependencies(self):
        """Test that features don't have circular dependencies."""
        engineer = FeatureEngineeringPipeline()

        # If predicting V50, should not use V50 or derived metrics as features
        features_for_v50 = engineer.get_features_for_target("v50")

        assert "v50" not in features_for_v50
        assert "ballistic_limit" not in features_for_v50

    @pytest.mark.scientific
    def test_feature_scaling_preserves_relationships(self):
        """Test that feature scaling preserves relationships."""
        engineer = FeatureEngineeringPipeline()

        # Original data
        data = {
            "hardness": np.array([20, 25, 30, 35]),
            "density": np.array([2.5, 3.0, 3.5, 4.0]),
        }

        # Scale features
        scaled = engineer.scale_features(data)

        # Relative ordering should be preserved
        assert np.all(np.diff(scaled["hardness"]) > 0)
        assert np.all(np.diff(scaled["density"]) > 0)


class TestBallisticPredictionAccuracy:
    """Test ballistic prediction accuracy."""

    @pytest.mark.scientific
    def test_v50_prediction_physical_bounds(self):
        """Test that V50 predictions are within physical bounds."""
        predictor = BallisticPredictor()

        # Test materials
        test_materials = [
            {
                "hardness": 28.0,
                "fracture_toughness": 4.5,
                "density": 3.21,
                "thermal_conductivity_1000C": 45.0,
            },
            {
                "hardness": 35.0,
                "fracture_toughness": 5.5,
                "density": 3.5,
                "thermal_conductivity_1000C": 50.0,
            },
        ]

        for material in test_materials:
            prediction = predictor.predict_v50(material)

            # V50 should be in reasonable range (m/s)
            # Typical ceramic armor: 500-1200 m/s
            assert 400 <= prediction["v50"] <= 1500

            # Should have confidence interval
            assert "confidence_interval" in prediction
            assert len(prediction["confidence_interval"]) == 2

            # Lower bound < prediction < upper bound
            assert prediction["confidence_interval"][0] < prediction["v50"]
            assert prediction["v50"] < prediction["confidence_interval"][1]

    @pytest.mark.scientific
    def test_property_correlation_with_performance(self):
        """Test that key properties correlate with ballistic performance."""
        predictor = BallisticPredictor()

        # Material with higher hardness should have higher V50
        material_low = {
            "hardness": 20.0,
            "fracture_toughness": 4.0,
            "density": 3.0,
            "thermal_conductivity_1000C": 40.0,
        }

        material_high = {
            "hardness": 35.0,
            "fracture_toughness": 4.0,
            "density": 3.0,
            "thermal_conductivity_1000C": 40.0,
        }

        v50_low = predictor.predict_v50(material_low)["v50"]
        v50_high = predictor.predict_v50(material_high)["v50"]

        # Higher hardness should generally lead to higher V50
        assert v50_high > v50_low


class TestPhysicalPlausibilityValidation:
    """Test physical plausibility validation."""

    @pytest.mark.scientific
    def test_elastic_moduli_relationships(self):
        """Test validation of elastic moduli relationships."""
        validator = PhysicalPlausibilityValidator()

        # Valid relationship: E ≈ 2G(1 + ν)
        properties_valid = {
            "youngs_modulus": 400.0,  # GPa
            "shear_modulus": 160.0,  # GPa
            "poisson_ratio": 0.25,  # E = 2*160*(1+0.25) = 400
        }

        report = validator.validate_material("test_valid", properties_valid)
        # Should not have critical violations
        assert not report.has_critical_violations()

        # Invalid relationship
        properties_invalid = {
            "youngs_modulus": 400.0,
            "shear_modulus": 100.0,  # Too low
            "poisson_ratio": 0.25,  # E = 2*100*(1+0.25) = 250, not 400
        }

        report = validator.validate_material("test_invalid", properties_invalid)
        # Should have warning or error
        assert len(report.violations) > 0

    @pytest.mark.scientific
    def test_thermal_conductivity_temperature_dependence(self):
        """Test validation of thermal conductivity temperature dependence."""
        validator = PhysicalPlausibilityValidator()

        # For ceramics, thermal conductivity typically decreases with temperature
        properties_correct = {
            "thermal_conductivity_25C": 120.0,
            "thermal_conductivity_1000C": 45.0,  # Lower at high temp
        }

        report = validator.validate_material("test_correct", properties_correct)
        # Should be valid
        assert not report.has_critical_violations()

        # Unusual: increasing with temperature
        properties_unusual = {
            "thermal_conductivity_25C": 45.0,
            "thermal_conductivity_1000C": 120.0,  # Higher at high temp
        }

        report = validator.validate_material("test_unusual", properties_unusual)
        # Should have warning
        assert len(report.violations) > 0

    @pytest.mark.scientific
    def test_density_composition_consistency(self):
        """Test density is consistent with composition."""
        validator = PhysicalPlausibilityValidator()

        # SiC should have density around 3.2 g/cm³
        properties_sic = {
            "composition": "SiC",
            "density": 3.21,
        }

        report = validator.validate_material("SiC", properties_sic)
        # Should be valid
        assert not report.has_critical_violations()

        # SiC with unrealistic density
        properties_wrong = {
            "composition": "SiC",
            "density": 10.0,  # Too high for SiC
        }

        report = validator.validate_material("SiC_wrong", properties_wrong)
        # Should have violation
        assert len(report.violations) > 0
