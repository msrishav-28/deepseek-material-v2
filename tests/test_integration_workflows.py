"""Integration tests for complete workflows."""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from ceramic_discovery.dft import (
    MaterialsProjectClient,
    StabilityAnalyzer,
    PropertyExtractor,
)
from ceramic_discovery.ceramics import CeramicSystemFactory
from ceramic_discovery.ml import (
    FeatureEngineeringPipeline,
    ModelTrainer,
    UncertaintyQuantifier,
)
from ceramic_discovery.ballistics import BallisticPredictor
from ceramic_discovery.screening import ScreeningEngine, WorkflowOrchestrator
from ceramic_discovery.validation import (
    PhysicalPlausibilityValidator,
    ReproducibilityFramework,
)


class TestEndToEndScreeningWorkflow:
    """Test complete screening workflow from DFT to predictions."""

    @pytest.mark.integration
    def test_complete_dopant_screening_workflow(self, temp_data_dir):
        """Test complete workflow: DFT → stability → ML → prediction."""
        # 1. Initialize components
        stability_analyzer = StabilityAnalyzer()
        feature_engineer = FeatureEngineeringPipeline()
        validator = PhysicalPlausibilityValidator()

        # 2. Create test materials (simulating DFT data)
        test_materials = [
            {
                "material_id": "test-sic-pure",
                "formula": "SiC",
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -0.65,
                "hardness": 28.0,
                "fracture_toughness": 4.5,
                "density": 3.21,
                "youngs_modulus": 410.0,
                "thermal_conductivity_1000C": 45.0,
            },
            {
                "material_id": "test-sic-b-doped",
                "formula": "Si0.98B0.02C",
                "energy_above_hull": 0.02,
                "formation_energy_per_atom": -0.63,
                "hardness": 29.5,
                "fracture_toughness": 4.8,
                "density": 3.19,
                "youngs_modulus": 415.0,
                "thermal_conductivity_1000C": 48.0,
            },
            {
                "material_id": "test-sic-unstable",
                "formula": "Si0.90Fe0.10C",
                "energy_above_hull": 0.25,
                "formation_energy_per_atom": -0.40,
                "hardness": 25.0,
                "fracture_toughness": 3.5,
                "density": 3.50,
                "youngs_modulus": 380.0,
                "thermal_conductivity_1000C": 35.0,
            },
        ]

        # 3. Filter by stability
        stability_results = stability_analyzer.batch_analyze(test_materials)
        viable_materials = [
            mat
            for mat, result in zip(test_materials, stability_results)
            if result.is_viable
        ]

        assert len(viable_materials) == 2  # Pure and B-doped should pass

        # 4. Validate physical plausibility
        validation_reports = []
        for material in viable_materials:
            properties = {
                k: v
                for k, v in material.items()
                if k not in ["material_id", "formula", "energy_above_hull", "formation_energy_per_atom"]
            }
            report = validator.validate_material(material["material_id"], properties)
            validation_reports.append(report)

        # All viable materials should pass validation
        assert all(not r.has_critical_violations() for r in validation_reports)

        # 5. Feature engineering
        features_list = []
        for material in viable_materials:
            features = feature_engineer.extract_features(material)
            features_list.append(features)

        assert len(features_list) == 2
        assert all(len(f) > 0 for f in features_list)

    @pytest.mark.integration
    def test_ml_training_and_prediction_workflow(self, temp_data_dir):
        """Test ML training and prediction workflow."""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 50

        X_train = {
            "hardness": np.random.uniform(20, 35, n_samples),
            "fracture_toughness": np.random.uniform(3, 6, n_samples),
            "density": np.random.uniform(2.5, 4.0, n_samples),
            "thermal_conductivity_1000C": np.random.uniform(30, 60, n_samples),
        }

        # Synthetic V50 values (simplified relationship)
        y_train = (
            X_train["hardness"] * 10
            + X_train["fracture_toughness"] * 20
            + X_train["density"] * 5
            + np.random.normal(0, 10, n_samples)
        )

        # 1. Feature engineering
        feature_engineer = FeatureEngineeringPipeline()
        X_features = feature_engineer.create_feature_matrix(X_train)

        # 2. Train model
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X_features, y_train)

        # 3. Make predictions with uncertainty
        X_test = X_features[:5]
        predictions = trainer.predict(X_test)
        
        assert len(predictions) == 5
        assert all(p > 0 for p in predictions)

        # 4. Uncertainty quantification
        uq = UncertaintyQuantifier()
        prediction_intervals = uq.bootstrap_prediction_intervals(
            trainer.model, X_features, y_train, X_test
        )

        assert prediction_intervals.shape == (5, 2)  # 5 samples, [lower, upper]
        assert all(prediction_intervals[:, 0] < prediction_intervals[:, 1])

    @pytest.mark.integration
    def test_reproducibility_workflow(self, temp_data_dir):
        """Test reproducibility framework in complete workflow."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        # Start experiment
        framework.start_experiment(
            experiment_id="integration_test_001",
            experiment_type="screening",
            parameters={"threshold": 0.1, "n_materials": 10},
        )

        # Simulate workflow steps
        framework.record_transformation(
            "Load DFT data", {"materials_loaded": 10}
        )

        framework.record_transformation(
            "Filter by stability", {"materials_viable": 7}
        )

        framework.record_transformation(
            "Train ML model", {"model_r2": 0.72}
        )

        # Finalize
        results = {"top_candidates": 3, "avg_v50": 850.0}
        snapshot = framework.finalize_experiment(results)

        assert snapshot.experiment_id == "integration_test_001"
        assert len(snapshot.provenance.transformations) == 3
        assert snapshot.results["top_candidates"] == 3


class TestScreeningEngineIntegration:
    """Test screening engine integration."""

    @pytest.mark.integration
    def test_screening_engine_with_real_ceramics(self):
        """Test screening engine with real ceramic systems."""
        engine = ScreeningEngine()

        # Get baseline ceramic systems
        sic = CeramicSystemFactory.create_sic()
        b4c = CeramicSystemFactory.create_b4c()

        # Create test materials
        materials = [
            {
                "id": "sic_baseline",
                "system": sic,
                "energy_above_hull": 0.0,
            },
            {
                "id": "b4c_baseline",
                "system": b4c,
                "energy_above_hull": 0.0,
            },
        ]

        # Run screening
        results = engine.screen_materials(
            materials, stability_threshold=0.1, min_hardness=20.0
        )

        assert len(results) > 0
        assert all(r["energy_above_hull"] <= 0.1 for r in results)

    @pytest.mark.integration
    def test_workflow_orchestrator(self, temp_data_dir):
        """Test workflow orchestrator."""
        orchestrator = WorkflowOrchestrator(output_dir=temp_data_dir)

        # Define simple workflow
        def simple_workflow():
            return {"status": "complete", "materials_processed": 5}

        # Execute workflow
        result = orchestrator.execute_workflow(
            workflow_id="test_workflow_001",
            workflow_function=simple_workflow,
            parameters={},
        )

        assert result["status"] == "complete"
        assert result["materials_processed"] == 5


class TestDataPipelineIntegration:
    """Test data pipeline integration."""

    @pytest.mark.integration
    def test_dft_to_database_pipeline(self, temp_data_dir):
        """Test DFT data collection to database storage."""
        # This would test the full pipeline from Materials Project to database
        # Skipping actual API calls in tests
        pass

    @pytest.mark.integration
    def test_property_extraction_pipeline(self):
        """Test property extraction pipeline."""
        extractor = PropertyExtractor()

        # Mock DFT calculation result
        dft_result = {
            "material_id": "mp-149",
            "formula": "SiC",
            "structure": {"lattice": {"a": 4.36}},
            "energy_per_atom": -9.5,
            "band_gap": 2.3,
        }

        # Extract properties
        properties = extractor.extract_all_properties(dft_result)

        assert "material_id" in properties
        assert "formula" in properties
        # Should have extracted available properties


class TestValidationIntegration:
    """Test validation integration."""

    @pytest.mark.integration
    def test_validation_in_screening_workflow(self):
        """Test validation integrated into screening workflow."""
        validator = PhysicalPlausibilityValidator()
        stability_analyzer = StabilityAnalyzer()

        # Create test material
        material = {
            "material_id": "test-001",
            "formula": "SiC",
            "energy_above_hull": 0.05,
            "formation_energy_per_atom": -0.65,
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
        }

        # 1. Check stability
        stability_result = stability_analyzer.analyze_stability(
            material["material_id"],
            material["formula"],
            material["energy_above_hull"],
            material["formation_energy_per_atom"],
        )

        assert stability_result.is_viable

        # 2. Validate physical plausibility
        properties = {
            "hardness": material["hardness"],
            "fracture_toughness": material["fracture_toughness"],
            "density": material["density"],
        }

        validation_report = validator.validate_material(
            material["material_id"], properties
        )

        assert not validation_report.has_critical_violations()

    @pytest.mark.integration
    def test_cross_validation_with_literature(self):
        """Test cross-validation with literature data."""
        validator = PhysicalPlausibilityValidator()

        # Add literature references
        validator.add_literature_reference(
            material_id="SiC",
            property_name="hardness",
            value=28.0,
            uncertainty=2.0,
            source="Munro, 1997",
        )

        # Test prediction
        violation = validator.cross_reference_literature(
            material_id="SiC",
            property_name="hardness",
            predicted_value=27.5,
            predicted_uncertainty=1.5,
        )

        # Should be within literature range
        assert violation is None or violation.severity != "CRITICAL"
