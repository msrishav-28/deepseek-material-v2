"""Reproducibility tests to ensure deterministic results."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from ceramic_discovery.validation import (
    RandomSeedManager,
    ReproducibilityFramework,
)
from ceramic_discovery.ml import ModelTrainer, FeatureEngineeringPipeline
from ceramic_discovery.dft import StabilityAnalyzer
from ceramic_discovery.screening import ScreeningEngine


class TestDeterministicBehavior:
    """Test that operations are deterministic with fixed seeds."""

    @pytest.mark.reproducibility
    def test_random_seed_manager_determinism(self):
        """Test that random seed manager produces deterministic results."""
        # Create two managers with same seed
        manager1 = RandomSeedManager(master_seed=42)
        manager2 = RandomSeedManager(master_seed=42)

        # Set global seeds
        manager1.set_global_seed()
        values1 = [np.random.random() for _ in range(10)]

        manager2.set_global_seed()
        values2 = [np.random.random() for _ in range(10)]

        # Should produce identical sequences
        assert np.allclose(values1, values2)

    @pytest.mark.reproducibility
    def test_component_seed_determinism(self):
        """Test that component seeds are deterministic."""
        manager1 = RandomSeedManager(master_seed=42)
        manager2 = RandomSeedManager(master_seed=42)

        # Get seeds for same components
        seed1_ml = manager1.get_component_seed("ml_training")
        seed1_dft = manager1.get_component_seed("dft_calculation")

        seed2_ml = manager2.get_component_seed("ml_training")
        seed2_dft = manager2.get_component_seed("dft_calculation")

        # Should be identical
        assert seed1_ml == seed2_ml
        assert seed1_dft == seed2_dft

        # Different components should have different seeds
        assert seed1_ml != seed1_dft

    @pytest.mark.reproducibility
    def test_ml_training_reproducibility(self):
        """Test that ML training is reproducible with fixed seed."""
        np.random.seed(42)

        # Create training data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Train first model
        trainer1 = ModelTrainer(model_type="random_forest", random_state=42)
        trainer1.train(X, y)
        predictions1 = trainer1.predict(X[:10])

        # Train second model with same seed
        trainer2 = ModelTrainer(model_type="random_forest", random_state=42)
        trainer2.train(X, y)
        predictions2 = trainer2.predict(X[:10])

        # Predictions should be identical
        assert np.allclose(predictions1, predictions2)

    @pytest.mark.reproducibility
    def test_feature_engineering_reproducibility(self):
        """Test that feature engineering is reproducible."""
        engineer = FeatureEngineeringPipeline()

        material = {
            "hardness": 28.0,
            "fracture_toughness": 4.5,
            "density": 3.21,
            "youngs_modulus": 410.0,
        }

        # Extract features multiple times
        features1 = engineer.extract_features(material)
        features2 = engineer.extract_features(material)

        # Should be identical
        assert features1 == features2

    @pytest.mark.reproducibility
    def test_stability_analysis_reproducibility(self):
        """Test that stability analysis is reproducible."""
        analyzer = StabilityAnalyzer()

        material = {
            "material_id": "test-001",
            "formula": "SiC",
            "energy_above_hull": 0.05,
            "formation_energy_per_atom": -0.65,
        }

        # Analyze multiple times
        result1 = analyzer.analyze_stability(
            material["material_id"],
            material["formula"],
            material["energy_above_hull"],
            material["formation_energy_per_atom"],
        )

        result2 = analyzer.analyze_stability(
            material["material_id"],
            material["formula"],
            material["energy_above_hull"],
            material["formation_energy_per_atom"],
        )

        # Results should be identical
        assert result1.classification == result2.classification
        assert result1.is_viable == result2.is_viable
        assert result1.confidence_score == result2.confidence_score


class TestExperimentReproducibility:
    """Test experiment-level reproducibility."""

    @pytest.mark.reproducibility
    def test_complete_workflow_reproducibility(self, temp_data_dir):
        """Test that complete workflow is reproducible."""
        # Run workflow twice with same seed
        results1 = self._run_workflow(temp_data_dir, seed=42)
        results2 = self._run_workflow(temp_data_dir, seed=42)

        # Results should be identical
        assert results1["n_viable"] == results2["n_viable"]
        assert np.allclose(results1["avg_confidence"], results2["avg_confidence"])

    def _run_workflow(self, output_dir, seed):
        """Helper to run a simple workflow."""
        np.random.seed(seed)

        # Create test materials
        materials = [
            {
                "material_id": f"test-{i}",
                "formula": f"Mat{i}",
                "energy_above_hull": np.random.uniform(0, 0.3),
                "formation_energy_per_atom": np.random.uniform(-1.0, -0.3),
            }
            for i in range(20)
        ]

        # Analyze stability
        analyzer = StabilityAnalyzer()
        results = analyzer.batch_analyze(materials)

        # Calculate statistics
        viable_results = [r for r in results if r.is_viable]
        avg_confidence = np.mean([r.confidence_score for r in results])

        return {
            "n_viable": len(viable_results),
            "avg_confidence": avg_confidence,
        }

    @pytest.mark.reproducibility
    def test_experiment_snapshot_reproducibility(self, temp_data_dir):
        """Test that experiment snapshots enable reproducibility."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        # Run experiment
        framework.start_experiment(
            experiment_id="repro_test_001",
            experiment_type="test",
            parameters={"n_samples": 100, "threshold": 0.1},
        )

        # Simulate workflow
        np.random.seed(42)
        data = np.random.randn(100)
        result = {"mean": float(np.mean(data)), "std": float(np.std(data))}

        framework.finalize_experiment(result)

        # Verify reproducibility
        def rerun_function(params):
            np.random.seed(42)
            data = np.random.randn(params["n_samples"])
            return {"mean": float(np.mean(data)), "std": float(np.std(data))}

        verification = framework.verify_reproducibility(
            "repro_test_001", rerun_function
        )

        assert verification["reproducible"]
        assert verification["hash_match"]

    @pytest.mark.reproducibility
    def test_cross_platform_reproducibility(self, temp_data_dir):
        """Test reproducibility across different runs."""
        # This tests that results are consistent within same environment
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        # Run 1
        framework.start_experiment(
            experiment_id="cross_platform_001",
            experiment_type="test",
            parameters={"seed": 42},
        )

        np.random.seed(42)
        result1 = {"value": float(np.random.random())}
        snapshot1 = framework.finalize_experiment(result1)

        # Run 2 (new framework instance)
        framework2 = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        framework2.start_experiment(
            experiment_id="cross_platform_002",
            experiment_type="test",
            parameters={"seed": 42},
        )

        np.random.seed(42)
        result2 = {"value": float(np.random.random())}
        snapshot2 = framework2.finalize_experiment(result2)

        # Results should be identical
        assert result1["value"] == result2["value"]


class TestProvenanceTracking:
    """Test provenance tracking for reproducibility."""

    @pytest.mark.reproducibility
    def test_provenance_captures_transformations(self, temp_data_dir):
        """Test that provenance captures all data transformations."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        framework.start_experiment(
            experiment_id="provenance_test_001",
            experiment_type="analysis",
            parameters={},
        )

        # Record transformations
        framework.record_transformation("Load raw data", {"n_materials": 100})
        framework.record_transformation("Filter by stability", {"n_viable": 75})
        framework.record_transformation("Train ML model", {"r2": 0.72})

        snapshot = framework.finalize_experiment({"final_count": 75})

        # Check provenance
        assert len(snapshot.provenance.transformations) == 3
        assert "Load raw data" in snapshot.provenance.transformations
        assert "Filter by stability" in snapshot.provenance.transformations
        assert "Train ML model" in snapshot.provenance.transformations

    @pytest.mark.reproducibility
    def test_provenance_hash_consistency(self, temp_data_dir):
        """Test that provenance hashes are consistent."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        # Run same workflow twice
        for i in range(2):
            framework.start_experiment(
                experiment_id=f"hash_test_{i:03d}",
                experiment_type="test",
                parameters={"data": [1, 2, 3]},
            )

            framework.record_transformation("Process", {"result": [2, 4, 6]})
            snapshot = framework.finalize_experiment({"sum": 12})

            if i == 0:
                hash1 = snapshot.provenance.final_hash
            else:
                hash2 = snapshot.provenance.final_hash

        # Hashes should be identical for identical workflows
        assert hash1 == hash2

    @pytest.mark.reproducibility
    def test_environment_capture(self, temp_data_dir):
        """Test that computational environment is captured."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        framework.start_experiment(
            experiment_id="env_test_001",
            experiment_type="test",
            parameters={},
        )

        snapshot = framework.finalize_experiment({})

        # Check environment was captured
        assert snapshot.environment.python_version
        assert snapshot.environment.platform
        assert snapshot.environment.platform_version


class TestReproducibilityPackageExport:
    """Test reproducibility package export."""

    @pytest.mark.reproducibility
    def test_export_package_completeness(self, temp_data_dir):
        """Test that exported package contains all necessary files."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        # Create and finalize experiment
        framework.start_experiment(
            experiment_id="export_test_001",
            experiment_type="test",
            parameters={"test": True},
        )

        framework.finalize_experiment({"status": "complete"})

        # Export package
        output_dir = temp_data_dir / "export_package"
        framework.export_reproducibility_package("export_test_001", output_dir)

        # Check required files exist
        assert (output_dir / "experiment_snapshot.json").exists()
        assert (output_dir / "requirements.txt").exists()
        assert (output_dir / "README.md").exists()

    @pytest.mark.reproducibility
    def test_export_package_readme_content(self, temp_data_dir):
        """Test that README contains necessary information."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        framework.start_experiment(
            experiment_id="readme_test_001",
            experiment_type="screening",
            parameters={"materials": 100},
        )

        framework.finalize_experiment({"viable": 75})

        # Export
        output_dir = temp_data_dir / "readme_package"
        framework.export_reproducibility_package("readme_test_001", output_dir)

        # Check README content
        readme_content = (output_dir / "README.md").read_text()

        assert "readme_test_001" in readme_content
        assert "screening" in readme_content
        assert "Reproducibility Package" in readme_content


class TestDataVersioning:
    """Test data versioning for reproducibility."""

    @pytest.mark.reproducibility
    def test_parameter_versioning(self, temp_data_dir):
        """Test that parameters are versioned."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        parameters = {
            "learning_rate": 0.01,
            "n_estimators": 100,
            "max_depth": 10,
        }

        framework.start_experiment(
            experiment_id="param_version_001",
            experiment_type="ml_training",
            parameters=parameters,
        )

        snapshot = framework.finalize_experiment({})

        # Check parameters are stored
        assert snapshot.parameters.parameters == parameters

    @pytest.mark.reproducibility
    def test_software_version_tracking(self, temp_data_dir):
        """Test that software versions are tracked."""
        framework = ReproducibilityFramework(
            experiment_dir=temp_data_dir, master_seed=42
        )

        framework.start_experiment(
            experiment_id="version_test_001",
            experiment_type="test",
            parameters={},
        )

        snapshot = framework.finalize_experiment({})

        # Check that Python version is captured
        assert snapshot.environment.python_version
        assert len(snapshot.environment.python_version) > 0


class TestRandomnessControl:
    """Test control of randomness for reproducibility."""

    @pytest.mark.reproducibility
    def test_numpy_random_state_control(self):
        """Test control of numpy random state."""
        manager = RandomSeedManager(master_seed=42)

        # Create RNG for component
        rng1 = manager.create_rng("component_1")
        values1 = [rng1.random() for _ in range(5)]

        # Create another RNG for same component
        rng2 = manager.create_rng("component_1")
        values2 = [rng2.random() for _ in range(5)]

        # Should produce same sequence
        assert np.allclose(values1, values2)

    @pytest.mark.reproducibility
    def test_different_components_different_sequences(self):
        """Test that different components get different random sequences."""
        manager = RandomSeedManager(master_seed=42)

        rng1 = manager.create_rng("component_1")
        rng2 = manager.create_rng("component_2")

        values1 = [rng1.random() for _ in range(5)]
        values2 = [rng2.random() for _ in range(5)]

        # Should produce different sequences
        assert not np.allclose(values1, values2)

    @pytest.mark.reproducibility
    def test_global_seed_reset(self):
        """Test that global seed can be reset."""
        manager = RandomSeedManager(master_seed=42)

        # Generate values
        manager.set_global_seed()
        values1 = [np.random.random() for _ in range(5)]

        # Generate more values (different)
        values_between = [np.random.random() for _ in range(5)]

        # Reset and generate again
        manager.set_global_seed()
        values2 = [np.random.random() for _ in range(5)]

        # First and third should be identical
        assert np.allclose(values1, values2)
        # But different from middle values
        assert not np.allclose(values1, values_between)
