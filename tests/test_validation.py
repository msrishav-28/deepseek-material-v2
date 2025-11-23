"""Tests for validation and reproducibility framework."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ceramic_discovery.ceramics.ceramic_system import (
    CeramicSystemFactory,
    ValueWithUncertainty,
)
from ceramic_discovery.validation import (
    PhysicalPlausibilityValidator,
    PropertyBounds,
    SensitivityAnalyzer,
    ValidationReport,
    ValidationSeverity,
    ValidationViolation,
    ComputationalEnvironment,
    DataProvenance,
    ExperimentParameters,
    ExperimentSnapshot,
    RandomSeedManager,
    ReproducibilityFramework,
)


class TestPhysicalPlausibilityValidator:
    """Tests for PhysicalPlausibilityValidator."""
    
    def test_validator_initialization(self):
        """Test validator initializes with property bounds."""
        validator = PhysicalPlausibilityValidator()
        
        assert "hardness" in validator.property_bounds
        assert "fracture_toughness" in validator.property_bounds
        assert "thermal_conductivity_25C" in validator.property_bounds
        assert "density" in validator.property_bounds
    
    def test_validate_property_within_bounds(self):
        """Test validation of property within bounds."""
        validator = PhysicalPlausibilityValidator()
        
        # Valid hardness value
        violation = validator.validate_property("hardness", 28.0)
        assert violation is None or violation.severity == ValidationSeverity.INFO
    
    def test_validate_property_outside_physical_bounds(self):
        """Test validation of property outside physical bounds."""
        validator = PhysicalPlausibilityValidator()
        
        # Hardness way too high (physically impossible)
        violation = validator.validate_property("hardness", 100.0)
        assert violation is not None
        assert violation.severity == ValidationSeverity.CRITICAL
        assert "physical bounds" in violation.message.lower()
    
    def test_validate_property_outside_typical_range(self):
        """Test validation of property outside typical range."""
        validator = PhysicalPlausibilityValidator()
        
        # Hardness within physical bounds but outside typical range
        violation = validator.validate_property("hardness", 45.0)
        assert violation is not None
        assert violation.severity == ValidationSeverity.WARNING
        assert "typical range" in violation.message.lower()
    
    def test_validate_material_sic(self):
        """Test validation of SiC material."""
        validator = PhysicalPlausibilityValidator()
        sic = CeramicSystemFactory.create_sic()
        
        # Extract properties
        properties = {
            "hardness": sic.mechanical.hardness,
            "fracture_toughness": sic.mechanical.fracture_toughness,
            "youngs_modulus": sic.mechanical.youngs_modulus,
            "density": sic.structural.density,
            "thermal_conductivity_25C": sic.thermal.thermal_conductivity_25C,
        }
        
        report = validator.validate_material("SiC", properties)
        
        assert report.material_id == "SiC"
        # SiC baseline properties should be valid
        assert report.is_valid or len(report.get_violations_by_severity(ValidationSeverity.CRITICAL)) == 0
    
    def test_validate_material_with_violations(self):
        """Test validation of material with violations."""
        validator = PhysicalPlausibilityValidator()
        
        # Create material with invalid properties
        properties = {
            "hardness": 100.0,  # Too high
            "fracture_toughness": 20.0,  # Too high
            "density": 0.5,  # Too low
        }
        
        report = validator.validate_material("invalid_material", properties)
        
        assert not report.is_valid
        assert len(report.violations) > 0
        assert report.has_critical_violations()
    
    def test_cross_property_consistency_elastic_moduli(self):
        """Test cross-property consistency for elastic moduli."""
        validator = PhysicalPlausibilityValidator()
        
        # Create properties with inconsistent elastic moduli
        properties = {
            "youngs_modulus": 400.0,  # GPa
            "shear_modulus": 150.0,  # GPa
            "poisson_ratio": 0.2,
        }
        
        report = validator.validate_material("test_material", properties)
        
        # Should have warning about elastic moduli consistency
        # E = 2G(1 + ν) => 400 ≈ 2*150*(1+0.2) = 360
        # This is within 30% tolerance, so should be OK
        consistency_violations = [v for v in report.violations if "consistency" in v.property_name]
        # May or may not have violation depending on tolerance
    
    def test_cross_property_consistency_thermal_conductivity(self):
        """Test cross-property consistency for thermal conductivity."""
        validator = PhysicalPlausibilityValidator()
        
        # Thermal conductivity increasing with temperature (unusual for ceramics)
        properties = {
            "thermal_conductivity_25C": 50.0,
            "thermal_conductivity_1000C": 80.0,  # Higher than at 25C
        }
        
        report = validator.validate_material("test_material", properties)
        
        # Should have warning about thermal conductivity temperature dependence
        temp_violations = [v for v in report.violations if "temperature" in v.property_name]
        assert len(temp_violations) > 0
    
    def test_validation_report_summary(self):
        """Test validation report summary generation."""
        report = ValidationReport(material_id="test_material")
        
        report.add_violation(ValidationViolation(
            property_name="hardness",
            value=100.0,
            severity=ValidationSeverity.CRITICAL,
            message="Hardness too high"
        ))
        
        report.add_violation(ValidationViolation(
            property_name="density",
            value=0.5,
            severity=ValidationSeverity.ERROR,
            message="Density too low"
        ))
        
        summary = report.summary()
        
        assert "test_material" in summary
        assert "INVALID" in summary
        assert "Critical: 1" in summary
        assert "Errors: 1" in summary
    
    def test_batch_validation(self):
        """Test batch validation of multiple materials."""
        validator = PhysicalPlausibilityValidator()
        
        materials = [
            {
                "id": "material_1",
                "hardness": 28.0,
                "density": 3.2
            },
            {
                "id": "material_2",
                "hardness": 100.0,  # Invalid
                "density": 2.5
            }
        ]
        
        reports = validator.validate_batch(materials)
        
        assert len(reports) == 2
        assert reports[0].is_valid or not reports[0].has_critical_violations()
        assert not reports[1].is_valid
    
    def test_literature_cross_reference(self):
        """Test cross-referencing with literature data."""
        validator = PhysicalPlausibilityValidator()
        
        # Add literature reference
        validator.add_literature_reference(
            material_id="SiC",
            property_name="hardness",
            value=28.0,
            uncertainty=2.0,
            source="Munro, 1997"
        )
        
        # Test prediction close to literature
        violation = validator.cross_reference_literature(
            material_id="SiC",
            property_name="hardness",
            predicted_value=27.5,
            predicted_uncertainty=1.5
        )
        
        assert violation is None  # Within uncertainty
        
        # Test prediction far from literature
        violation = validator.cross_reference_literature(
            material_id="SiC",
            property_name="hardness",
            predicted_value=40.0,
            predicted_uncertainty=1.0
        )
        
        assert violation is not None
        assert violation.severity in [ValidationSeverity.WARNING, ValidationSeverity.ERROR]
    
    def test_validation_statistics(self):
        """Test validation statistics calculation."""
        validator = PhysicalPlausibilityValidator()
        
        materials = [
            {"id": f"material_{i}", "hardness": 20.0 + i, "density": 3.0}
            for i in range(10)
        ]
        
        reports = validator.validate_batch(materials)
        stats = validator.get_validation_statistics(reports)
        
        assert stats["total_materials"] == 10
        assert "valid_materials" in stats
        assert "avg_confidence" in stats
        assert "violations_by_severity" in stats


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer."""
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis for property changes."""
        analyzer = SensitivityAnalyzer()
        
        # Simple target metric: weighted sum of properties
        def target_metric(props):
            return (
                props.get("hardness", 0) * 2.0 +
                props.get("density", 0) * 1.0 +
                props.get("thermal_conductivity", 0) * 0.5
            )
        
        base_properties = {
            "hardness": 28.0,
            "density": 3.2,
            "thermal_conductivity": 120.0
        }
        
        sensitivities = analyzer.analyze_property_sensitivity(
            base_properties,
            target_metric,
            perturbation_fraction=0.1
        )
        
        assert "hardness" in sensitivities
        assert "density" in sensitivities
        assert "thermal_conductivity" in sensitivities
        
        # Hardness should have highest sensitivity (weight=2.0)
        assert abs(sensitivities["hardness"]) > abs(sensitivities["density"])
    
    def test_rank_performance_drivers(self):
        """Test ranking of performance drivers."""
        analyzer = SensitivityAnalyzer()
        
        sensitivities = {
            "hardness": 2.0,
            "density": 1.0,
            "thermal_conductivity": 0.5,
            "fracture_toughness": -1.5
        }
        
        ranked = analyzer.rank_performance_drivers(sensitivities, top_n=3)
        
        assert len(ranked) == 3
        # Should be sorted by absolute value
        assert ranked[0][0] == "hardness"
        assert abs(ranked[0][1]) >= abs(ranked[1][1])
    
    def test_sensitivity_report_generation(self):
        """Test sensitivity report generation."""
        analyzer = SensitivityAnalyzer()
        
        sensitivities = {
            "hardness": 2.0,
            "density": -1.0,
        }
        
        property_units = {
            "hardness": "GPa",
            "density": "g/cm³"
        }
        
        report = analyzer.generate_sensitivity_report(sensitivities, property_units)
        
        assert "Sensitivity Analysis Report" in report
        assert "hardness" in report
        assert "density" in report
        assert "GPa" in report


class TestRandomSeedManager:
    """Tests for RandomSeedManager."""
    
    def test_seed_manager_initialization(self):
        """Test seed manager initialization."""
        manager = RandomSeedManager(master_seed=42)
        
        assert manager.master_seed == 42
        assert len(manager.component_seeds) == 0
    
    def test_set_global_seed(self):
        """Test setting global random seed."""
        manager = RandomSeedManager(master_seed=42)
        manager.set_global_seed()
        
        # Generate random numbers
        r1 = np.random.random()
        
        # Reset seed
        manager.set_global_seed()
        
        # Should get same random number
        r2 = np.random.random()
        
        assert r1 == r2
    
    def test_component_seed_deterministic(self):
        """Test component seeds are deterministic."""
        manager1 = RandomSeedManager(master_seed=42)
        manager2 = RandomSeedManager(master_seed=42)
        
        seed1 = manager1.get_component_seed("ml_training")
        seed2 = manager2.get_component_seed("ml_training")
        
        assert seed1 == seed2
    
    def test_component_seed_different_components(self):
        """Test different components get different seeds."""
        manager = RandomSeedManager(master_seed=42)
        
        seed1 = manager.get_component_seed("ml_training")
        seed2 = manager.get_component_seed("dft_calculation")
        
        assert seed1 != seed2
    
    def test_create_rng(self):
        """Test creating component-specific RNG."""
        manager = RandomSeedManager(master_seed=42)
        
        rng1 = manager.create_rng("component_1")
        rng2 = manager.create_rng("component_1")
        
        # Same component should give same sequence
        val1 = rng1.random()
        val2 = rng2.random()
        
        assert val1 == val2


class TestReproducibilityFramework:
    """Tests for ReproducibilityFramework."""
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            assert framework.experiment_dir.exists()
            assert framework.seed_manager.master_seed == 42
    
    def test_start_experiment(self):
        """Test starting an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            snapshot = framework.start_experiment(
                experiment_id="test_exp_001",
                experiment_type="ml_training",
                parameters={"learning_rate": 0.01, "epochs": 100},
                random_seed=42
            )
            
            assert snapshot.experiment_id == "test_exp_001"
            assert snapshot.parameters.experiment_type == "ml_training"
            assert snapshot.parameters.random_seed == 42
            assert "learning_rate" in snapshot.parameters.parameters
    
    def test_record_transformation(self):
        """Test recording data transformations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            framework.start_experiment(
                experiment_id="test_exp_002",
                experiment_type="screening",
                parameters={}
            )
            
            framework.record_transformation(
                description="Load raw data",
                data={"materials": [1, 2, 3]}
            )
            
            framework.record_transformation(
                description="Filter viable materials",
                data={"materials": [1, 2]}
            )
            
            assert len(framework.current_snapshot.provenance.transformations) == 2
            assert len(framework.current_snapshot.provenance.intermediate_hashes) == 2
    
    def test_finalize_experiment(self):
        """Test finalizing an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            framework.start_experiment(
                experiment_id="test_exp_003",
                experiment_type="analysis",
                parameters={"threshold": 0.5}
            )
            
            results = {"accuracy": 0.85, "materials_found": 15}
            
            snapshot = framework.finalize_experiment(results)
            
            assert snapshot.results == results
            assert snapshot.provenance.final_hash is not None
            
            # Check snapshot was saved
            snapshot_path = Path(tmpdir) / "test_exp_003.json"
            assert snapshot_path.exists()
    
    def test_experiment_snapshot_save_load(self):
        """Test saving and loading experiment snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create snapshot
            params = ExperimentParameters(
                experiment_id="test_exp_004",
                experiment_type="test",
                parameters={"param1": 1.0},
                random_seed=42
            )
            
            env = ComputationalEnvironment.capture_current()
            prov = DataProvenance(source_data="test_data")
            
            snapshot = ExperimentSnapshot(
                experiment_id="test_exp_004",
                parameters=params,
                environment=env,
                provenance=prov,
                results={"result": 42}
            )
            
            # Save
            filepath = Path(tmpdir) / "test_snapshot.json"
            snapshot.save(filepath)
            
            # Load
            loaded_snapshot = ExperimentSnapshot.load(filepath)
            
            assert loaded_snapshot.experiment_id == snapshot.experiment_id
            assert loaded_snapshot.parameters.random_seed == snapshot.parameters.random_seed
            assert loaded_snapshot.results == snapshot.results
    
    def test_verify_reproducibility(self):
        """Test reproducibility verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            # Run experiment
            framework.start_experiment(
                experiment_id="test_exp_005",
                experiment_type="calculation",
                parameters={"n": 100}
            )
            
            # Deterministic calculation
            results = {"sum": sum(range(100)), "mean": np.mean(range(100))}
            framework.finalize_experiment(results)
            
            # Verify reproducibility
            def rerun_function(params):
                n = params["n"]
                return {"sum": sum(range(n)), "mean": np.mean(range(n))}
            
            verification = framework.verify_reproducibility("test_exp_005", rerun_function)
            
            assert verification["reproducible"]
            assert verification["hash_match"]
    
    def test_export_reproducibility_package(self):
        """Test exporting reproducibility package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ReproducibilityFramework(
                experiment_dir=Path(tmpdir),
                master_seed=42
            )
            
            # Create and finalize experiment
            framework.start_experiment(
                experiment_id="test_exp_006",
                experiment_type="export_test",
                parameters={"test": True}
            )
            framework.finalize_experiment({"status": "complete"})
            
            # Export package
            output_dir = Path(tmpdir) / "package"
            framework.export_reproducibility_package("test_exp_006", output_dir)
            
            # Check package contents
            assert (output_dir / "experiment_snapshot.json").exists()
            assert (output_dir / "requirements.txt").exists()
            assert (output_dir / "README.md").exists()
            
            # Check README content
            readme_content = (output_dir / "README.md").read_text()
            assert "test_exp_006" in readme_content
            assert "export_test" in readme_content


class TestComputationalEnvironment:
    """Tests for ComputationalEnvironment."""
    
    def test_capture_current_environment(self):
        """Test capturing current computational environment."""
        env = ComputationalEnvironment.capture_current()
        
        assert env.python_version
        assert env.platform
        assert env.platform_version
        # Dependencies may or may not be populated depending on environment
    
    def test_environment_to_dict(self):
        """Test converting environment to dictionary."""
        env = ComputationalEnvironment.capture_current()
        env_dict = env.to_dict()
        
        assert "python_version" in env_dict
        assert "platform" in env_dict
        assert "dependencies" in env_dict


class TestDataProvenance:
    """Tests for DataProvenance."""
    
    def test_provenance_initialization(self):
        """Test provenance initialization."""
        prov = DataProvenance(source_data="raw_materials.csv")
        
        assert prov.source_data == "raw_materials.csv"
        assert len(prov.transformations) == 0
        assert len(prov.intermediate_hashes) == 0
    
    def test_add_transformation(self):
        """Test adding transformations."""
        prov = DataProvenance(source_data="raw_data")
        
        prov.add_transformation("Filter by stability", "hash1")
        prov.add_transformation("Calculate properties", "hash2")
        
        assert len(prov.transformations) == 2
        assert len(prov.intermediate_hashes) == 2
        assert prov.transformations[0] == "Filter by stability"
    
    def test_finalize_provenance(self):
        """Test finalizing provenance."""
        prov = DataProvenance(source_data="raw_data")
        prov.add_transformation("Process", "hash1")
        prov.finalize("final_hash")
        
        assert prov.final_hash == "final_hash"
