"""Reproducibility framework for computational experiments."""

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SoftwareVersion:
    """Software version information."""
    
    name: str
    version: str
    
    def __str__(self) -> str:
        return f"{self.name}=={self.version}"


@dataclass
class ComputationalEnvironment:
    """Complete computational environment specification."""
    
    python_version: str
    platform: str
    platform_version: str
    processor: str
    dependencies: List[SoftwareVersion] = field(default_factory=list)
    
    @classmethod
    def capture_current(cls) -> "ComputationalEnvironment":
        """Capture current computational environment."""
        return cls(
            python_version=sys.version,
            platform=platform.system(),
            platform_version=platform.version(),
            processor=platform.processor(),
            dependencies=cls._get_installed_packages()
        )
    
    @staticmethod
    def _get_installed_packages() -> List[SoftwareVersion]:
        """Get list of installed Python packages."""
        try:
            import pkg_resources
            packages = []
            for dist in pkg_resources.working_set:
                packages.append(SoftwareVersion(
                    name=dist.project_name,
                    version=dist.version
                ))
            return sorted(packages, key=lambda x: x.name)
        except Exception:
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'python_version': self.python_version,
            'platform': self.platform,
            'platform_version': self.platform_version,
            'processor': self.processor,
            'dependencies': [{'name': d.name, 'version': d.version} for d in self.dependencies]
        }


@dataclass
class ExperimentParameters:
    """Parameters for a computational experiment."""
    
    experiment_id: str
    experiment_type: str  # 'screening', 'ml_training', 'property_calculation', etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of parameters."""
        # Sort keys for deterministic hashing
        param_str = json.dumps(self.parameters, sort_keys=True)
        hash_input = f"{self.experiment_type}:{param_str}:{self.random_seed}"
        return hashlib.sha256(hash_input.encode()).hexdigest()


@dataclass
class DataProvenance:
    """Provenance tracking for data transformations."""
    
    source_data: str  # Description or hash of source data
    transformations: List[str] = field(default_factory=list)
    intermediate_hashes: List[str] = field(default_factory=list)
    final_hash: Optional[str] = None
    
    def add_transformation(self, description: str, data_hash: str) -> None:
        """Add a transformation step."""
        self.transformations.append(description)
        self.intermediate_hashes.append(data_hash)
    
    def finalize(self, final_hash: str) -> None:
        """Finalize provenance with final data hash."""
        self.final_hash = final_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentSnapshot:
    """Complete snapshot of a computational experiment."""
    
    experiment_id: str
    parameters: ExperimentParameters
    environment: ComputationalEnvironment
    provenance: DataProvenance
    results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'parameters': self.parameters.to_dict(),
            'environment': self.environment.to_dict(),
            'provenance': self.provenance.to_dict(),
            'results': self.results,
            'metadata': self.metadata
        }
    
    def save(self, filepath: Path) -> None:
        """Save snapshot to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "ExperimentSnapshot":
        """Load snapshot from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        params = ExperimentParameters(**data['parameters'])
        
        env_data = data['environment']
        env = ComputationalEnvironment(
            python_version=env_data['python_version'],
            platform=env_data['platform'],
            platform_version=env_data['platform_version'],
            processor=env_data['processor'],
            dependencies=[SoftwareVersion(**d) for d in env_data['dependencies']]
        )
        
        prov = DataProvenance(**data['provenance'])
        
        return cls(
            experiment_id=data['experiment_id'],
            parameters=params,
            environment=env,
            provenance=prov,
            results=data.get('results'),
            metadata=data.get('metadata', {})
        )


class RandomSeedManager:
    """
    Manager for deterministic random number generation.
    
    Ensures reproducibility by managing random seeds for:
    - Python's random module
    - NumPy
    - Other libraries that use randomness
    """
    
    def __init__(self, master_seed: int = 42):
        """
        Initialize random seed manager.
        
        Args:
            master_seed: Master seed for all random number generators
        """
        self.master_seed = master_seed
        self.component_seeds = {}
    
    def set_global_seed(self, seed: Optional[int] = None) -> None:
        """
        Set global random seed for all libraries.
        
        Args:
            seed: Random seed (uses master_seed if None)
        """
        seed = seed if seed is not None else self.master_seed
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy seed
        np.random.seed(seed)
        
        # Set seeds for other libraries if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def get_component_seed(self, component_name: str) -> int:
        """
        Get deterministic seed for a specific component.
        
        Args:
            component_name: Name of the component
        
        Returns:
            Deterministic seed derived from master seed and component name
        """
        if component_name not in self.component_seeds:
            # Generate deterministic seed from master seed and component name
            hash_input = f"{self.master_seed}:{component_name}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            # Convert first 8 hex digits to integer
            seed = int(hash_value[:8], 16)
            self.component_seeds[component_name] = seed
        
        return self.component_seeds[component_name]
    
    def create_rng(self, component_name: str) -> np.random.Generator:
        """
        Create a dedicated random number generator for a component.
        
        Args:
            component_name: Name of the component
        
        Returns:
            NumPy random number generator with component-specific seed
        """
        seed = self.get_component_seed(component_name)
        return np.random.default_rng(seed)


class ReproducibilityFramework:
    """
    Framework for ensuring computational reproducibility.
    
    Provides tools for:
    - Capturing complete experiment snapshots
    - Managing random seeds
    - Tracking data provenance
    - Verifying reproducibility
    """
    
    def __init__(
        self,
        experiment_dir: Path,
        master_seed: int = 42
    ):
        """
        Initialize reproducibility framework.
        
        Args:
            experiment_dir: Directory for storing experiment snapshots
            master_seed: Master random seed
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed_manager = RandomSeedManager(master_seed)
        self.current_snapshot: Optional[ExperimentSnapshot] = None
    
    def start_experiment(
        self,
        experiment_id: str,
        experiment_type: str,
        parameters: Dict[str, Any],
        random_seed: Optional[int] = None
    ) -> ExperimentSnapshot:
        """
        Start a new experiment with full provenance tracking.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_type: Type of experiment
            parameters: Experiment parameters
            random_seed: Random seed (uses master seed if None)
        
        Returns:
            ExperimentSnapshot object
        """
        # Set random seed
        seed = random_seed if random_seed is not None else self.seed_manager.master_seed
        self.seed_manager.set_global_seed(seed)
        
        # Create experiment parameters
        exp_params = ExperimentParameters(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            parameters=parameters,
            random_seed=seed
        )
        
        # Capture environment
        environment = ComputationalEnvironment.capture_current()
        
        # Initialize provenance
        provenance = DataProvenance(
            source_data=f"Experiment {experiment_id} started at {datetime.utcnow().isoformat()}"
        )
        
        # Create snapshot
        snapshot = ExperimentSnapshot(
            experiment_id=experiment_id,
            parameters=exp_params,
            environment=environment,
            provenance=provenance
        )
        
        self.current_snapshot = snapshot
        
        return snapshot
    
    def record_transformation(
        self,
        description: str,
        data: Any
    ) -> None:
        """
        Record a data transformation step.
        
        Args:
            description: Description of the transformation
            data: Data after transformation (for hashing)
        """
        if self.current_snapshot is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        # Compute hash of data
        data_hash = self._compute_data_hash(data)
        
        # Add to provenance
        self.current_snapshot.provenance.add_transformation(description, data_hash)
    
    def finalize_experiment(
        self,
        results: Dict[str, Any]
    ) -> ExperimentSnapshot:
        """
        Finalize experiment and save snapshot.
        
        Args:
            results: Experiment results
        
        Returns:
            Finalized ExperimentSnapshot
        """
        if self.current_snapshot is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        # Add results
        self.current_snapshot.results = results
        
        # Finalize provenance
        results_hash = self._compute_data_hash(results)
        self.current_snapshot.provenance.finalize(results_hash)
        
        # Save snapshot
        snapshot_path = self.experiment_dir / f"{self.current_snapshot.experiment_id}.json"
        self.current_snapshot.save(snapshot_path)
        
        snapshot = self.current_snapshot
        self.current_snapshot = None
        
        return snapshot
    
    def verify_reproducibility(
        self,
        experiment_id: str,
        rerun_function: callable
    ) -> Dict[str, Any]:
        """
        Verify that an experiment is reproducible.
        
        Args:
            experiment_id: ID of experiment to verify
            rerun_function: Function to rerun the experiment
        
        Returns:
            Dictionary with verification results
        """
        # Load original snapshot
        snapshot_path = self.experiment_dir / f"{experiment_id}.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
        
        original_snapshot = ExperimentSnapshot.load(snapshot_path)
        
        # Rerun experiment with same parameters and seed
        self.seed_manager.set_global_seed(original_snapshot.parameters.random_seed)
        
        new_results = rerun_function(original_snapshot.parameters.parameters)
        
        # Compare results
        original_results = original_snapshot.results
        results_match = self._compare_results(original_results, new_results)
        
        # Compute hashes
        original_hash = self._compute_data_hash(original_results)
        new_hash = self._compute_data_hash(new_results)
        
        return {
            'experiment_id': experiment_id,
            'reproducible': results_match,
            'original_hash': original_hash,
            'new_hash': new_hash,
            'hash_match': original_hash == new_hash,
            'original_results': original_results,
            'new_results': new_results
        }
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute deterministic hash of data."""
        # Convert to JSON string for hashing
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
        except TypeError:
            # Fallback for non-serializable objects
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _compare_results(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Compare two result dictionaries for equality.
        
        Args:
            results1: First results dictionary
            results2: Second results dictionary
            rtol: Relative tolerance for numerical comparisons
            atol: Absolute tolerance for numerical comparisons
        
        Returns:
            True if results match within tolerance
        """
        if results1 is None and results2 is None:
            return True
        
        if results1 is None or results2 is None:
            return False
        
        if set(results1.keys()) != set(results2.keys()):
            return False
        
        for key in results1.keys():
            val1 = results1[key]
            val2 = results2[key]
            
            # Handle numpy arrays
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.allclose(val1, val2, rtol=rtol, atol=atol):
                    return False
            # Handle floats
            elif isinstance(val1, float) and isinstance(val2, float):
                if not np.isclose(val1, val2, rtol=rtol, atol=atol):
                    return False
            # Handle other types
            elif val1 != val2:
                return False
        
        return True
    
    def export_reproducibility_package(
        self,
        experiment_id: str,
        output_dir: Path
    ) -> None:
        """
        Export complete reproducibility package for an experiment.
        
        Includes:
        - Experiment snapshot
        - Environment specification
        - Requirements file
        - README with instructions
        
        Args:
            experiment_id: ID of experiment to export
            output_dir: Output directory for package
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load snapshot
        snapshot_path = self.experiment_dir / f"{experiment_id}.json"
        snapshot = ExperimentSnapshot.load(snapshot_path)
        
        # Copy snapshot
        snapshot.save(output_dir / "experiment_snapshot.json")
        
        # Create requirements.txt
        requirements_path = output_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            for dep in snapshot.environment.dependencies:
                f.write(f"{dep}\n")
        
        # Create README
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# Reproducibility Package: {experiment_id}\n\n")
            f.write(f"## Experiment Information\n\n")
            f.write(f"- **Type**: {snapshot.parameters.experiment_type}\n")
            f.write(f"- **Timestamp**: {snapshot.parameters.timestamp}\n")
            f.write(f"- **Random Seed**: {snapshot.parameters.random_seed}\n\n")
            f.write(f"## Environment\n\n")
            f.write(f"- **Python**: {snapshot.environment.python_version}\n")
            f.write(f"- **Platform**: {snapshot.environment.platform} {snapshot.environment.platform_version}\n\n")
            f.write(f"## Reproduction Instructions\n\n")
            f.write(f"1. Install dependencies:\n")
            f.write(f"   ```bash\n")
            f.write(f"   pip install -r requirements.txt\n")
            f.write(f"   ```\n\n")
            f.write(f"2. Load experiment snapshot and rerun with same parameters and seed\n\n")
            f.write(f"## Parameters\n\n")
            f.write(f"```json\n")
            f.write(json.dumps(snapshot.parameters.parameters, indent=2))
            f.write(f"\n```\n")
