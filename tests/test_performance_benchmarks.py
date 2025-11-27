"""
Performance benchmarking tests for SiC Alloy Designer Integration.

This module benchmarks the performance of major components:
- JARVIS data loading
- Feature engineering
- Application ranking
- Full pipeline execution

Performance targets:
- JARVIS loading: <5 minutes for 25K materials
- Feature engineering: <1 minute for 1K materials
- Application ranking: <30 seconds for 1K materials
- Full pipeline: <30 minutes end-to-end
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import tempfile

import pytest
import pandas as pd
import numpy as np

from ceramic_discovery.dft.jarvis_client import JarvisClient
from ceramic_discovery.ml.feature_engineering import FeatureEngineeringPipeline
from ceramic_discovery.screening.application_ranker import ApplicationRanker
from ceramic_discovery.screening.sic_alloy_designer_pipeline import SiCAlloyDesignerPipeline


logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.duration_seconds = 0.0
        self.items_processed = 0
        self.throughput = 0.0  # items per second
        self.target_seconds = 0.0
        self.passed = False
        self.metadata = {}
    
    def calculate_throughput(self):
        """Calculate throughput in items per second."""
        if self.duration_seconds > 0:
            self.throughput = self.items_processed / self.duration_seconds
    
    def check_target(self):
        """Check if benchmark met target."""
        if self.target_seconds > 0:
            self.passed = self.duration_seconds <= self.target_seconds
        return self.passed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'duration_seconds': self.duration_seconds,
            'items_processed': self.items_processed,
            'throughput_items_per_sec': self.throughput,
            'target_seconds': self.target_seconds,
            'passed': self.passed,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{status} {self.name}: {self.duration_seconds:.2f}s "
            f"({self.items_processed} items, {self.throughput:.1f} items/s) "
            f"[Target: {self.target_seconds:.0f}s]"
        )


class BenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks: List[PerformanceBenchmark] = []
    
    def run_benchmark(
        self,
        name: str,
        func,
        target_seconds: float,
        *args,
        **kwargs
    ) -> PerformanceBenchmark:
        """
        Run a benchmark function.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            target_seconds: Target execution time
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            PerformanceBenchmark with results
        """
        benchmark = PerformanceBenchmark(name)
        benchmark.target_seconds = target_seconds
        
        logger.info(f"Running benchmark: {name}")
        logger.info(f"Target: {target_seconds:.0f}s")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            benchmark.duration_seconds = end_time - start_time
            
            # Extract items processed from result
            if isinstance(result, (list, pd.DataFrame)):
                benchmark.items_processed = len(result)
            elif isinstance(result, dict) and 'count' in result:
                benchmark.items_processed = result['count']
            elif isinstance(result, int):
                benchmark.items_processed = result
            
            benchmark.calculate_throughput()
            benchmark.check_target()
            
            logger.info(str(benchmark))
            
        except Exception as e:
            logger.error(f"Benchmark {name} failed: {e}")
            benchmark.metadata['error'] = str(e)
            benchmark.passed = False
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        results = {
            'benchmarks': [b.to_dict() for b in self.benchmarks],
            'summary': {
                'total_benchmarks': len(self.benchmarks),
                'passed': sum(1 for b in self.benchmarks if b.passed),
                'failed': sum(1 for b in self.benchmarks if not b.passed),
                'total_duration': sum(b.duration_seconds for b in self.benchmarks)
            }
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_path}")
        
        return results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        for benchmark in self.benchmarks:
            print(benchmark)
        
        passed = sum(1 for b in self.benchmarks if b.passed)
        failed = len(self.benchmarks) - passed
        
        print("=" * 80)
        print(f"Total: {len(self.benchmarks)} benchmarks")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print("=" * 80)


# Benchmark fixtures and helpers

def create_mock_jarvis_data(n_materials: int = 25000) -> List[Dict[str, Any]]:
    """Create mock JARVIS data for benchmarking."""
    materials = []
    
    metals = ["Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr"]
    
    for i in range(n_materials):
        # Create carbide formulas
        metal = metals[i % len(metals)]
        formula = f"{metal}C"
        
        material = {
            "jid": f"JVASP-{i:05d}-{formula}",
            "formula": formula,
            "formation_energy_peratom": np.random.uniform(-3.0, -0.5),
            "optb88vdw_bandgap": np.random.uniform(0.0, 5.0),
            "bulk_modulus_kv": np.random.uniform(200.0, 500.0),
            "shear_modulus_gv": np.random.uniform(150.0, 300.0),
            "ehull": np.random.uniform(0.0, 0.5),
            "density": np.random.uniform(2.0, 8.0),
            "natoms": np.random.randint(2, 10)
        }
        
        materials.append(material)
    
    return materials


def create_mock_materials_dataframe(n_materials: int = 1000) -> pd.DataFrame:
    """Create mock materials dataframe for benchmarking."""
    data = {
        'material_id': [f'mat-{i}' for i in range(n_materials)],
        'formula': [f'SiC{i%10}' for i in range(n_materials)],
        'formation_energy': np.random.uniform(-3.0, -0.5, n_materials),
        'band_gap': np.random.uniform(0.0, 5.0, n_materials),
        'bulk_modulus': np.random.uniform(200.0, 500.0, n_materials),
        'shear_modulus': np.random.uniform(150.0, 300.0, n_materials),
        'density': np.random.uniform(2.0, 8.0, n_materials),
        'hardness': np.random.uniform(20.0, 40.0, n_materials),
        'thermal_conductivity': np.random.uniform(50.0, 200.0, n_materials),
        'melting_point': np.random.uniform(2000.0, 4000.0, n_materials),
    }
    
    return pd.DataFrame(data)


# Benchmark tests

@pytest.mark.benchmark
def test_benchmark_jarvis_loading():
    """
    Benchmark JARVIS data loading.
    
    Target: <5 minutes (300 seconds) for 25K materials
    """
    runner = BenchmarkRunner()
    
    # Create temporary JARVIS file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        jarvis_data = create_mock_jarvis_data(25000)
        json.dump(jarvis_data, f)
        jarvis_file = f.name
    
    try:
        def load_jarvis():
            client = JarvisClient(jarvis_file)
            carbides = client.load_carbides({"Si", "Ti", "Zr", "Hf", "Ta"})
            return carbides
        
        benchmark = runner.run_benchmark(
            "JARVIS Data Loading (25K materials)",
            load_jarvis,
            target_seconds=300.0  # 5 minutes
        )
        
        runner.save_results("jarvis_loading_benchmark.json")
        runner.print_summary()
        
        assert benchmark.passed, f"JARVIS loading exceeded target: {benchmark.duration_seconds:.2f}s > 300s"
        
    finally:
        # Cleanup
        Path(jarvis_file).unlink(missing_ok=True)


@pytest.mark.benchmark
def test_benchmark_feature_engineering():
    """
    Benchmark feature engineering.
    
    Target: <1 minute (60 seconds) for 1K materials
    """
    runner = BenchmarkRunner()
    
    # Create mock data
    df = create_mock_materials_dataframe(1000)
    formulas = df['formula']
    
    # Select numeric features
    X = df[['formation_energy', 'band_gap', 'bulk_modulus', 'shear_modulus', 'density']]
    
    def engineer_features():
        pipeline = FeatureEngineeringPipeline(
            scaling_method='standard',
            handle_missing='drop',
            include_composition_descriptors=True,
            include_structure_descriptors=True
        )
        
        X_engineered = pipeline.fit_transform(X, formulas=formulas)
        return X_engineered
    
    benchmark = runner.run_benchmark(
        "Feature Engineering (1K materials)",
        engineer_features,
        target_seconds=60.0  # 1 minute
    )
    
    runner.save_results("feature_engineering_benchmark.json")
    runner.print_summary()
    
    assert benchmark.passed, f"Feature engineering exceeded target: {benchmark.duration_seconds:.2f}s > 60s"


@pytest.mark.benchmark
def test_benchmark_application_ranking():
    """
    Benchmark application ranking.
    
    Target: <30 seconds for 1K materials
    """
    runner = BenchmarkRunner()
    
    # Create mock materials
    df = create_mock_materials_dataframe(1000)
    materials = df.to_dict('records')
    
    def rank_materials():
        ranker = ApplicationRanker()
        rankings = ranker.rank_for_all_applications(materials)
        return rankings
    
    benchmark = runner.run_benchmark(
        "Application Ranking (1K materials, 5 applications)",
        rank_materials,
        target_seconds=30.0  # 30 seconds
    )
    
    runner.save_results("application_ranking_benchmark.json")
    runner.print_summary()
    
    assert benchmark.passed, f"Application ranking exceeded target: {benchmark.duration_seconds:.2f}s > 30s"


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_full_pipeline():
    """
    Benchmark full pipeline execution.
    
    Target: <30 minutes (1800 seconds) end-to-end
    
    This is a comprehensive test that runs the entire pipeline.
    """
    runner = BenchmarkRunner()
    
    # Create temporary JARVIS file with smaller dataset for faster testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        jarvis_data = create_mock_jarvis_data(5000)  # Smaller for faster test
        json.dump(jarvis_data, f)
        jarvis_file = f.name
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            def run_pipeline():
                pipeline = SiCAlloyDesignerPipeline(output_dir=output_dir)
                
                results = pipeline.run_full_pipeline(
                    jarvis_file=jarvis_file,
                    metal_elements={"Si", "Ti", "Zr", "Hf", "Ta"},
                    target_property="formation_energy",
                    top_n_candidates=10
                )
                
                return results
            
            benchmark = runner.run_benchmark(
                "Full Pipeline (5K materials)",
                run_pipeline,
                target_seconds=1800.0  # 30 minutes
            )
            
            runner.save_results("full_pipeline_benchmark.json")
            runner.print_summary()
            
            assert benchmark.passed, f"Full pipeline exceeded target: {benchmark.duration_seconds:.2f}s > 1800s"
            
        finally:
            # Cleanup
            Path(jarvis_file).unlink(missing_ok=True)


@pytest.mark.benchmark
def test_benchmark_composition_descriptors():
    """
    Benchmark composition descriptor calculation.
    
    Target: <10 seconds for 1K materials
    """
    runner = BenchmarkRunner()
    
    from ceramic_discovery.ml.composition_descriptors import CompositionDescriptorCalculator
    
    # Create formulas
    formulas = [f"Si{i%5+1}C{i%3+1}" for i in range(1000)]
    
    def calculate_descriptors():
        calculator = CompositionDescriptorCalculator()
        descriptors = []
        
        for formula in formulas:
            desc = calculator.calculate_descriptors(formula)
            descriptors.append(desc)
        
        return descriptors
    
    benchmark = runner.run_benchmark(
        "Composition Descriptors (1K materials)",
        calculate_descriptors,
        target_seconds=10.0  # 10 seconds
    )
    
    runner.save_results("composition_descriptors_benchmark.json")
    runner.print_summary()
    
    assert benchmark.passed, f"Composition descriptors exceeded target: {benchmark.duration_seconds:.2f}s > 10s"


@pytest.mark.benchmark
def test_benchmark_structure_descriptors():
    """
    Benchmark structure descriptor calculation.
    
    Target: <5 seconds for 1K materials
    """
    runner = BenchmarkRunner()
    
    from ceramic_discovery.ml.structure_descriptors import StructureDescriptorCalculator
    
    # Create property dictionaries
    properties_list = []
    for i in range(1000):
        properties = {
            'bulk_modulus': np.random.uniform(200.0, 500.0),
            'shear_modulus': np.random.uniform(150.0, 300.0),
            'formation_energy': np.random.uniform(-3.0, -0.5),
            'density': np.random.uniform(2.0, 8.0)
        }
        properties_list.append(properties)
    
    def calculate_descriptors():
        calculator = StructureDescriptorCalculator()
        descriptors = []
        
        for properties in properties_list:
            desc = calculator.calculate_descriptors(properties)
            descriptors.append(desc)
        
        return descriptors
    
    benchmark = runner.run_benchmark(
        "Structure Descriptors (1K materials)",
        calculate_descriptors,
        target_seconds=5.0  # 5 seconds
    )
    
    runner.save_results("structure_descriptors_benchmark.json")
    runner.print_summary()
    
    assert benchmark.passed, f"Structure descriptors exceeded target: {benchmark.duration_seconds:.2f}s > 5s"


# Main benchmark runner

def run_all_benchmarks():
    """Run all performance benchmarks and generate report."""
    print("\n" + "=" * 80)
    print("RUNNING ALL PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    runner = BenchmarkRunner()
    
    # Run each benchmark
    benchmarks_to_run = [
        ("JARVIS Loading", test_benchmark_jarvis_loading),
        ("Feature Engineering", test_benchmark_feature_engineering),
        ("Application Ranking", test_benchmark_application_ranking),
        ("Composition Descriptors", test_benchmark_composition_descriptors),
        ("Structure Descriptors", test_benchmark_structure_descriptors),
    ]
    
    for name, test_func in benchmarks_to_run:
        print(f"\n{'=' * 80}")
        print(f"Running: {name}")
        print(f"{'=' * 80}")
        
        try:
            test_func()
        except AssertionError as e:
            print(f"FAILED: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run benchmarks directly
    run_all_benchmarks()
