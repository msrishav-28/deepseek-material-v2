"""
Standalone performance benchmark runner for SiC Alloy Designer Integration.

This script runs performance benchmarks and generates a report.
Can be run directly without pytest.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from ceramic_discovery.dft.jarvis_client import JarvisClient
from ceramic_discovery.ml.feature_engineering import FeatureEngineeringPipeline
from ceramic_discovery.ml.composition_descriptors import CompositionDescriptorCalculator
from ceramic_discovery.ml.structure_descriptors import StructureDescriptorCalculator
from ceramic_discovery.screening.application_ranker import ApplicationRanker
from ceramic_discovery.screening.sic_alloy_designer_pipeline import SiCAlloyDesignerPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.duration_seconds = 0.0
        self.items_processed = 0
        self.throughput = 0.0
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
            'duration_seconds': round(self.duration_seconds, 2),
            'items_processed': self.items_processed,
            'throughput_items_per_sec': round(self.throughput, 2),
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
        """Run a benchmark function."""
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
            logger.error(f"Benchmark {name} failed: {e}", exc_info=True)
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
                'total_duration': round(sum(b.duration_seconds for b in self.benchmarks), 2)
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


# Helper functions

def create_mock_jarvis_data(n_materials: int = 25000) -> List[Dict[str, Any]]:
    """Create mock JARVIS data for benchmarking."""
    materials = []
    
    metals = ["Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr"]
    
    for i in range(n_materials):
        metal = metals[i % len(metals)]
        formula = f"{metal}C"
        
        material = {
            "jid": f"JVASP-{i:05d}-{formula}",
            "formula": formula,
            "formation_energy_peratom": float(np.random.uniform(-3.0, -0.5)),
            "optb88vdw_bandgap": float(np.random.uniform(0.0, 5.0)),
            "bulk_modulus_kv": float(np.random.uniform(200.0, 500.0)),
            "shear_modulus_gv": float(np.random.uniform(150.0, 300.0)),
            "ehull": float(np.random.uniform(0.0, 0.5)),
            "density": float(np.random.uniform(2.0, 8.0)),
            "natoms": int(np.random.randint(2, 10))
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


# Benchmark functions

def benchmark_jarvis_loading(runner: BenchmarkRunner, n_materials: int = 25000):
    """Benchmark JARVIS data loading."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 1: JARVIS Data Loading ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        jarvis_data = create_mock_jarvis_data(n_materials)
        json.dump(jarvis_data, f)
        jarvis_file = f.name
    
    try:
        def load_jarvis():
            client = JarvisClient(jarvis_file)
            carbides = client.load_carbides({"Si", "Ti", "Zr", "Hf", "Ta"})
            return carbides
        
        benchmark = runner.run_benchmark(
            f"JARVIS Data Loading ({n_materials} materials)",
            load_jarvis,
            target_seconds=300.0  # 5 minutes
        )
        
        return benchmark
        
    finally:
        Path(jarvis_file).unlink(missing_ok=True)


def benchmark_feature_engineering(runner: BenchmarkRunner, n_materials: int = 1000):
    """Benchmark feature engineering."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 2: Feature Engineering ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    df = create_mock_materials_dataframe(n_materials)
    formulas = df['formula']
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
        f"Feature Engineering ({n_materials} materials)",
        engineer_features,
        target_seconds=60.0  # 1 minute
    )
    
    return benchmark


def benchmark_application_ranking(runner: BenchmarkRunner, n_materials: int = 1000):
    """Benchmark application ranking."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 3: Application Ranking ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    df = create_mock_materials_dataframe(n_materials)
    materials = df.to_dict('records')
    
    def rank_materials():
        ranker = ApplicationRanker()
        rankings = ranker.rank_for_all_applications(materials)
        return rankings
    
    benchmark = runner.run_benchmark(
        f"Application Ranking ({n_materials} materials, 5 applications)",
        rank_materials,
        target_seconds=30.0  # 30 seconds
    )
    
    return benchmark


def benchmark_composition_descriptors(runner: BenchmarkRunner, n_materials: int = 1000):
    """Benchmark composition descriptor calculation."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 4: Composition Descriptors ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    formulas = [f"Si{i%5+1}C{i%3+1}" for i in range(n_materials)]
    
    def calculate_descriptors():
        calculator = CompositionDescriptorCalculator()
        descriptors = []
        
        for formula in formulas:
            desc = calculator.calculate_descriptors(formula)
            descriptors.append(desc)
        
        return descriptors
    
    benchmark = runner.run_benchmark(
        f"Composition Descriptors ({n_materials} materials)",
        calculate_descriptors,
        target_seconds=10.0  # 10 seconds
    )
    
    return benchmark


def benchmark_structure_descriptors(runner: BenchmarkRunner, n_materials: int = 1000):
    """Benchmark structure descriptor calculation."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 5: Structure Descriptors ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    properties_list = []
    for i in range(n_materials):
        properties = {
            'bulk_modulus': float(np.random.uniform(200.0, 500.0)),
            'shear_modulus': float(np.random.uniform(150.0, 300.0)),
            'formation_energy': float(np.random.uniform(-3.0, -0.5)),
            'density': float(np.random.uniform(2.0, 8.0))
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
        f"Structure Descriptors ({n_materials} materials)",
        calculate_descriptors,
        target_seconds=5.0  # 5 seconds
    )
    
    return benchmark


def benchmark_full_pipeline(runner: BenchmarkRunner, n_materials: int = 5000):
    """Benchmark full pipeline execution."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark 6: Full Pipeline ({n_materials} materials)")
    print(f"{'=' * 80}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        jarvis_data = create_mock_jarvis_data(n_materials)
        json.dump(jarvis_data, f)
        jarvis_file = f.name
    
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
                f"Full Pipeline ({n_materials} materials)",
                run_pipeline,
                target_seconds=1800.0  # 30 minutes
            )
            
            return benchmark
            
        finally:
            Path(jarvis_file).unlink(missing_ok=True)


def main():
    """Run all performance benchmarks."""
    print("\n" + "=" * 80)
    print("SIC ALLOY DESIGNER PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print("\nThis script benchmarks the performance of major components:")
    print("1. JARVIS data loading (target: <5 min for 25K materials)")
    print("2. Feature engineering (target: <1 min for 1K materials)")
    print("3. Application ranking (target: <30 sec for 1K materials)")
    print("4. Composition descriptors (target: <10 sec for 1K materials)")
    print("5. Structure descriptors (target: <5 sec for 1K materials)")
    print("6. Full pipeline (target: <30 min for 5K materials)")
    print("=" * 80)
    
    runner = BenchmarkRunner()
    
    # Run benchmarks
    try:
        # Quick benchmarks first
        benchmark_composition_descriptors(runner, n_materials=1000)
        benchmark_structure_descriptors(runner, n_materials=1000)
        benchmark_feature_engineering(runner, n_materials=1000)
        benchmark_application_ranking(runner, n_materials=1000)
        
        # Slower benchmarks
        benchmark_jarvis_loading(runner, n_materials=25000)
        
        # Full pipeline (optional - very slow)
        print("\n" + "=" * 80)
        print("Full pipeline benchmark is optional (takes ~30 minutes)")
        response = input("Run full pipeline benchmark? (y/n): ")
        if response.lower() == 'y':
            benchmark_full_pipeline(runner, n_materials=5000)
        else:
            print("Skipping full pipeline benchmark")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
    
    # Save and print results
    runner.save_results("performance_benchmark_results.json")
    runner.print_summary()
    
    # Return exit code based on results
    failed = sum(1 for b in runner.benchmarks if not b.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
