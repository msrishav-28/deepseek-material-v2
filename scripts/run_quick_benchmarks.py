"""
Quick performance benchmark runner (skips full pipeline).

Runs all benchmarks except the full pipeline test.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the main benchmark module
from run_performance_benchmarks import (
    BenchmarkRunner,
    benchmark_jarvis_loading,
    benchmark_feature_engineering,
    benchmark_application_ranking,
    benchmark_composition_descriptors,
    benchmark_structure_descriptors
)


def main():
    """Run quick performance benchmarks (skip full pipeline)."""
    print("\n" + "=" * 80)
    print("SIC ALLOY DESIGNER QUICK PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print("\nRunning all benchmarks except full pipeline:")
    print("1. JARVIS data loading (target: <5 min for 25K materials)")
    print("2. Feature engineering (target: <1 min for 1K materials)")
    print("3. Application ranking (target: <30 sec for 1K materials)")
    print("4. Composition descriptors (target: <10 sec for 1K materials)")
    print("5. Structure descriptors (target: <5 sec for 1K materials)")
    print("=" * 80)
    
    runner = BenchmarkRunner()
    
    # Run benchmarks
    try:
        # Quick benchmarks first
        benchmark_composition_descriptors(runner, n_materials=1000)
        benchmark_structure_descriptors(runner, n_materials=1000)
        benchmark_feature_engineering(runner, n_materials=1000)
        benchmark_application_ranking(runner, n_materials=1000)
        
        # Slower benchmark
        benchmark_jarvis_loading(runner, n_materials=25000)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save and print results
    runner.save_results("performance_benchmark_results.json")
    runner.print_summary()
    
    # Return exit code based on results
    failed = sum(1 for b in runner.benchmarks if not b.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
