"""Example demonstrating performance optimization features."""

import time
from pathlib import Path
import pandas as pd
import numpy as np

from ceramic_discovery.performance import (
    ParallelProcessor,
    ParallelConfig,
    CacheManager,
    CacheConfig,
    MemoryManager,
    IncrementalLearner,
)
from ceramic_discovery.resources import (
    CostManager,
    CostConfig,
    ResourceMonitor,
    BudgetTracker,
)


def example_parallel_processing():
    """Demonstrate parallel processing for DFT data collection."""
    print("\n=== Parallel Processing Example ===\n")
    
    # Configure parallel processor
    config = ParallelConfig(
        max_workers=4,
        use_processes=True,
        chunk_size=10
    )
    
    processor = ParallelProcessor(config)
    
    # Simulate DFT data collection
    def fetch_material_data(material_id: str) -> dict:
        """Simulate fetching material data."""
        time.sleep(0.1)  # Simulate API call
        return {
            'material_id': material_id,
            'energy': np.random.random(),
            'band_gap': np.random.random() * 5,
        }
    
    # Process materials in parallel
    material_ids = [f"mp-{i}" for i in range(100, 120)]
    
    print(f"Processing {len(material_ids)} materials in parallel...")
    start_time = time.time()
    
    results = processor.map(fetch_material_data, material_ids, show_progress=True)
    
    end_time = time.time()
    
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} materials")


def example_caching():
    """Demonstrate smart caching for expensive calculations."""
    print("\n=== Caching Example ===\n")
    
    # Configure cache
    config = CacheConfig(
        cache_dir=Path("./data/cache"),
        max_cache_size_mb=100,
        ttl_hours=24,
        use_memory_cache=True,
        use_disk_cache=True,
    )
    
    cache = CacheManager(config)
    
    # Define expensive calculation
    @cache.cached
    def expensive_calculation(x: float) -> float:
        """Simulate expensive calculation."""
        time.sleep(0.5)
        return x ** 2 + np.sin(x)
    
    # First call (cache miss)
    print("First call (cache miss)...")
    start_time = time.time()
    result1 = expensive_calculation(5.0)
    time1 = time.time() - start_time
    print(f"Result: {result1:.4f}, Time: {time1:.3f}s")
    
    # Second call (cache hit)
    print("\nSecond call (cache hit)...")
    start_time = time.time()
    result2 = expensive_calculation(5.0)
    time2 = time.time() - start_time
    print(f"Result: {result2:.4f}, Time: {time2:.3f}s")
    
    print(f"\nSpeedup: {time1/time2:.1f}x")
    
    # Print cache statistics
    cache.print_stats()


def example_memory_management():
    """Demonstrate memory-efficient data handling."""
    print("\n=== Memory Management Example ===\n")
    
    memory_manager = MemoryManager()
    
    # Create large dataset
    print("Creating large dataset...")
    data = pd.DataFrame({
        'material_id': [f"mp-{i}" for i in range(10000)],
        'energy': np.random.random(10000),
        'band_gap': np.random.random(10000) * 5,
        'density': np.random.random(10000) * 10,
    })
    
    print(f"Original memory usage: {memory_manager.get_memory_usage(data):.2f} MB")
    
    # Optimize memory
    print("\nOptimizing memory usage...")
    data_optimized = memory_manager.optimize_dataframe_memory(data)
    
    print(f"Optimized memory usage: {memory_manager.get_memory_usage(data_optimized):.2f} MB")
    
    # Process in chunks
    print("\nProcessing data in chunks...")
    
    def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of data."""
        chunk['energy_normalized'] = (chunk['energy'] - chunk['energy'].mean()) / chunk['energy'].std()
        return chunk
    
    result = memory_manager.chunk_processor.process_dataframe_chunks(
        data_optimized,
        process_chunk,
        show_progress=True
    )
    
    print(f"Processed {len(result)} rows")


def example_incremental_learning():
    """Demonstrate incremental learning for large datasets."""
    print("\n=== Incremental Learning Example ===\n")
    
    # Create incremental learner
    learner = IncrementalLearner(model_type='sgd')
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    n_samples = 5000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.random((n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(X.sum(axis=1) + np.random.random(n_samples) * 0.1)
    
    # Train in batches
    print(f"Training on {n_samples} samples in batches...")
    learner.fit_batches(X, y, show_progress=True)
    
    # Evaluate
    print("\nEvaluating model...")
    score = learner.score(X[:1000], y[:1000])
    print(f"R² score: {score:.4f}")
    
    # Print model info
    info = learner.get_info()
    print(f"Samples seen: {info['n_samples_seen']}")
    print(f"Features: {info['n_features']}")


def example_cost_management():
    """Demonstrate cost-aware job scheduling."""
    print("\n=== Cost Management Example ===\n")
    
    # Configure cost manager
    config = CostConfig(
        daily_budget=50.0,
        monthly_budget=1000.0,
    )
    
    cost_manager = CostManager(config)
    
    # Simulate API calls
    print("Simulating API calls...")
    for i in range(10):
        if cost_manager.can_make_api_call():
            cost_manager.record_api_call(job_id=f"job_{i}")
        else:
            print("Rate limit reached!")
            break
    
    # Simulate compute usage
    print("\nSimulating compute usage...")
    cost_manager.record_compute_usage(
        cpu_hours=5.0,
        gpu_hours=2.0,
        memory_gb_hours=10.0,
        job_id="training_job_1"
    )
    
    # Check budget status
    print("\nChecking budget status...")
    status = cost_manager.check_budget_status()
    
    # Print cost summary
    cost_manager.print_cost_summary()


def example_resource_monitoring():
    """Demonstrate resource usage monitoring."""
    print("\n=== Resource Monitoring Example ===\n")
    
    monitor = ResourceMonitor()
    
    # Get system info
    print("System Information:")
    system_info = monitor.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Monitor a function
    print("\nMonitoring function execution...")
    
    def compute_intensive_task():
        """Simulate compute-intensive task."""
        result = 0
        for i in range(1000000):
            result += i ** 2
        return result
    
    result, stats = monitor.monitor_function(compute_intensive_task)
    
    print(f"\nFunction Statistics:")
    print(f"  Duration: {stats['duration_seconds']:.3f}s")
    print(f"  CPU: {stats['cpu_percent_avg']:.1f}%")
    print(f"  Memory Delta: {stats['memory_mb_delta']:.2f} MB")
    
    # Print current usage
    monitor.print_summary()


def example_budget_tracking():
    """Demonstrate computational budget tracking."""
    print("\n=== Budget Tracking Example ===\n")
    
    tracker = BudgetTracker()
    
    # Create budgets
    print("Creating budgets...")
    tracker.create_budget(
        name="research_project",
        total_amount=1000.0,
        duration_days=30,
        currency="USD"
    )
    
    tracker.create_budget(
        name="api_calls",
        total_amount=100.0,
        duration_days=7,
        currency="USD"
    )
    
    # Record expenses
    print("\nRecording expenses...")
    tracker.record_expense(
        budget_name="research_project",
        amount=250.0,
        description="GPU compute time",
        category="compute"
    )
    
    tracker.record_expense(
        budget_name="research_project",
        amount=50.0,
        description="Storage costs",
        category="storage"
    )
    
    tracker.record_expense(
        budget_name="api_calls",
        amount=25.0,
        description="Materials Project API",
        category="api"
    )
    
    # Print summary
    tracker.print_summary()
    
    # Get expenses by category
    print("\nExpenses by category:")
    by_category = tracker.get_expenses_by_category("research_project")
    for category, amount in by_category.items():
        print(f"  {category}: ${amount:.2f}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Performance Optimization Examples")
    print("=" * 60)
    
    # Run examples
    example_parallel_processing()
    example_caching()
    example_memory_management()
    example_incremental_learning()
    example_cost_management()
    example_resource_monitoring()
    example_budget_tracking()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
