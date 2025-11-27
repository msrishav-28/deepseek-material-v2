"""Performance and scalability tests."""

import pytest
import time
import numpy as np
from pathlib import Path

from ceramic_discovery.performance import (
    ParallelProcessor,
    CacheManager,
    MemoryManager,
    IncrementalLearner,
)
from ceramic_discovery.ml import ModelTrainer, FeatureEngineeringPipeline
from ceramic_discovery.dft import StabilityAnalyzer


class TestPerformanceScalability:
    """Test performance and scalability."""

    @pytest.mark.performance
    def test_stability_analysis_performance(self):
        """Test stability analysis performance at scale."""
        analyzer = StabilityAnalyzer()

        # Create large batch of materials
        n_materials = 1000
        materials = [
            {
                "material_id": f"test-{i}",
                "formula": f"Material{i}",
                "energy_above_hull": np.random.uniform(0, 0.3),
                "formation_energy_per_atom": np.random.uniform(-1.0, -0.3),
            }
            for i in range(n_materials)
        ]

        # Measure performance
        start_time = time.time()
        results = analyzer.batch_analyze(materials)
        elapsed_time = time.time() - start_time

        assert len(results) == n_materials
        # Should process at least 100 materials per second
        assert elapsed_time < n_materials / 100

    @pytest.mark.performance
    def test_feature_engineering_performance(self):
        """Test feature engineering performance."""
        engineer = FeatureEngineeringPipeline()

        # Create test data
        n_samples = 500
        materials = [
            {
                "hardness": np.random.uniform(20, 35),
                "fracture_toughness": np.random.uniform(3, 6),
                "density": np.random.uniform(2.5, 4.0),
                "youngs_modulus": np.random.uniform(300, 500),
                "thermal_conductivity_25C": np.random.uniform(50, 150),
            }
            for _ in range(n_samples)
        ]

        # Measure performance
        start_time = time.time()
        for material in materials:
            features = engineer.extract_features(material)
        elapsed_time = time.time() - start_time

        # Should process at least 50 materials per second
        assert elapsed_time < n_samples / 50

    @pytest.mark.performance
    def test_ml_training_performance(self):
        """Test ML training performance."""
        np.random.seed(42)

        # Create training data
        n_samples = 200
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Train model
        trainer = ModelTrainer(model_type="random_forest")

        start_time = time.time()
        trainer.train(X, y)
        elapsed_time = time.time() - start_time

        # Training should complete in reasonable time
        assert elapsed_time < 10.0  # seconds

    @pytest.mark.performance
    def test_parallel_processing_speedup(self):
        """Test parallel processing provides speedup."""
        processor = ParallelProcessor(n_workers=4)

        # Define compute-intensive task
        def expensive_task(x):
            return sum(i**2 for i in range(x))

        tasks = [10000] * 100

        # Sequential processing
        start_time = time.time()
        sequential_results = [expensive_task(t) for t in tasks]
        sequential_time = time.time() - start_time

        # Parallel processing
        start_time = time.time()
        parallel_results = processor.map(expensive_task, tasks)
        parallel_time = time.time() - start_time

        assert sequential_results == parallel_results
        # Parallel should be faster (at least 1.5x speedup)
        assert parallel_time < sequential_time / 1.5


class TestCachingPerformance:
    """Test caching performance improvements."""

    @pytest.mark.performance
    def test_cache_hit_performance(self, temp_data_dir):
        """Test cache provides performance improvement."""
        cache = CacheManager(cache_dir=temp_data_dir)

        # Expensive computation
        def expensive_computation(x):
            time.sleep(0.1)  # Simulate expensive operation
            return x**2

        key = "test_computation_5"

        # First call (cache miss)
        start_time = time.time()
        result1 = cache.get_or_compute(key, lambda: expensive_computation(5))
        first_call_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        result2 = cache.get_or_compute(key, lambda: expensive_computation(5))
        second_call_time = time.time() - start_time

        assert result1 == result2
        # Cache hit should be much faster
        assert second_call_time < first_call_time / 10

    @pytest.mark.performance
    def test_cache_memory_efficiency(self, temp_data_dir):
        """Test cache memory efficiency."""
        cache = CacheManager(cache_dir=temp_data_dir, max_memory_mb=10)

        # Store multiple items
        for i in range(100):
            cache.set(f"key_{i}", {"data": list(range(100))})

        # Cache should manage memory
        stats = cache.get_stats()
        assert stats["memory_usage_mb"] <= 10


class TestMemoryManagement:
    """Test memory management."""

    @pytest.mark.performance
    def test_memory_efficient_data_loading(self):
        """Test memory-efficient data loading."""
        manager = MemoryManager()

        # Create large dataset
        large_data = np.random.randn(10000, 100)

        # Load in chunks
        chunk_size = 1000
        chunks = manager.chunk_data(large_data, chunk_size=chunk_size)

        total_rows = 0
        for chunk in chunks:
            assert chunk.shape[0] <= chunk_size
            total_rows += chunk.shape[0]

        assert total_rows == large_data.shape[0]

    @pytest.mark.performance
    def test_memory_monitoring(self):
        """Test memory monitoring."""
        manager = MemoryManager()

        # Get memory usage
        memory_usage = manager.get_memory_usage()

        assert "total_mb" in memory_usage
        assert "available_mb" in memory_usage
        assert "percent_used" in memory_usage
        assert 0 <= memory_usage["percent_used"] <= 100


class TestIncrementalLearning:
    """Test incremental learning performance."""

    @pytest.mark.performance
    def test_incremental_model_update(self):
        """Test incremental model updates."""
        np.random.seed(42)

        # Initial training data
        X_initial = np.random.randn(100, 5)
        y_initial = np.random.randn(100)

        # Create incremental learner
        learner = IncrementalLearner(model_type="sgd")
        learner.fit(X_initial, y_initial)

        # New data arrives
        X_new = np.random.randn(20, 5)
        y_new = np.random.randn(20)

        # Update model incrementally
        start_time = time.time()
        learner.partial_fit(X_new, y_new)
        update_time = time.time() - start_time

        # Incremental update should be fast
        assert update_time < 1.0  # seconds

        # Model should still make predictions
        predictions = learner.predict(X_new)
        assert len(predictions) == len(y_new)

    @pytest.mark.performance
    def test_incremental_vs_batch_training(self):
        """Compare incremental vs batch training performance."""
        np.random.seed(42)

        # Create data in batches
        batches = [
            (np.random.randn(50, 5), np.random.randn(50)) for _ in range(5)
        ]

        # Incremental learning
        learner_incremental = IncrementalLearner(model_type="sgd")
        start_time = time.time()
        for X_batch, y_batch in batches:
            learner_incremental.partial_fit(X_batch, y_batch)
        incremental_time = time.time() - start_time

        # Batch learning (retrain from scratch each time)
        X_all = np.vstack([X for X, _ in batches])
        y_all = np.concatenate([y for _, y in batches])

        trainer = ModelTrainer(model_type="linear")
        start_time = time.time()
        trainer.train(X_all, y_all)
        batch_time = time.time() - start_time

        # Incremental should be comparable or faster for large datasets
        # (In this small example, batch might be faster, but incremental scales better)
        assert incremental_time < batch_time * 3  # Allow some overhead


class TestScalabilityBenchmarks:
    """Benchmark scalability with increasing data sizes."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_screening_scalability(self):
        """Test screening engine scalability."""
        analyzer = StabilityAnalyzer()

        # Test with increasing data sizes
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            materials = [
                {
                    "material_id": f"test-{i}",
                    "formula": f"Mat{i}",
                    "energy_above_hull": np.random.uniform(0, 0.3),
                    "formation_energy_per_atom": np.random.uniform(-1.0, -0.3),
                }
                for i in range(size)
            ]

            start_time = time.time()
            results = analyzer.batch_analyze(materials)
            elapsed = time.time() - start_time
            times.append(elapsed)

            assert len(results) == size

        # Check that time scales roughly linearly (not quadratically)
        # Time ratio should be close to size ratio
        time_ratio = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]

        # Allow up to 1.5x overhead for larger datasets
        assert time_ratio < size_ratio * 1.5

    @pytest.mark.performance
    @pytest.mark.slow
    def test_ml_training_scalability(self):
        """Test ML training scalability."""
        np.random.seed(42)

        sizes = [100, 200, 400]
        times = []

        for size in sizes:
            X = np.random.randn(size, 10)
            y = np.random.randn(size)

            trainer = ModelTrainer(model_type="random_forest")

            start_time = time.time()
            trainer.train(X, y)
            elapsed = time.time() - start_time
            times.append(elapsed)

        # Training time should scale reasonably
        # (Random Forest is roughly O(n log n))
        time_ratio = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]

        # Allow up to 2x overhead for larger datasets
        assert time_ratio < size_ratio * 2


class TestResourceUtilization:
    """Test resource utilization."""

    @pytest.mark.performance
    def test_cpu_utilization(self):
        """Test CPU utilization during parallel processing."""
        processor = ParallelProcessor(n_workers=4)

        # CPU-intensive task
        def cpu_task(n):
            return sum(i**2 for i in range(n))

        tasks = [100000] * 20

        # Process in parallel
        results = processor.map(cpu_task, tasks)

        assert len(results) == len(tasks)
        # All results should be identical
        assert len(set(results)) == 1

    @pytest.mark.performance
    def test_memory_footprint(self):
        """Test memory footprint of operations."""
        manager = MemoryManager()

        # Get initial memory
        initial_memory = manager.get_memory_usage()["percent_used"]

        # Create large array
        large_array = np.random.randn(1000, 1000)

        # Get memory after allocation
        after_memory = manager.get_memory_usage()["percent_used"]

        # Memory should have increased
        assert after_memory >= initial_memory

        # Clean up
        del large_array

    @pytest.mark.performance
    def test_disk_io_performance(self, temp_data_dir):
        """Test disk I/O performance."""
        cache = CacheManager(cache_dir=temp_data_dir)

        # Write performance
        data = {"array": np.random.randn(1000, 100).tolist()}

        start_time = time.time()
        for i in range(10):
            cache.set(f"data_{i}", data)
        write_time = time.time() - start_time

        # Read performance
        start_time = time.time()
        for i in range(10):
            retrieved = cache.get(f"data_{i}")
        read_time = time.time() - start_time

        # Both should complete in reasonable time
        assert write_time < 5.0
        assert read_time < 2.0  # Reading should be faster
