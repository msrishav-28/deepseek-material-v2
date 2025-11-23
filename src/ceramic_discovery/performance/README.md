# Performance Optimization Module

This module provides computational efficiency improvements for the Ceramic Armor Discovery Framework.

## Components

### ParallelProcessor
Enables parallel processing for DFT data collection and expensive computations.

**Features:**
- Process-based and thread-based parallelism
- Configurable worker count
- Progress tracking
- Error handling with detailed reporting

**Usage:**
```python
from ceramic_discovery.performance import ParallelProcessor, ParallelConfig

config = ParallelConfig(max_workers=4, use_processes=True)
processor = ParallelProcessor(config)

results = processor.map(fetch_function, material_ids, show_progress=True)
```

### CacheManager
Smart caching system for expensive calculations with TTL and LRU eviction.

**Features:**
- Memory and disk caching
- Configurable TTL (time-to-live)
- Size-based eviction (LRU)
- Cache statistics and hit rate tracking
- Decorator support for easy integration

**Usage:**
```python
from ceramic_discovery.performance import CacheManager, CacheConfig

cache = CacheManager(CacheConfig(max_cache_size_mb=1000, ttl_hours=24))

@cache.cached
def expensive_calculation(x):
    return complex_computation(x)
```

### MemoryManager
Memory-efficient data handling for large datasets.

**Features:**
- Chunked data processing
- HDF5 storage support
- Memory optimization for DataFrames
- Streaming data processing
- Memory usage monitoring

**Usage:**
```python
from ceramic_discovery.performance import MemoryManager

manager = MemoryManager()

# Optimize DataFrame memory
df_optimized = manager.optimize_dataframe_memory(df)

# Process in chunks
result = manager.chunk_processor.process_dataframe_chunks(
    df, process_func, show_progress=True
)
```

### IncrementalLearner
Incremental learning for ML models with large datasets.

**Features:**
- Online learning algorithms (SGD, MLP)
- Batch processing without loading entire dataset
- File-based training
- Model persistence
- Automatic feature scaling

**Usage:**
```python
from ceramic_discovery.performance import IncrementalLearner

learner = IncrementalLearner(model_type='sgd')

# Train in batches
learner.fit_batches(X_train, y_train, show_progress=True)

# Or train from file
learner.fit_from_file('data.csv', target_column='target')
```

### PerformanceBenchmark
Benchmarking utilities for measuring execution time and memory usage.

**Features:**
- Function benchmarking
- Data processing throughput measurement
- Implementation comparison
- Parallel scaling analysis
- Results export

**Usage:**
```python
from ceramic_discovery.performance.benchmarks import PerformanceBenchmark

benchmark = PerformanceBenchmark()

result = benchmark.benchmark_function(
    my_function, "test_name", *args, n_runs=5
)

benchmark.print_summary()
```

## Performance Best Practices

1. **Use parallel processing** for I/O-bound operations (API calls, file reading)
2. **Enable caching** for expensive calculations that are repeated
3. **Process data in chunks** when working with large datasets
4. **Use incremental learning** for training on datasets that don't fit in memory
5. **Monitor resource usage** to identify bottlenecks
6. **Benchmark different implementations** to choose the most efficient approach

## Requirements

- `numpy`
- `pandas`
- `scikit-learn`
- `psutil` (for resource monitoring)
- `h5py` (optional, for HDF5 support)

## Examples

See `examples/performance_optimization_example.py` for comprehensive usage examples.
