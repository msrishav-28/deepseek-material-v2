"""Performance benchmarking and testing utilities."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    
    name: str
    duration_seconds: float
    memory_mb: float
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'duration_seconds': self.duration_seconds,
            'memory_mb': self.memory_mb,
            'throughput': self.throughput,
            'metadata': self.metadata,
        }


class PerformanceBenchmark:
    """
    Performance benchmarking utilities.
    
    Provides tools for measuring execution time, memory usage,
    and throughput of computational operations.
    """
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.results: List[BenchmarkResult] = []
        logger.info("Initialized PerformanceBenchmark")
    
    def benchmark_function(
        self,
        func: Callable,
        name: str,
        *args,
        n_runs: int = 1,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function's performance.
        
        Args:
            func: Function to benchmark
            name: Benchmark name
            *args: Function arguments
            n_runs: Number of runs to average
            **kwargs: Function keyword arguments
        
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking: {name} ({n_runs} runs)")
        
        durations = []
        memory_usages = []
        
        for i in range(n_runs):
            # Measure memory before
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024**2
            
            # Measure time
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024**2
            
            durations.append(end_time - start_time)
            memory_usages.append(mem_after - mem_before)
        
        # Calculate averages
        avg_duration = np.mean(durations)
        avg_memory = np.mean(memory_usages)
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration_seconds=avg_duration,
            memory_mb=avg_memory,
            metadata={
                'n_runs': n_runs,
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
            }
        )
        
        self.results.append(benchmark_result)
        
        logger.info(
            f"Benchmark complete: {name} - "
            f"Duration: {avg_duration:.3f}s, Memory: {avg_memory:.2f} MB"
        )
        
        return benchmark_result
    
    def benchmark_data_processing(
        self,
        data: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        name: str
    ) -> BenchmarkResult:
        """
        Benchmark data processing operation.
        
        Args:
            data: Input data
            process_func: Processing function
            name: Benchmark name
        
        Returns:
            Benchmark result with throughput
        """
        n_rows = len(data)
        
        result = self.benchmark_function(process_func, name, data)
        
        # Calculate throughput (rows per second)
        result.throughput = n_rows / result.duration_seconds
        result.metadata['n_rows'] = n_rows
        
        logger.info(f"Throughput: {result.throughput:.0f} rows/second")
        
        return result
    
    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        test_data: Any,
        n_runs: int = 3
    ) -> pd.DataFrame:
        """
        Compare performance of multiple implementations.
        
        Args:
            implementations: Dictionary of name -> function
            test_data: Test data to pass to functions
            n_runs: Number of runs per implementation
        
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(implementations)} implementations")
        
        comparison_results = []
        
        for name, func in implementations.items():
            result = self.benchmark_function(func, name, test_data, n_runs=n_runs)
            comparison_results.append(result.to_dict())
        
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('duration_seconds')
        
        # Calculate speedup relative to slowest
        slowest_time = df['duration_seconds'].max()
        df['speedup'] = slowest_time / df['duration_seconds']
        
        return df
    
    def benchmark_parallel_scaling(
        self,
        func: Callable,
        data: Any,
        max_workers_list: List[int],
        name: str = "parallel_scaling"
    ) -> pd.DataFrame:
        """
        Benchmark parallel scaling with different worker counts.
        
        Args:
            func: Function that accepts max_workers parameter
            data: Test data
            max_workers_list: List of worker counts to test
            name: Benchmark name
        
        Returns:
            DataFrame with scaling results
        """
        logger.info(f"Benchmarking parallel scaling: {name}")
        
        results = []
        
        for n_workers in max_workers_list:
            result = self.benchmark_function(
                func,
                f"{name}_workers_{n_workers}",
                data,
                max_workers=n_workers
            )
            
            results.append({
                'n_workers': n_workers,
                'duration_seconds': result.duration_seconds,
                'memory_mb': result.memory_mb,
            })
        
        df = pd.DataFrame(results)
        
        # Calculate speedup relative to single worker
        baseline_time = df[df['n_workers'] == 1]['duration_seconds'].values[0]
        df['speedup'] = baseline_time / df['duration_seconds']
        df['efficiency'] = df['speedup'] / df['n_workers']
        
        return df
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all benchmark results.
        
        Returns:
            DataFrame with benchmark summary
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(summary_data)
        
        return df
    
    def save_results(self, output_path: Path) -> None:
        """
        Save benchmark results to file.
        
        Args:
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.get_summary()
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved benchmark results to {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        df = self.get_summary()
        
        if df.empty:
            print("No benchmark results available")
            return
        
        print("\n=== Performance Benchmark Summary ===")
        print(df.to_string(index=False))
        print("=====================================\n")


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.3f} seconds")
        
        return result
    
    return wrapper


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage.
    
    Args:
        func: Function to profile
    
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024**2
        mem_diff = mem_after - mem_before
        
        logger.info(f"{func.__name__} used {mem_diff:.2f} MB of memory")
        
        return result
    
    return wrapper
