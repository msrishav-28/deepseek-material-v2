"""Resource usage monitoring and optimization."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import warnings
    warnings.warn("psutil not available. Install with: pip install psutil")

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Snapshot of resource usage."""
    
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'disk_read_mb': self.disk_read_mb,
            'disk_write_mb': self.disk_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
        }


class ResourceMonitor:
    """
    Resource usage monitoring and optimization.
    
    Tracks CPU, memory, disk, and network usage to help
    optimize resource allocation and identify bottlenecks.
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available. Install with: pip install psutil")
        
        self.sampling_interval = sampling_interval
        self.usage_history: List[ResourceUsage] = []
        self._monitoring = False
        
        # Initialize counters
        self._disk_io_start = psutil.disk_io_counters()
        self._net_io_start = psutil.net_io_counters()
        
        logger.info("Initialized ResourceMonitor")
    
    def get_current_usage(self) -> ResourceUsage:
        """
        Get current resource usage snapshot.
        
        Returns:
            Resource usage snapshot
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024**2
        memory_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self._disk_io_start.read_bytes) / 1024**2
        disk_write_mb = (disk_io.write_bytes - self._disk_io_start.write_bytes) / 1024**2
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_sent_mb = (net_io.bytes_sent - self._net_io_start.bytes_sent) / 1024**2
        network_recv_mb = (net_io.bytes_recv - self._net_io_start.bytes_recv) / 1024**2
        
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
        )
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        self._monitoring = True
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring = False
        logger.info("Stopped resource monitoring")
    
    def record_usage(self) -> ResourceUsage:
        """
        Record current usage to history.
        
        Returns:
            Resource usage snapshot
        """
        usage = self.get_current_usage()
        self.usage_history.append(usage)
        return usage
    
    def monitor_function(self, func, *args, **kwargs):
        """
        Monitor resource usage during function execution.
        
        Args:
            func: Function to monitor
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (function result, usage statistics)
        """
        # Record initial state
        start_usage = self.get_current_usage()
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record final state
        end_time = time.time()
        end_usage = self.get_current_usage()
        
        # Calculate statistics
        duration = end_time - start_time
        
        stats = {
            'duration_seconds': duration,
            'cpu_percent_avg': (start_usage.cpu_percent + end_usage.cpu_percent) / 2,
            'memory_mb_peak': max(start_usage.memory_mb, end_usage.memory_mb),
            'memory_mb_delta': end_usage.memory_mb - start_usage.memory_mb,
            'disk_read_mb': end_usage.disk_read_mb - start_usage.disk_read_mb,
            'disk_write_mb': end_usage.disk_write_mb - start_usage.disk_write_mb,
            'network_sent_mb': end_usage.network_sent_mb - start_usage.network_sent_mb,
            'network_recv_mb': end_usage.network_recv_mb - start_usage.network_recv_mb,
        }
        
        logger.info(
            f"Function {func.__name__} completed in {duration:.2f}s, "
            f"Memory: {stats['memory_mb_delta']:.2f} MB"
        )
        
        return result, stats
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / 1024**3
        
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / 1024**3
        
        return {
            'cpu_count_physical': cpu_count,
            'cpu_count_logical': cpu_count_logical,
            'memory_total_gb': memory_total_gb,
            'disk_total_gb': disk_total_gb,
        }
    
    def get_usage_statistics(self) -> Dict:
        """
        Get statistics from usage history.
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.usage_history:
            return {}
        
        cpu_values = [u.cpu_percent for u in self.usage_history]
        memory_values = [u.memory_mb for u in self.usage_history]
        
        import numpy as np
        
        return {
            'cpu_percent_mean': np.mean(cpu_values),
            'cpu_percent_max': np.max(cpu_values),
            'cpu_percent_min': np.min(cpu_values),
            'memory_mb_mean': np.mean(memory_values),
            'memory_mb_max': np.max(memory_values),
            'memory_mb_min': np.min(memory_values),
            'n_samples': len(self.usage_history),
        }
    
    def check_resource_limits(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0
    ) -> List[str]:
        """
        Check if resource usage exceeds thresholds.
        
        Args:
            cpu_threshold: CPU usage threshold (%)
            memory_threshold: Memory usage threshold (%)
        
        Returns:
            List of warnings
        """
        usage = self.get_current_usage()
        warnings = []
        
        if usage.cpu_percent > cpu_threshold:
            warnings.append(
                f"High CPU usage: {usage.cpu_percent:.1f}% (threshold: {cpu_threshold}%)"
            )
        
        if usage.memory_percent > memory_threshold:
            warnings.append(
                f"High memory usage: {usage.memory_percent:.1f}% "
                f"(threshold: {memory_threshold}%)"
            )
        
        for warning in warnings:
            logger.warning(warning)
        
        return warnings
    
    def optimize_recommendations(self) -> List[str]:
        """
        Get optimization recommendations based on usage patterns.
        
        Returns:
            List of recommendations
        """
        if not self.usage_history:
            return ["Insufficient data for recommendations"]
        
        stats = self.get_usage_statistics()
        recommendations = []
        
        # CPU recommendations
        if stats['cpu_percent_mean'] < 30:
            recommendations.append(
                "Low CPU utilization. Consider increasing parallelism or batch size."
            )
        elif stats['cpu_percent_mean'] > 80:
            recommendations.append(
                "High CPU utilization. Consider reducing parallelism or optimizing algorithms."
            )
        
        # Memory recommendations
        if stats['memory_mb_mean'] > 0.8 * psutil.virtual_memory().total / 1024**2:
            recommendations.append(
                "High memory usage. Consider using chunked processing or reducing batch size."
            )
        
        return recommendations
    
    def save_history(self, output_path: Path) -> None:
        """
        Save usage history to file.
        
        Args:
            output_path: Output file path
        """
        import pandas as pd
        
        if not self.usage_history:
            logger.warning("No usage history to save")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [usage.to_dict() for usage in self.usage_history]
        df = pd.DataFrame(data)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved usage history to {output_path}")
    
    def print_summary(self) -> None:
        """Print resource usage summary."""
        current = self.get_current_usage()
        system_info = self.get_system_info()
        
        print("\n=== Resource Usage Summary ===")
        print(f"CPU: {current.cpu_percent:.1f}% "
              f"({system_info['cpu_count_logical']} cores)")
        print(f"Memory: {current.memory_mb:.0f} MB "
              f"({current.memory_percent:.1f}% of "
              f"{system_info['memory_total_gb']:.1f} GB)")
        print(f"Disk Read: {current.disk_read_mb:.2f} MB")
        print(f"Disk Write: {current.disk_write_mb:.2f} MB")
        print(f"Network Sent: {current.network_sent_mb:.2f} MB")
        print(f"Network Received: {current.network_recv_mb:.2f} MB")
        
        if self.usage_history:
            stats = self.get_usage_statistics()
            print(f"\nHistory Statistics ({stats['n_samples']} samples):")
            print(f"  CPU: {stats['cpu_percent_mean']:.1f}% avg, "
                  f"{stats['cpu_percent_max']:.1f}% max")
            print(f"  Memory: {stats['memory_mb_mean']:.0f} MB avg, "
                  f"{stats['memory_mb_max']:.0f} MB max")
        
        print("==============================\n")
