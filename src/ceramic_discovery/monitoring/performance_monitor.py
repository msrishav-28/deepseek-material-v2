"""Performance monitoring for computational operations."""

import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class PerformanceMetrics:
    """Performance metrics for a computational operation."""
    
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # CPU metrics
    cpu_percent: Optional[float] = None
    cpu_count: Optional[int] = None
    
    # Memory metrics
    memory_used_mb: Optional[float] = None
    memory_percent: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    
    # I/O metrics
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "memory_used_mb": self.memory_used_mb,
            "memory_percent": self.memory_percent,
            "peak_memory_mb": self.peak_memory_mb,
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "custom_metrics": self.custom_metrics
        }


class PerformanceMonitor:
    """Monitor computational performance and resource usage."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize performance monitor.
        
        Args:
            log_dir: Directory to store performance logs
        """
        self.log_dir = log_dir or Path("./logs/performance")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_operation: Optional[PerformanceMetrics] = None
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Initial system state
        self.process = psutil.Process()
        self.initial_io = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
    
    def start_operation(self, operation_name: str) -> None:
        """Start monitoring an operation.
        
        Args:
            operation_name: Name of the operation to monitor
        """
        self.current_operation = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            cpu_count=psutil.cpu_count()
        )
        
        # Record initial state
        self.current_operation.memory_used_mb = self.process.memory_info().rss / 1024 / 1024
        self.current_operation.cpu_percent = self.process.cpu_percent(interval=0.1)
    
    def end_operation(self, custom_metrics: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """End monitoring current operation.
        
        Args:
            custom_metrics: Additional custom metrics to record
            
        Returns:
            Performance metrics for the operation
        """
        if not self.current_operation:
            raise ValueError("No operation currently being monitored")
        
        # Record end state
        self.current_operation.end_time = datetime.now()
        self.current_operation.duration_seconds = (
            self.current_operation.end_time - self.current_operation.start_time
        ).total_seconds()
        
        # CPU metrics
        self.current_operation.cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Memory metrics
        mem_info = self.process.memory_info()
        self.current_operation.memory_used_mb = mem_info.rss / 1024 / 1024
        self.current_operation.memory_percent = self.process.memory_percent()
        
        # Peak memory (if available)
        if hasattr(mem_info, 'peak_wset'):
            self.current_operation.peak_memory_mb = mem_info.peak_wset / 1024 / 1024
        
        # I/O metrics
        if self.initial_io and hasattr(self.process, 'io_counters'):
            current_io = self.process.io_counters()
            self.current_operation.disk_read_mb = (
                current_io.read_bytes - self.initial_io.read_bytes
            ) / 1024 / 1024
            self.current_operation.disk_write_mb = (
                current_io.write_bytes - self.initial_io.write_bytes
            ) / 1024 / 1024
        
        # Custom metrics
        if custom_metrics:
            self.current_operation.custom_metrics = custom_metrics
        
        # Save to history
        self.metrics_history.append(self.current_operation)
        
        # Log to file
        self._log_metrics(self.current_operation)
        
        result = self.current_operation
        self.current_operation = None
        
        return result
    
    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log metrics to file.
        
        Args:
            metrics: Performance metrics to log
        """
        log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored operations.
        
        Returns:
            Summary statistics
        """
        if not self.metrics_history:
            return {"message": "No operations monitored yet"}
        
        total_duration = sum(m.duration_seconds or 0 for m in self.metrics_history)
        avg_cpu = sum(m.cpu_percent or 0 for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_used_mb or 0 for m in self.metrics_history) / len(self.metrics_history)
        peak_memory = max(m.peak_memory_mb or m.memory_used_mb or 0 for m in self.metrics_history)
        
        return {
            "total_operations": len(self.metrics_history),
            "total_duration_seconds": total_duration,
            "average_cpu_percent": avg_cpu,
            "average_memory_mb": avg_memory,
            "peak_memory_mb": peak_memory,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration_seconds,
                    "memory_mb": m.memory_used_mb
                }
                for m in self.metrics_history
            ]
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information.
        
        Returns:
            System resource information
        """
        return {
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
                "per_cpu": psutil.cpu_percent(interval=1, percpu=True)
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
                "free_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024,
                "percent": psutil.disk_usage('/').percent
            }
        }


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, 
                 custom_metrics: Optional[Dict[str, Any]] = None):
        """Initialize operation timer.
        
        Args:
            monitor: Performance monitor instance
            operation_name: Name of the operation
            custom_metrics: Custom metrics to record
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.custom_metrics = custom_metrics or {}
    
    def __enter__(self):
        """Start monitoring."""
        self.monitor.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring."""
        if exc_type:
            self.custom_metrics["error"] = str(exc_val)
            self.custom_metrics["error_type"] = exc_type.__name__
        
        self.monitor.end_operation(self.custom_metrics)
        return False
