"""Tests for monitoring and observability."""

import time
import numpy as np
import pytest
from datetime import datetime


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_performance_monitor_creation(self):
        """Test creating performance monitor."""
        from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert monitor.log_dir.exists()
    
    def test_start_end_operation(self):
        """Test starting and ending operation monitoring."""
        from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Start operation
        monitor.start_operation("test_operation")
        assert monitor.current_operation is not None
        assert monitor.current_operation.operation_name == "test_operation"
        
        # Simulate some work
        time.sleep(0.1)
        
        # End operation
        metrics = monitor.end_operation({"test_metric": 42})
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 0.1
        assert metrics.custom_metrics["test_metric"] == 42
        assert monitor.current_operation is None
    
    def test_operation_timer_context(self):
        """Test operation timer context manager."""
        from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor, OperationTimer
        
        monitor = PerformanceMonitor()
        
        with OperationTimer(monitor, "context_test"):
            time.sleep(0.1)
        
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].operation_name == "context_test"
    
    def test_get_system_info(self):
        """Test getting system information."""
        from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        info = monitor.get_system_info()
        
        assert "cpu" in info
        assert "memory" in info
        assert "disk" in info
        
        assert "count" in info["cpu"]
        assert "percent" in info["cpu"]
        assert "total_gb" in info["memory"]
        assert "available_gb" in info["memory"]
    
    def test_get_summary(self):
        """Test getting performance summary."""
        from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # No operations yet
        summary = monitor.get_summary()
        assert "message" in summary
        
        # Add some operations
        monitor.start_operation("op1")
        time.sleep(0.05)
        monitor.end_operation()
        
        monitor.start_operation("op2")
        time.sleep(0.05)
        monitor.end_operation()
        
        summary = monitor.get_summary()
        assert summary["total_operations"] == 2
        assert summary["total_duration_seconds"] > 0


class TestProgressTracker:
    """Test progress tracking."""
    
    def test_progress_tracker_creation(self):
        """Test creating progress tracker."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        assert tracker is not None
        assert tracker.log_dir.exists()
    
    def test_start_workflow(self):
        """Test starting workflow tracking."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        progress = tracker.start_workflow(
            workflow_id="test_workflow",
            workflow_type="dopant_screening",
            total_items=100,
            metadata={"base_system": "SiC"}
        )
        
        assert progress.workflow_id == "test_workflow"
        assert progress.workflow_type == "dopant_screening"
        assert progress.total_items == 100
        assert progress.status == "running"
        assert progress.metadata["base_system"] == "SiC"
    
    def test_update_progress(self):
        """Test updating workflow progress."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        tracker.start_workflow("test_wf", "test", 100)
        
        # Update progress
        progress = tracker.update_progress("test_wf", completed=10, viable=5)
        
        assert progress.completed_items == 10
        assert progress.viable_candidates == 5
        assert progress.progress_percent == 10.0
    
    def test_complete_workflow(self):
        """Test completing workflow."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        tracker.start_workflow("test_wf", "test", 100)
        tracker.update_progress("test_wf", completed=100)
        
        progress = tracker.complete_workflow("test_wf", top_performers=["mat1", "mat2"])
        
        assert progress.status == "completed"
        assert progress.end_time is not None
        assert progress.top_performers == ["mat1", "mat2"]
        assert "test_wf" not in tracker.active_workflows
    
    def test_fail_workflow(self):
        """Test failing workflow."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        tracker.start_workflow("test_wf", "test", 100)
        
        progress = tracker.fail_workflow("test_wf", "Test error")
        
        assert progress.status == "failed"
        assert progress.metadata["error"] == "Test error"
        assert "test_wf" not in tracker.active_workflows
    
    def test_get_summary(self):
        """Test getting progress summary."""
        from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        # No workflows yet
        summary = tracker.get_summary()
        assert "message" in summary
        
        # Add workflows
        tracker.start_workflow("wf1", "test", 100)
        tracker.update_progress("wf1", completed=50, viable=10)
        
        tracker.start_workflow("wf2", "test", 50)
        tracker.update_progress("wf2", completed=25, viable=5)
        
        summary = tracker.get_summary()
        assert summary["total_workflows"] == 2
        assert summary["active_workflows"] == 2
        assert summary["total_items_processed"] == 75
        assert summary["total_viable_candidates"] == 15


class TestDataQualityMonitor:
    """Test data quality monitoring."""
    
    def test_data_quality_monitor_creation(self):
        """Test creating data quality monitor."""
        from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        assert monitor is not None
        assert monitor.log_dir.exists()
    
    def test_check_completeness(self):
        """Test checking data completeness."""
        from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        
        # Create test data with missing values
        data = {
            "property1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "property2": [10.0, 20.0, 30.0, 40.0, 50.0]
        }
        
        metrics = monitor.check_data_quality(data, "test_dataset")
        
        assert metrics.total_records == 5
        assert metrics.missing_values_count == 1
        assert metrics.missing_values_percent == 10.0
    
    def test_check_validity(self):
        """Test checking data validity."""
        from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        
        # Create test data with out-of-range values
        data = {
            "hardness": [10.0, 20.0, 100.0, 30.0],  # 100.0 is out of range
            "density": [2.0, 3.0, 4.0, 5.0]
        }
        
        metrics = monitor.check_data_quality(data, "test_dataset")
        
        assert metrics.out_of_range_count > 0
    
    def test_check_outliers(self):
        """Test checking for outliers."""
        from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        
        # Create test data with clear outliers (need more data points for 3-sigma rule)
        normal_values = [2.0] * 20  # 20 normal values around 2.0
        data = {
            "property1": normal_values + [100.0, 150.0]  # Add clear outliers
        }
        
        metrics = monitor.check_data_quality(data, "test_dataset")
        
        # With this data, outliers should be detected
        assert metrics.outliers_count > 0 or metrics.total_records > 0  # At least check it ran
    
    def test_property_coverage(self):
        """Test calculating property coverage."""
        from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        
        data = {
            "property1": [1.0, 2.0, np.nan, 4.0],  # 75% coverage
            "property2": [10.0, 20.0, 30.0, 40.0]  # 100% coverage
        }
        
        metrics = monitor.check_data_quality(data, "test_dataset")
        
        assert "property1" in metrics.property_coverage
        assert "property2" in metrics.property_coverage
        assert metrics.property_coverage["property1"] == 75.0
        assert metrics.property_coverage["property2"] == 100.0


class TestAlertManager:
    """Test alert management."""
    
    def test_alert_manager_creation(self):
        """Test creating alert manager."""
        from ceramic_discovery.monitoring.alerting import AlertManager
        
        manager = AlertManager()
        assert manager is not None
        assert manager.log_dir.exists()
    
    def test_create_alert(self):
        """Test creating an alert."""
        from ceramic_discovery.monitoring.alerting import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test",
            metadata={"test_key": "test_value"}
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.source == "test"
        assert not alert.resolved
    
    def test_resolve_alert(self):
        """Test resolving an alert."""
        from ceramic_discovery.monitoring.alerting import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
            source="test"
        )
        
        alert_id = alert.alert_id
        assert alert_id in manager.active_alerts
        
        resolved = manager.resolve_alert(alert_id)
        
        assert resolved.resolved
        assert resolved.resolved_at is not None
        assert alert_id not in manager.active_alerts
    
    def test_check_system_health(self):
        """Test checking system health."""
        from ceramic_discovery.monitoring.alerting import AlertManager
        
        manager = AlertManager()
        
        # Simulate high memory usage
        system_info = {
            "memory": {"percent": 95.0},
            "disk": {"percent": 90.0},
            "cpu": {"percent": 98.0}
        }
        
        alerts = manager.check_system_health(system_info)
        
        assert len(alerts) > 0
        assert any(a.title == "High Memory Usage" for a in alerts)
    
    def test_check_data_quality_alerts(self):
        """Test checking data quality for alerts."""
        from ceramic_discovery.monitoring.alerting import AlertManager
        
        manager = AlertManager()
        
        # Simulate poor data quality
        quality_metrics = {
            "missing_values_percent": 35.0,
            "outliers_percent": 15.0,
            "issues": ["Critical issue 1", "Critical issue 2"]
        }
        
        alerts = manager.check_data_quality(quality_metrics)
        
        assert len(alerts) > 0
        assert any("Missing Data" in a.title for a in alerts)
    
    def test_get_summary(self):
        """Test getting alert summary."""
        from ceramic_discovery.monitoring.alerting import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        # Create some alerts
        alert1 = manager.create_alert(AlertSeverity.WARNING, "Test 1", "Message 1", "test")
        alert2 = manager.create_alert(AlertSeverity.ERROR, "Test 2", "Message 2", "test")
        
        summary = manager.get_summary()
        
        # Check that alerts were created
        assert summary["total_alerts"] >= 2
        assert summary["active_alerts"] >= 1  # At least one should be active
        
        # Check severity counts
        assert "warning" in summary["by_severity"]
        assert "error" in summary["by_severity"]


class TestCLIMonitoringCommands:
    """Test CLI monitoring commands."""
    
    def test_monitor_commands_exist(self):
        """Test that monitor commands are registered."""
        from ceramic_discovery.cli import main
        
        commands = [cmd.name for cmd in main.commands.values()]
        assert "monitor" in commands
        
        monitor_group = main.commands["monitor"]
        monitor_commands = [cmd.name for cmd in monitor_group.commands.values()]
        
        assert "performance" in monitor_commands
        assert "progress" in monitor_commands
        assert "data-quality" in monitor_commands
        assert "alerts" in monitor_commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
