"""Alerting system for monitoring issues."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification."""
    
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize alert manager.
        
        Args:
            log_dir: Directory to store alert logs
        """
        self.log_dir = log_dir or Path("./logs/alerts")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }
        
        # Set up logging
        self.logger = logging.getLogger("ceramic_discovery.alerting")
        self._setup_logging()
        
        # Alert thresholds
        self.thresholds = {
            "memory_percent": 90.0,
            "disk_percent": 85.0,
            "cpu_percent": 95.0,
            "missing_data_percent": 30.0,
            "outliers_percent": 10.0,
            "failure_rate": 20.0
        }
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        handler = logging.FileHandler(self.log_dir / "alerts.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def create_alert(self, severity: AlertSeverity, title: str, message: str,
                    source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert.
        
        Args:
            severity: Alert severity level
            title: Alert title
            message: Alert message
            source: Source of the alert
            metadata: Additional metadata
            
        Returns:
            Created alert
        """
        alert_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self._log_alert(alert)
        
        # Trigger handlers
        self._trigger_handlers(alert)
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            Resolved alert or None if not found
        """
        if alert_id not in self.active_alerts:
            return None
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Log resolution
        self.logger.info(f"Alert resolved: {alert_id}")
        
        return alert
    
    def register_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler.
        
        Args:
            severity: Severity level to handle
            handler: Handler function
        """
        self.alert_handlers[severity].append(handler)
    
    def _trigger_handlers(self, alert: Alert) -> None:
        """Trigger registered handlers for an alert.
        
        Args:
            alert: Alert to handle
        """
        for handler in self.alert_handlers[alert.severity]:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to file and logger.
        
        Args:
            alert: Alert to log
        """
        # Log to file
        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
        
        # Log to logger
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[alert.severity]
        
        self.logger.log(log_level, f"{alert.title}: {alert.message}")
    
    def check_system_health(self, system_info: Dict[str, Any]) -> List[Alert]:
        """Check system health and create alerts if needed.
        
        Args:
            system_info: System information from health check
            
        Returns:
            List of created alerts
        """
        alerts = []
        
        # Check memory
        if "memory" in system_info:
            memory_percent = system_info["memory"].get("percent", 0)
            if memory_percent > self.thresholds["memory_percent"]:
                alert = self.create_alert(
                    severity=AlertSeverity.WARNING,
                    title="High Memory Usage",
                    message=f"Memory usage at {memory_percent:.1f}%",
                    source="system_monitor",
                    metadata={"memory_percent": memory_percent}
                )
                alerts.append(alert)
        
        # Check disk
        if "disk" in system_info:
            disk_percent = system_info["disk"].get("percent", 0)
            if disk_percent > self.thresholds["disk_percent"]:
                alert = self.create_alert(
                    severity=AlertSeverity.WARNING,
                    title="High Disk Usage",
                    message=f"Disk usage at {disk_percent:.1f}%",
                    source="system_monitor",
                    metadata={"disk_percent": disk_percent}
                )
                alerts.append(alert)
        
        # Check CPU
        if "cpu" in system_info:
            cpu_percent = system_info["cpu"].get("percent", 0)
            if cpu_percent > self.thresholds["cpu_percent"]:
                alert = self.create_alert(
                    severity=AlertSeverity.WARNING,
                    title="High CPU Usage",
                    message=f"CPU usage at {cpu_percent:.1f}%",
                    source="system_monitor",
                    metadata={"cpu_percent": cpu_percent}
                )
                alerts.append(alert)
        
        return alerts
    
    def check_data_quality(self, quality_metrics: Dict[str, Any]) -> List[Alert]:
        """Check data quality and create alerts if needed.
        
        Args:
            quality_metrics: Data quality metrics
            
        Returns:
            List of created alerts
        """
        alerts = []
        
        # Check missing data
        missing_percent = quality_metrics.get("missing_values_percent", 0)
        if missing_percent > self.thresholds["missing_data_percent"]:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR,
                title="High Missing Data Rate",
                message=f"Missing data at {missing_percent:.1f}%",
                source="data_quality_monitor",
                metadata={"missing_percent": missing_percent}
            )
            alerts.append(alert)
        
        # Check outliers
        outliers_percent = quality_metrics.get("outliers_percent", 0)
        if outliers_percent > self.thresholds["outliers_percent"]:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                title="High Outlier Rate",
                message=f"Outliers at {outliers_percent:.1f}%",
                source="data_quality_monitor",
                metadata={"outliers_percent": outliers_percent}
            )
            alerts.append(alert)
        
        # Check for critical issues
        issues = quality_metrics.get("issues", [])
        if issues:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR,
                title="Data Quality Issues",
                message=f"Found {len(issues)} critical data quality issues",
                source="data_quality_monitor",
                metadata={"issues": issues}
            )
            alerts.append(alert)
        
        return alerts
    
    def check_workflow_progress(self, progress: Dict[str, Any]) -> List[Alert]:
        """Check workflow progress and create alerts if needed.
        
        Args:
            progress: Workflow progress metrics
            
        Returns:
            List of created alerts
        """
        alerts = []
        
        # Check failure rate
        completed = progress.get("completed_items", 0)
        failed = progress.get("failed_items", 0)
        total_processed = completed + failed
        
        if total_processed > 0:
            failure_rate = (failed / total_processed) * 100
            if failure_rate > self.thresholds["failure_rate"]:
                alert = self.create_alert(
                    severity=AlertSeverity.WARNING,
                    title="High Workflow Failure Rate",
                    message=f"Failure rate at {failure_rate:.1f}%",
                    source="progress_tracker",
                    metadata={
                        "failure_rate": failure_rate,
                        "failed_items": failed,
                        "total_processed": total_processed
                    }
                )
                alerts.append(alert)
        
        # Check if workflow is stalled
        status = progress.get("status", "unknown")
        if status == "failed":
            alert = self.create_alert(
                severity=AlertSeverity.ERROR,
                title="Workflow Failed",
                message=f"Workflow {progress.get('workflow_id', 'unknown')} has failed",
                source="progress_tracker",
                metadata=progress
            )
            alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts.
        
        Args:
            severity: Filter by severity (optional)
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary.
        
        Returns:
            Summary statistics
        """
        severity_counts = {severity: 0 for severity in AlertSeverity}
        for alert in self.active_alerts.values():
            severity_counts[alert.severity] += 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "resolved_alerts": len(self.alert_history) - len(self.active_alerts),
            "by_severity": {
                severity.value: count
                for severity, count in severity_counts.items()
            },
            "recent_alerts": [
                alert.to_dict()
                for alert in self.alert_history[-10:]
            ]
        }
