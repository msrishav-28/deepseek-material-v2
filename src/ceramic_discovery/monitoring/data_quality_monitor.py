"""Data quality monitoring for materials database."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataQualityMetrics:
    """Data quality metrics for materials data."""
    
    dataset_name: str
    timestamp: datetime
    
    # Completeness metrics
    total_records: int = 0
    complete_records: int = 0
    missing_values_count: int = 0
    missing_values_percent: float = 0.0
    
    # Validity metrics
    invalid_values_count: int = 0
    out_of_range_count: int = 0
    
    # Consistency metrics
    duplicate_records: int = 0
    inconsistent_units: int = 0
    
    # Accuracy metrics
    outliers_count: int = 0
    outliers_percent: float = 0.0
    
    # Property-specific metrics
    property_coverage: Dict[str, float] = field(default_factory=dict)
    property_quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Issues
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_records": self.total_records,
            "complete_records": self.complete_records,
            "missing_values_count": self.missing_values_count,
            "missing_values_percent": self.missing_values_percent,
            "invalid_values_count": self.invalid_values_count,
            "out_of_range_count": self.out_of_range_count,
            "duplicate_records": self.duplicate_records,
            "inconsistent_units": self.inconsistent_units,
            "outliers_count": self.outliers_count,
            "outliers_percent": self.outliers_percent,
            "property_coverage": self.property_coverage,
            "property_quality_scores": self.property_quality_scores,
            "issues": self.issues,
            "warnings": self.warnings
        }


class DataQualityMonitor:
    """Monitor data quality for materials database."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize data quality monitor.
        
        Args:
            log_dir: Directory to store quality logs
        """
        self.log_dir = log_dir or Path("./logs/data_quality")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[DataQualityMetrics] = []
        
        # Define expected ranges for common properties
        self.property_ranges = {
            "hardness": (0, 50),  # GPa
            "density": (1, 20),  # g/cm³
            "fracture_toughness": (0, 15),  # MPa·m^0.5
            "youngs_modulus": (0, 1000),  # GPa
            "thermal_conductivity": (0, 500),  # W/m·K
            "melting_point": (0, 5000),  # K
            "formation_energy": (-10, 10),  # eV/atom
            "energy_above_hull": (0, 2)  # eV/atom
        }
    
    def check_data_quality(self, data: Dict[str, Any], dataset_name: str) -> DataQualityMetrics:
        """Check data quality for a dataset.
        
        Args:
            data: Dataset to check (dict of property arrays)
            dataset_name: Name of the dataset
            
        Returns:
            Data quality metrics
        """
        metrics = DataQualityMetrics(
            dataset_name=dataset_name,
            timestamp=datetime.now()
        )
        
        # Determine total records
        if data:
            first_key = next(iter(data.keys()))
            if isinstance(data[first_key], (list, np.ndarray)):
                metrics.total_records = len(data[first_key])
            else:
                metrics.total_records = 1
        
        # Check completeness
        self._check_completeness(data, metrics)
        
        # Check validity
        self._check_validity(data, metrics)
        
        # Check consistency
        self._check_consistency(data, metrics)
        
        # Check for outliers
        self._check_outliers(data, metrics)
        
        # Calculate property coverage
        self._calculate_property_coverage(data, metrics)
        
        # Save metrics
        self._save_metrics(metrics)
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _check_completeness(self, data: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Check data completeness.
        
        Args:
            data: Dataset to check
            metrics: Metrics object to update
        """
        total_values = 0
        missing_values = 0
        
        for key, values in data.items():
            if isinstance(values, (list, np.ndarray)):
                total_values += len(values)
                if isinstance(values, np.ndarray):
                    missing_values += np.isnan(values).sum()
                else:
                    missing_values += sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
        
        metrics.missing_values_count = missing_values
        if total_values > 0:
            metrics.missing_values_percent = (missing_values / total_values) * 100
        
        # Count complete records (no missing values)
        if metrics.total_records > 0:
            complete_count = 0
            for i in range(metrics.total_records):
                is_complete = True
                for values in data.values():
                    if isinstance(values, (list, np.ndarray)) and len(values) > i:
                        val = values[i]
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            is_complete = False
                            break
                if is_complete:
                    complete_count += 1
            metrics.complete_records = complete_count
        
        # Add warnings
        if metrics.missing_values_percent > 10:
            metrics.warnings.append(f"High missing data rate: {metrics.missing_values_percent:.1f}%")
        if metrics.missing_values_percent > 30:
            metrics.issues.append(f"Critical missing data rate: {metrics.missing_values_percent:.1f}%")
    
    def _check_validity(self, data: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Check data validity.
        
        Args:
            data: Dataset to check
            metrics: Metrics object to update
        """
        invalid_count = 0
        out_of_range_count = 0
        
        for key, values in data.items():
            if key not in self.property_ranges:
                continue
            
            min_val, max_val = self.property_ranges[key]
            
            if isinstance(values, (list, np.ndarray)):
                for val in values:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    
                    # Check if value is valid number
                    try:
                        float_val = float(val)
                        if not np.isfinite(float_val):
                            invalid_count += 1
                        elif float_val < min_val or float_val > max_val:
                            out_of_range_count += 1
                    except (ValueError, TypeError):
                        invalid_count += 1
        
        metrics.invalid_values_count = invalid_count
        metrics.out_of_range_count = out_of_range_count
        
        if invalid_count > 0:
            metrics.issues.append(f"Found {invalid_count} invalid values")
        if out_of_range_count > 0:
            metrics.warnings.append(f"Found {out_of_range_count} out-of-range values")
    
    def _check_consistency(self, data: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Check data consistency.
        
        Args:
            data: Dataset to check
            metrics: Metrics object to update
        """
        # Check for duplicates (simplified - would need more sophisticated logic)
        if "material_id" in data:
            ids = data["material_id"]
            if isinstance(ids, (list, np.ndarray)):
                unique_ids = set(ids)
                metrics.duplicate_records = len(ids) - len(unique_ids)
                
                if metrics.duplicate_records > 0:
                    metrics.warnings.append(f"Found {metrics.duplicate_records} duplicate records")
    
    def _check_outliers(self, data: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Check for outliers using 3-sigma rule.
        
        Args:
            data: Dataset to check
            metrics: Metrics object to update
        """
        outliers_count = 0
        
        for key, values in data.items():
            if not isinstance(values, (list, np.ndarray)):
                continue
            
            # Convert to numpy array
            arr = np.array(values, dtype=float)
            
            # Remove NaN values
            arr = arr[~np.isnan(arr)]
            
            if len(arr) < 3:
                continue
            
            # Calculate mean and std
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                continue
            
            # Count outliers (>3 sigma)
            outliers = np.abs(arr - mean) > 3 * std
            outliers_count += outliers.sum()
        
        metrics.outliers_count = outliers_count
        if metrics.total_records > 0:
            metrics.outliers_percent = (outliers_count / metrics.total_records) * 100
        
        if metrics.outliers_percent > 5:
            metrics.warnings.append(f"High outlier rate: {metrics.outliers_percent:.1f}%")
    
    def _calculate_property_coverage(self, data: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Calculate property coverage and quality scores.
        
        Args:
            data: Dataset to check
            metrics: Metrics object to update
        """
        for key, values in data.items():
            if not isinstance(values, (list, np.ndarray)):
                continue
            
            # Calculate coverage (non-missing values)
            arr = np.array(values, dtype=float)
            non_missing = ~np.isnan(arr)
            coverage = non_missing.sum() / len(arr) * 100 if len(arr) > 0 else 0
            
            metrics.property_coverage[key] = coverage
            
            # Calculate quality score (coverage * validity)
            if key in self.property_ranges:
                min_val, max_val = self.property_ranges[key]
                valid_values = arr[non_missing]
                in_range = ((valid_values >= min_val) & (valid_values <= max_val)).sum()
                validity = in_range / len(valid_values) * 100 if len(valid_values) > 0 else 0
                quality_score = (coverage * validity) / 100
            else:
                quality_score = coverage
            
            metrics.property_quality_scores[key] = quality_score
    
    def _save_metrics(self, metrics: DataQualityMetrics) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Metrics to save
        """
        metrics_file = self.log_dir / f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to Python types for JSON serialization
        metrics_dict = metrics.to_dict()
        metrics_dict = self._convert_numpy_types(metrics_dict)
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of data quality over time.
        
        Returns:
            Summary statistics
        """
        if not self.metrics_history:
            return {"message": "No quality checks performed yet"}
        
        latest = self.metrics_history[-1]
        
        return {
            "total_checks": len(self.metrics_history),
            "latest_check": latest.to_dict(),
            "trends": {
                "missing_data_trend": [m.missing_values_percent for m in self.metrics_history[-10:]],
                "outliers_trend": [m.outliers_percent for m in self.metrics_history[-10:]],
                "completeness_trend": [
                    (m.complete_records / m.total_records * 100) if m.total_records > 0 else 0
                    for m in self.metrics_history[-10:]
                ]
            }
        }
