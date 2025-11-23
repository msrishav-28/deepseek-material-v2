"""Materials property analysis toolkit for statistical analysis and trend detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr


@dataclass
class PropertyStatistics:
    """Statistical summary of a material property."""
    
    property_name: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'property': self.property_name,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'q25': self.q25,
            'q75': self.q75,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'n_samples': self.n_samples,
        }


@dataclass
class CorrelationAnalysis:
    """Correlation analysis between two properties."""
    
    property1: str
    property2: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    is_significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'property1': self.property1,
            'property2': self.property2,
            'pearson_r': self.pearson_r,
            'pearson_p': self.pearson_p,
            'spearman_r': self.spearman_r,
            'spearman_p': self.spearman_p,
            'is_significant': self.is_significant,
        }


@dataclass
class TrendAnalysis:
    """Trend analysis for concentration-dependent effects."""
    
    property_name: str
    concentrations: List[float]
    values: List[float]
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    is_significant: bool
    trend_direction: str  # 'increasing', 'decreasing', 'no_trend'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'property': self.property_name,
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'trend_direction': self.trend_direction,
        }


@dataclass
class OutlierDetection:
    """Outlier detection results."""
    
    property_name: str
    outlier_indices: List[int]
    outlier_values: List[float]
    z_scores: List[float]
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'property': self.property_name,
            'n_outliers': len(self.outlier_indices),
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.outlier_values,
            'z_scores': self.z_scores,
            'threshold': self.threshold,
        }


class PropertyAnalyzer:
    """
    Materials property analysis toolkit for statistical analysis and trend detection.
    
    Provides comprehensive statistical analysis, correlation analysis, trend detection,
    and outlier identification for material properties.
    """
    
    def __init__(self, significance_level: float = 0.05, outlier_threshold: float = 3.0):
        """
        Initialize property analyzer.
        
        Args:
            significance_level: P-value threshold for statistical significance
            outlier_threshold: Z-score threshold for outlier detection (default: 3σ)
        """
        self.significance_level = significance_level
        self.outlier_threshold = outlier_threshold
    
    def compute_statistics(self, data: pd.DataFrame, property_name: str) -> PropertyStatistics:
        """
        Compute comprehensive statistics for a property.
        
        Args:
            data: DataFrame containing material properties
            property_name: Name of the property to analyze
            
        Returns:
            PropertyStatistics object with statistical summary
        """
        if property_name not in data.columns:
            raise ValueError(f"Property '{property_name}' not found in data")
        
        values = data[property_name].dropna()
        
        if len(values) == 0:
            raise ValueError(f"No valid values for property '{property_name}'")
        
        return PropertyStatistics(
            property_name=property_name,
            mean=float(values.mean()),
            median=float(values.median()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            q25=float(values.quantile(0.25)),
            q75=float(values.quantile(0.75)),
            skewness=float(stats.skew(values)),
            kurtosis=float(stats.kurtosis(values)),
            n_samples=len(values),
        )
    
    def compute_all_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for all numeric properties.
        
        Args:
            data: DataFrame containing material properties
            
        Returns:
            DataFrame with statistics for all properties
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        stats_list = []
        for col in numeric_cols:
            try:
                stat = self.compute_statistics(data, col)
                stats_list.append(stat.to_dict())
            except ValueError:
                continue
        
        return pd.DataFrame(stats_list)
    
    def analyze_correlation(
        self, 
        data: pd.DataFrame, 
        property1: str, 
        property2: str
    ) -> CorrelationAnalysis:
        """
        Analyze correlation between two properties.
        
        Args:
            data: DataFrame containing material properties
            property1: First property name
            property2: Second property name
            
        Returns:
            CorrelationAnalysis object with correlation metrics
        """
        if property1 not in data.columns or property2 not in data.columns:
            raise ValueError(f"Properties not found in data")
        
        # Remove rows with missing values in either property
        valid_data = data[[property1, property2]].dropna()
        
        if len(valid_data) < 3:
            raise ValueError("Insufficient data for correlation analysis")
        
        # Pearson correlation (linear)
        pearson_r, pearson_p = pearsonr(valid_data[property1], valid_data[property2])
        
        # Spearman correlation (monotonic)
        spearman_r, spearman_p = spearmanr(valid_data[property1], valid_data[property2])
        
        # Check significance
        is_significant = min(pearson_p, spearman_p) < self.significance_level
        
        return CorrelationAnalysis(
            property1=property1,
            property2=property2,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_r=float(spearman_r),
            spearman_p=float(spearman_p),
            is_significant=is_significant,
        )
    
    def compute_correlation_matrix(
        self, 
        data: pd.DataFrame, 
        properties: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for multiple properties.
        
        Args:
            data: DataFrame containing material properties
            properties: List of property names (if None, use all numeric columns)
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            DataFrame with correlation matrix
        """
        if properties is None:
            properties = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to valid properties
        valid_props = [p for p in properties if p in data.columns]
        
        if method == 'pearson':
            corr_matrix = data[valid_props].corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = data[valid_props].corr(method='spearman')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return corr_matrix
    
    def analyze_trend(
        self, 
        data: pd.DataFrame, 
        property_name: str,
        concentration_col: str = 'dopant_concentration'
    ) -> TrendAnalysis:
        """
        Analyze concentration-dependent trends.
        
        Args:
            data: DataFrame containing material properties
            property_name: Property to analyze
            concentration_col: Column name for concentration values
            
        Returns:
            TrendAnalysis object with trend metrics
        """
        if property_name not in data.columns or concentration_col not in data.columns:
            raise ValueError(f"Required columns not found in data")
        
        # Remove rows with missing values
        valid_data = data[[concentration_col, property_name]].dropna()
        
        if len(valid_data) < 3:
            raise ValueError("Insufficient data for trend analysis")
        
        x = valid_data[concentration_col].values
        y = valid_data[property_name].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Determine trend direction
        is_significant = p_value < self.significance_level
        if is_significant:
            if slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'no_trend'
        
        return TrendAnalysis(
            property_name=property_name,
            concentrations=x.tolist(),
            values=y.tolist(),
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            p_value=float(p_value),
            is_significant=is_significant,
            trend_direction=trend_direction,
        )
    
    def analyze_all_trends(
        self, 
        data: pd.DataFrame,
        concentration_col: str = 'dopant_concentration'
    ) -> pd.DataFrame:
        """
        Analyze trends for all numeric properties.
        
        Args:
            data: DataFrame containing material properties
            concentration_col: Column name for concentration values
            
        Returns:
            DataFrame with trend analysis for all properties
        """
        if concentration_col not in data.columns:
            raise ValueError(f"Concentration column '{concentration_col}' not found")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Exclude the concentration column itself
        property_cols = [c for c in numeric_cols if c != concentration_col]
        
        trends_list = []
        for col in property_cols:
            try:
                trend = self.analyze_trend(data, col, concentration_col)
                trends_list.append(trend.to_dict())
            except ValueError:
                continue
        
        return pd.DataFrame(trends_list)
    
    def detect_outliers(
        self, 
        data: pd.DataFrame, 
        property_name: str
    ) -> OutlierDetection:
        """
        Detect outliers using z-score method.
        
        Args:
            data: DataFrame containing material properties
            property_name: Property to analyze
            
        Returns:
            OutlierDetection object with outlier information
        """
        if property_name not in data.columns:
            raise ValueError(f"Property '{property_name}' not found in data")
        
        values = data[property_name].dropna()
        
        if len(values) < 3:
            raise ValueError("Insufficient data for outlier detection")
        
        # Compute z-scores
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            # No variation, no outliers
            return OutlierDetection(
                property_name=property_name,
                outlier_indices=[],
                outlier_values=[],
                z_scores=[],
                threshold=self.outlier_threshold,
            )
        
        z_scores = np.abs((values - mean) / std)
        
        # Identify outliers
        outlier_mask = z_scores > self.outlier_threshold
        outlier_indices = values.index[outlier_mask].tolist()
        outlier_values = values[outlier_mask].tolist()
        outlier_z_scores = z_scores[outlier_mask].tolist()
        
        return OutlierDetection(
            property_name=property_name,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            z_scores=outlier_z_scores,
            threshold=self.outlier_threshold,
        )
    
    def detect_all_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers for all numeric properties.
        
        Args:
            data: DataFrame containing material properties
            
        Returns:
            DataFrame with outlier detection results
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        outliers_list = []
        for col in numeric_cols:
            try:
                outlier_result = self.detect_outliers(data, col)
                outliers_list.append({
                    'property': outlier_result.property_name,
                    'n_outliers': len(outlier_result.outlier_indices),
                    'outlier_indices': outlier_result.outlier_indices,
                    'threshold': outlier_result.threshold,
                })
            except ValueError:
                continue
        
        return pd.DataFrame(outliers_list)
    
    def compare_distributions(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame,
        property_name: str
    ) -> Dict[str, Any]:
        """
        Compare distributions of a property between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            property_name: Property to compare
            
        Returns:
            Dictionary with comparison statistics
        """
        if property_name not in data1.columns or property_name not in data2.columns:
            raise ValueError(f"Property '{property_name}' not found in both datasets")
        
        values1 = data1[property_name].dropna()
        values2 = data2[property_name].dropna()
        
        if len(values1) < 2 or len(values2) < 2:
            raise ValueError("Insufficient data for distribution comparison")
        
        # T-test for mean difference
        t_stat, t_pvalue = stats.ttest_ind(values1, values2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_pvalue = stats.ks_2samp(values1, values2)
        
        return {
            'property': property_name,
            'dataset1_mean': float(values1.mean()),
            'dataset2_mean': float(values2.mean()),
            'dataset1_std': float(values1.std()),
            'dataset2_std': float(values2.std()),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'u_statistic': float(u_stat),
            'u_pvalue': float(u_pvalue),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'means_significantly_different': t_pvalue < self.significance_level,
            'distributions_significantly_different': ks_pvalue < self.significance_level,
        }
