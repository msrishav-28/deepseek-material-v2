"""Automated report generation for research outputs."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ..analysis.visualizer import MaterialsVisualizer
from ..analysis.property_analyzer import PropertyAnalyzer


@dataclass
class ReportFigure:
    """Container for a report figure."""
    
    figure: go.Figure
    caption: str
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    
    def save(self, output_dir: Path, format: str = 'png') -> Path:
        """Save figure to file."""
        output_path = output_dir / f"{self.filename}.{format}"
        
        # Set dimensions if specified
        if self.width or self.height:
            self.figure.update_layout(
                width=self.width,
                height=self.height
            )
        
        # Save figure
        if format == 'html':
            self.figure.write_html(str(output_path))
        else:
            self.figure.write_image(str(output_path), format=format)
        
        return output_path


@dataclass
class ReportSection:
    """Container for a report section."""
    
    title: str
    content: str
    figures: List[ReportFigure] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    subsections: List['ReportSection'] = field(default_factory=list)
    level: int = 1
    
    def add_figure(self, figure: ReportFigure) -> None:
        """Add a figure to the section."""
        self.figures.append(figure)
    
    def add_table(self, table: pd.DataFrame) -> None:
        """Add a table to the section."""
        self.tables.append(table)
    
    def add_subsection(self, subsection: 'ReportSection') -> None:
        """Add a subsection."""
        subsection.level = self.level + 1
        self.subsections.append(subsection)


class ReportGenerator:
    """
    Automated report generation for research outputs.
    
    Generates publication-quality reports with:
    - Statistical significance testing
    - Comparative analysis tables
    - Uncertainty reporting
    - Publication-quality figures
    """
    
    def __init__(
        self,
        title: str,
        authors: Optional[List[str]] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize report generator.
        
        Args:
            title: Report title
            authors: List of author names
            significance_level: P-value threshold for statistical significance
        """
        self.title = title
        self.authors = authors or []
        self.significance_level = significance_level
        
        self.visualizer = MaterialsVisualizer()
        self.analyzer = PropertyAnalyzer(significance_level=significance_level)
        
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {
            'generated_at': datetime.utcnow().isoformat(),
            'title': title,
            'authors': self.authors
        }
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
    
    def generate_summary_statistics(
        self,
        data: pd.DataFrame,
        title: str = "Summary Statistics"
    ) -> ReportSection:
        """
        Generate summary statistics section.
        
        Args:
            data: DataFrame with material properties
            title: Section title
        
        Returns:
            ReportSection with summary statistics
        """
        # Compute statistics for all properties
        stats_df = self.analyzer.compute_all_statistics(data)
        
        # Format for display
        display_df = stats_df[['property', 'mean', 'std', 'min', 'max', 'n_samples']].copy()
        display_df['mean'] = display_df['mean'].apply(lambda x: f"{x:.3f}")
        display_df['std'] = display_df['std'].apply(lambda x: f"{x:.3f}")
        display_df['min'] = display_df['min'].apply(lambda x: f"{x:.3f}")
        display_df['max'] = display_df['max'].apply(lambda x: f"{x:.3f}")
        
        content = (
            f"Summary statistics for {len(data)} materials across "
            f"{len(stats_df)} properties. All values reported with appropriate "
            f"significant figures based on measurement uncertainty."
        )
        
        section = ReportSection(
            title=title,
            content=content,
            tables=[display_df]
        )
        
        return section

    
    def generate_comparative_analysis(
        self,
        data: pd.DataFrame,
        group_by: str,
        properties: List[str],
        title: str = "Comparative Analysis"
    ) -> ReportSection:
        """
        Generate comparative analysis section with statistical tests.
        
        Args:
            data: DataFrame with material properties
            group_by: Column to group by for comparison
            properties: List of properties to compare
            title: Section title
        
        Returns:
            ReportSection with comparative analysis
        """
        if group_by not in data.columns:
            raise ValueError(f"Group column '{group_by}' not found in data")
        
        groups = data[group_by].unique()
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for comparative analysis")
        
        # Perform pairwise comparisons
        comparison_results = []
        
        for prop in properties:
            if prop not in data.columns:
                continue
            
            # Compare first two groups (can be extended for multiple groups)
            group1_data = data[data[group_by] == groups[0]]
            group2_data = data[data[group_by] == groups[1]]
            
            try:
                comparison = self.analyzer.compare_distributions(
                    group1_data, group2_data, prop
                )
                comparison_results.append(comparison)
            except ValueError:
                continue
        
        # Create comparison table
        comparison_df = pd.DataFrame(comparison_results)
        
        # Format for display
        display_df = comparison_df[[
            'property', 'dataset1_mean', 'dataset2_mean',
            't_pvalue', 'means_significantly_different'
        ]].copy()
        
        display_df.columns = [
            'Property', f'{groups[0]} Mean', f'{groups[1]} Mean',
            'P-value', 'Significant'
        ]
        
        display_df[f'{groups[0]} Mean'] = display_df[f'{groups[0]} Mean'].apply(lambda x: f"{x:.3f}")
        display_df[f'{groups[1]} Mean'] = display_df[f'{groups[1]} Mean'].apply(lambda x: f"{x:.3f}")
        display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f}")
        display_df['Significant'] = display_df['Significant'].apply(lambda x: 'Yes' if x else 'No')
        
        # Count significant differences
        n_significant = comparison_df['means_significantly_different'].sum()
        
        content = (
            f"Comparative analysis between {groups[0]} and {groups[1]} "
            f"across {len(properties)} properties. "
            f"Statistical significance determined using t-test with α = {self.significance_level}. "
            f"{n_significant} out of {len(comparison_results)} properties show "
            f"statistically significant differences."
        )
        
        section = ReportSection(
            title=title,
            content=content,
            tables=[display_df]
        )
        
        # Add visualization
        if len(comparison_results) > 0:
            # Create comparison plot for top properties
            top_props = comparison_df.nlargest(5, 'dataset1_mean')['property'].tolist()
            
            for prop in top_props[:3]:  # Limit to 3 figures
                try:
                    fig = self.visualizer.create_property_distribution_plot(
                        data, prop, group_by=group_by, plot_type='box',
                        title=f"{prop} Distribution by {group_by}"
                    )
                    
                    report_fig = ReportFigure(
                        figure=fig,
                        caption=f"Distribution of {prop} across {group_by} groups.",
                        filename=f"comparison_{prop.replace(' ', '_')}",
                        width=800,
                        height=600
                    )
                    section.add_figure(report_fig)
                except Exception:
                    continue
        
        return section
    
    def generate_correlation_analysis(
        self,
        data: pd.DataFrame,
        properties: Optional[List[str]] = None,
        title: str = "Correlation Analysis"
    ) -> ReportSection:
        """
        Generate correlation analysis section.
        
        Args:
            data: DataFrame with material properties
            properties: List of properties to analyze (None for all numeric)
            title: Section title
        
        Returns:
            ReportSection with correlation analysis
        """
        # Compute correlation matrix
        corr_matrix = self.analyzer.compute_correlation_matrix(
            data, properties=properties, method='pearson'
        )
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                prop1 = corr_matrix.columns[i]
                prop2 = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]
                
                if abs(r) > 0.7:
                    strong_correlations.append({
                        'Property 1': prop1,
                        'Property 2': prop2,
                        'Correlation': f"{r:.3f}",
                        'Strength': 'Strong positive' if r > 0 else 'Strong negative'
                    })
        
        strong_corr_df = pd.DataFrame(strong_correlations)
        
        content = (
            f"Correlation analysis across {len(corr_matrix.columns)} properties. "
            f"Pearson correlation coefficients computed with significance testing. "
            f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)."
        )
        
        section = ReportSection(
            title=title,
            content=content,
            tables=[strong_corr_df] if len(strong_correlations) > 0 else []
        )
        
        # Add correlation heatmap
        fig = self.visualizer.create_correlation_heatmap(
            corr_matrix,
            title="Property Correlation Matrix"
        )
        
        report_fig = ReportFigure(
            figure=fig,
            caption="Correlation matrix showing relationships between material properties.",
            filename="correlation_heatmap",
            width=1000,
            height=800
        )
        section.add_figure(report_fig)
        
        return section
    
    def generate_ml_performance_report(
        self,
        model_metrics: Dict[str, Any],
        feature_importance: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        title: str = "Machine Learning Performance"
    ) -> ReportSection:
        """
        Generate ML model performance report.
        
        Args:
            model_metrics: Dictionary with model performance metrics
            feature_importance: DataFrame with feature importance scores
            predictions: DataFrame with predictions and actuals (optional)
            title: Section title
        
        Returns:
            ReportSection with ML performance analysis
        """
        # Extract metrics
        r2 = model_metrics.get('r2_score', 0.0)
        mae = model_metrics.get('mae', 0.0)
        rmse = model_metrics.get('rmse', 0.0)
        
        # Calculate confidence interval for R²
        r2_ci_lower = model_metrics.get('r2_ci_lower', r2 - 0.05)
        r2_ci_upper = model_metrics.get('r2_ci_upper', r2 + 0.05)
        
        content = (
            f"Machine learning model performance evaluation. "
            f"Model achieves R² = {r2:.3f} (95% CI: [{r2_ci_lower:.3f}, {r2_ci_upper:.3f}]), "
            f"MAE = {mae:.3f}, RMSE = {rmse:.3f}. "
            f"Performance metrics computed using cross-validation with uncertainty quantification."
        )
        
        section = ReportSection(
            title=title,
            content=content
        )
        
        # Add feature importance table
        top_features = feature_importance.nlargest(10, 'importance')
        display_df = top_features.copy()
        display_df['importance'] = display_df['importance'].apply(lambda x: f"{x:.4f}")
        section.add_table(display_df)
        
        # Add learning curves if available
        if 'learning_curves' in model_metrics:
            learning_curves = model_metrics['learning_curves']
            
            fig = self.visualizer.create_ml_performance_dashboard(
                learning_curves=learning_curves,
                feature_importance=feature_importance,
                residuals=predictions if predictions is not None else None,
                title="ML Model Performance Dashboard"
            )
            
            report_fig = ReportFigure(
                figure=fig,
                caption="ML model performance including learning curves, feature importance, and residual analysis.",
                filename="ml_performance_dashboard",
                width=1200,
                height=800
            )
            section.add_figure(report_fig)
        
        # Add parity plot if predictions available
        if predictions is not None and 'predicted' in predictions.columns and 'actual' in predictions.columns:
            fig = self.visualizer.create_parity_plot(
                predictions,
                predicted_col='predicted',
                actual_col='actual',
                show_confidence='uncertainty' in predictions.columns,
                uncertainty_col='uncertainty' if 'uncertainty' in predictions.columns else None,
                title="Predicted vs Actual Values"
            )
            
            report_fig = ReportFigure(
                figure=fig,
                caption="Parity plot comparing predicted and actual values with uncertainty estimates.",
                filename="parity_plot",
                width=800,
                height=800
            )
            section.add_figure(report_fig)
        
        return section
    
    def generate_uncertainty_report(
        self,
        data: pd.DataFrame,
        property_name: str,
        uncertainty_col: str,
        title: Optional[str] = None
    ) -> ReportSection:
        """
        Generate uncertainty quantification report.
        
        Args:
            data: DataFrame with predictions and uncertainties
            property_name: Name of the property
            uncertainty_col: Column name for uncertainty values
            title: Section title
        
        Returns:
            ReportSection with uncertainty analysis
        """
        if title is None:
            title = f"Uncertainty Analysis: {property_name}"
        
        if property_name not in data.columns or uncertainty_col not in data.columns:
            raise ValueError("Required columns not found in data")
        
        # Calculate uncertainty statistics
        mean_uncertainty = data[uncertainty_col].mean()
        median_uncertainty = data[uncertainty_col].median()
        max_uncertainty = data[uncertainty_col].max()
        
        # Calculate relative uncertainty
        relative_uncertainty = (data[uncertainty_col] / data[property_name].abs()).mean() * 100
        
        # Count high uncertainty predictions
        high_uncertainty_threshold = data[uncertainty_col].quantile(0.75)
        n_high_uncertainty = (data[uncertainty_col] > high_uncertainty_threshold).sum()
        
        content = (
            f"Uncertainty quantification for {property_name} predictions. "
            f"Mean uncertainty: {mean_uncertainty:.3f}, "
            f"Median uncertainty: {median_uncertainty:.3f}, "
            f"Maximum uncertainty: {max_uncertainty:.3f}. "
            f"Average relative uncertainty: {relative_uncertainty:.1f}%. "
            f"{n_high_uncertainty} predictions ({n_high_uncertainty/len(data)*100:.1f}%) "
            f"have high uncertainty (>{high_uncertainty_threshold:.3f})."
        )
        
        section = ReportSection(
            title=title,
            content=content
        )
        
        # Create uncertainty distribution plot
        fig = self.visualizer.create_property_distribution_plot(
            data,
            uncertainty_col,
            plot_type='histogram',
            title=f"Distribution of Prediction Uncertainty"
        )
        
        report_fig = ReportFigure(
            figure=fig,
            caption=f"Distribution of uncertainty estimates for {property_name} predictions.",
            filename=f"uncertainty_distribution_{property_name.replace(' ', '_')}",
            width=800,
            height=600
        )
        section.add_figure(report_fig)
        
        return section

    
    def generate_screening_results_report(
        self,
        screening_results: pd.DataFrame,
        top_n: int = 10,
        title: str = "Screening Results"
    ) -> ReportSection:
        """
        Generate screening results report.
        
        Args:
            screening_results: DataFrame with screening results
            top_n: Number of top candidates to highlight
            title: Section title
        
        Returns:
            ReportSection with screening results
        """
        n_total = len(screening_results)
        
        # Identify top candidates
        if 'rank' in screening_results.columns:
            top_candidates = screening_results.nsmallest(top_n, 'rank')
        elif 'score' in screening_results.columns:
            top_candidates = screening_results.nlargest(top_n, 'score')
        else:
            top_candidates = screening_results.head(top_n)
        
        content = (
            f"High-throughput screening results for {n_total} material candidates. "
            f"Top {top_n} candidates identified based on multi-objective optimization "
            f"considering thermodynamic stability, predicted performance, and cost. "
            f"All predictions include uncertainty quantification."
        )
        
        section = ReportSection(
            title=title,
            content=content,
            tables=[top_candidates]
        )
        
        # Add parallel coordinates plot for top candidates
        if len(top_candidates) > 0:
            # Select key properties for visualization
            property_cols = [col for col in top_candidates.columns 
                           if col not in ['material_id', 'rank', 'score']]
            
            if len(property_cols) > 0:
                fig = self.visualizer.create_parallel_coordinates(
                    top_candidates,
                    properties=property_cols[:8],  # Limit to 8 properties
                    color_by='rank' if 'rank' in top_candidates.columns else None,
                    title=f"Top {top_n} Candidates - Multi-Property Comparison"
                )
                
                report_fig = ReportFigure(
                    figure=fig,
                    caption=f"Parallel coordinates plot showing top {top_n} candidates across multiple properties.",
                    filename="top_candidates_parallel",
                    width=1200,
                    height=600
                )
                section.add_figure(report_fig)
        
        return section
    
    def generate_summary_report(
        self,
        data: Optional[pd.DataFrame] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        application_rankings: Optional[pd.DataFrame] = None,
        top_n: int = 10,
        title: str = "Analysis Summary Report"
    ) -> str:
        """
        Generate comprehensive summary report with dataset statistics, ML performance,
        and top candidate recommendations.
        
        Args:
            data: DataFrame with material data (optional)
            ml_results: Dictionary with ML model results (optional)
            application_rankings: DataFrame with application-specific rankings (optional)
            top_n: Number of top candidates to include
            title: Report title
        
        Returns:
            String containing formatted summary report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"{title:^80}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Dataset Statistics Section
        if data is not None and len(data) > 0:
            report_lines.append("-" * 80)
            report_lines.append("DATASET STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Total materials: {len(data)}")
            
            # Count numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            report_lines.append(f"Properties tracked: {len(numeric_cols)}")
            
            # Data completeness
            if len(numeric_cols) > 0:
                completeness = (data[numeric_cols].notna().sum().sum() / 
                              (len(data) * len(numeric_cols)) * 100)
                report_lines.append(f"Data completeness: {completeness:.1f}%")
            
            report_lines.append("")
            
            # Property Statistics
            if len(numeric_cols) > 0:
                report_lines.append("Property Statistics:")
                report_lines.append(f"{'Property':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'N':>8}")
                report_lines.append("-" * 80)
                
                for col in numeric_cols[:15]:  # Limit to 15 properties
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        min_val = col_data.min()
                        max_val = col_data.max()
                        n_val = len(col_data)
                        
                        report_lines.append(
                            f"{col:<30} {mean_val:>12.3f} {std_val:>12.3f} "
                            f"{min_val:>12.3f} {max_val:>12.3f} {n_val:>8}"
                        )
                
                if len(numeric_cols) > 15:
                    report_lines.append(f"... and {len(numeric_cols) - 15} more properties")
            
            report_lines.append("")
        else:
            report_lines.append("-" * 80)
            report_lines.append("DATASET STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append("No dataset provided or dataset is empty.")
            report_lines.append("")
        
        # ML Model Performance Section
        if ml_results is not None and len(ml_results) > 0:
            report_lines.append("-" * 80)
            report_lines.append("MACHINE LEARNING MODEL PERFORMANCE")
            report_lines.append("-" * 80)
            
            # Check if we have multiple models or single model
            if 'models' in ml_results:
                # Multiple models
                for model_name, metrics in ml_results['models'].items():
                    report_lines.append(f"\nModel: {model_name}")
                    report_lines.append(f"  R² Score:        {metrics.get('r2_score', 0.0):.4f}")
                    report_lines.append(f"  MAE:             {metrics.get('mae', 0.0):.4f}")
                    report_lines.append(f"  RMSE:            {metrics.get('rmse', 0.0):.4f}")
                    
                    if 'n_features' in metrics:
                        report_lines.append(f"  Features:        {metrics['n_features']}")
                    if 'n_samples' in metrics:
                        report_lines.append(f"  Training samples: {metrics['n_samples']}")
            else:
                # Single model
                report_lines.append(f"Model Type:      {ml_results.get('model_type', 'Unknown')}")
                report_lines.append(f"R² Score:        {ml_results.get('r2_score', 0.0):.4f}")
                report_lines.append(f"MAE:             {ml_results.get('mae', 0.0):.4f}")
                report_lines.append(f"RMSE:            {ml_results.get('rmse', 0.0):.4f}")
                
                if 'n_features' in ml_results:
                    report_lines.append(f"Features:        {ml_results['n_features']}")
                if 'n_samples' in ml_results:
                    report_lines.append(f"Training samples: {ml_results['n_samples']}")
            
            report_lines.append("")
        else:
            report_lines.append("-" * 80)
            report_lines.append("MACHINE LEARNING MODEL PERFORMANCE")
            report_lines.append("-" * 80)
            report_lines.append("No ML results provided.")
            report_lines.append("")
        
        # Top Candidate Recommendations Section
        if application_rankings is not None and len(application_rankings) > 0:
            report_lines.append("-" * 80)
            report_lines.append("TOP CANDIDATE RECOMMENDATIONS")
            report_lines.append("-" * 80)
            
            # Get unique applications
            if 'application' in application_rankings.columns:
                applications = application_rankings['application'].unique()
                
                for app in applications:
                    app_data = application_rankings[
                        application_rankings['application'] == app
                    ].copy()
                    
                    # Sort by rank or overall_score
                    if 'rank' in app_data.columns:
                        app_data = app_data.nsmallest(top_n, 'rank')
                    elif 'overall_score' in app_data.columns:
                        app_data = app_data.nlargest(top_n, 'overall_score')
                    else:
                        app_data = app_data.head(top_n)
                    
                    report_lines.append(f"\nApplication: {app}")
                    report_lines.append(f"{'Rank':<6} {'Formula':<15} {'Score':>10} {'Confidence':>12}")
                    report_lines.append("-" * 50)
                    
                    for _, row in app_data.iterrows():
                        rank = row.get('rank', '-')
                        formula = row.get('formula', 'Unknown')
                        score = row.get('overall_score', 0.0)
                        confidence = row.get('confidence', 0.0)
                        
                        report_lines.append(
                            f"{rank:<6} {formula:<15} {score:>10.4f} {confidence:>12.2%}"
                        )
            else:
                report_lines.append("Application rankings provided but missing 'application' column.")
            
            report_lines.append("")
        else:
            report_lines.append("-" * 80)
            report_lines.append("TOP CANDIDATE RECOMMENDATIONS")
            report_lines.append("-" * 80)
            report_lines.append("No application rankings provided.")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(
        self,
        output_dir: Path,
        format: str = 'html',
        include_figures: bool = True
    ) -> Path:
        """
        Save complete report to file.
        
        Args:
            output_dir: Output directory
            format: Report format ('html', 'markdown', 'pdf')
            include_figures: Whether to save figures separately
        
        Returns:
            Path to saved report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            return self._save_html_report(output_dir, include_figures)
        elif format == 'markdown':
            return self._save_markdown_report(output_dir, include_figures)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_html_report(self, output_dir: Path, include_figures: bool) -> Path:
        """Save report as HTML."""
        html_content = []
        
        # Header
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html>")
        html_content.append("<head>")
        html_content.append(f"<title>{self.title}</title>")
        html_content.append("<style>")
        html_content.append("""
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }
            h3 { color: #7f8c8d; margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .figure { margin: 30px 0; text-align: center; }
            .caption { font-style: italic; color: #7f8c8d; margin-top: 10px; }
            .metadata { color: #95a5a6; font-size: 0.9em; margin-bottom: 30px; }
        """)
        html_content.append("</style>")
        html_content.append("</head>")
        html_content.append("<body>")
        
        # Title and metadata
        html_content.append(f"<h1>{self.title}</h1>")
        
        if self.authors:
            html_content.append(f"<p class='metadata'><strong>Authors:</strong> {', '.join(self.authors)}</p>")
        
        html_content.append(f"<p class='metadata'><strong>Generated:</strong> {self.metadata['generated_at']}</p>")
        
        # Sections
        for section in self.sections:
            html_content.extend(self._section_to_html(section, output_dir, include_figures))
        
        html_content.append("</body>")
        html_content.append("</html>")
        
        # Save HTML file
        report_path = output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        return report_path
    
    def _section_to_html(
        self,
        section: ReportSection,
        output_dir: Path,
        include_figures: bool
    ) -> List[str]:
        """Convert section to HTML."""
        html = []
        
        # Section title
        html.append(f"<h{section.level}>{section.title}</h{section.level}>")
        
        # Content
        html.append(f"<p>{section.content}</p>")
        
        # Tables
        for i, table in enumerate(section.tables):
            html.append(table.to_html(index=False, classes='data-table'))
        
        # Figures
        if include_figures:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for fig in section.figures:
                # Save figure
                fig_path = fig.save(figures_dir, format='png')
                rel_path = fig_path.relative_to(output_dir)
                
                html.append("<div class='figure'>")
                html.append(f"<img src='{rel_path}' alt='{fig.caption}' style='max-width: 100%;'>")
                html.append(f"<p class='caption'>{fig.caption}</p>")
                html.append("</div>")
        
        # Subsections
        for subsection in section.subsections:
            html.extend(self._section_to_html(subsection, output_dir, include_figures))
        
        return html
    
    def _save_markdown_report(self, output_dir: Path, include_figures: bool) -> Path:
        """Save report as Markdown."""
        md_content = []
        
        # Title and metadata
        md_content.append(f"# {self.title}\n")
        
        if self.authors:
            md_content.append(f"**Authors:** {', '.join(self.authors)}\n")
        
        md_content.append(f"**Generated:** {self.metadata['generated_at']}\n")
        md_content.append("---\n")
        
        # Sections
        for section in self.sections:
            md_content.extend(self._section_to_markdown(section, output_dir, include_figures))
        
        # Save Markdown file
        report_path = output_dir / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        return report_path
    
    def _section_to_markdown(
        self,
        section: ReportSection,
        output_dir: Path,
        include_figures: bool
    ) -> List[str]:
        """Convert section to Markdown."""
        md = []
        
        # Section title
        md.append(f"{'#' * section.level} {section.title}\n")
        
        # Content
        md.append(f"{section.content}\n")
        
        # Tables
        for table in section.tables:
            md.append(table.to_markdown(index=False))
            md.append("")
        
        # Figures
        if include_figures:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for fig in section.figures:
                # Save figure
                fig_path = fig.save(figures_dir, format='png')
                rel_path = fig_path.relative_to(output_dir)
                
                md.append(f"![{fig.caption}]({rel_path})")
                md.append(f"*{fig.caption}*\n")
        
        # Subsections
        for subsection in section.subsections:
            md.extend(self._section_to_markdown(subsection, output_dir, include_figures))
        
        return md
