"""Research visualization system for materials discovery."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class MaterialsVisualizer:
    """
    Research visualization system for materials property analysis.
    
    Provides parallel coordinates plots, structure-property visualizations,
    ML performance dashboards, and interactive filtering capabilities.
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer.
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', etc.)
        """
        self.theme = theme
    
    def create_parallel_coordinates(
        self,
        data: pd.DataFrame,
        properties: List[str],
        color_by: Optional[str] = None,
        title: str = "Multi-Property Comparison"
    ) -> go.Figure:
        """
        Create parallel coordinates plot for multi-property comparison.
        
        Args:
            data: DataFrame containing material properties
            properties: List of property names to include
            color_by: Property to use for coloring (optional)
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        # Validate properties
        missing_props = [p for p in properties if p not in data.columns]
        if missing_props:
            raise ValueError(f"Properties not found in data: {missing_props}")
        
        # Prepare dimensions for parallel coordinates
        dimensions = []
        for prop in properties:
            dimensions.append(
                dict(
                    label=prop,
                    values=data[prop],
                    range=[data[prop].min(), data[prop].max()]
                )
            )
        
        # Determine color scale
        if color_by and color_by in data.columns:
            color = data[color_by]
            colorscale = 'Viridis'
        else:
            color = np.arange(len(data))
            colorscale = 'Blues'
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color,
                    colorscale=colorscale,
                    showscale=True,
                    cmin=color.min() if hasattr(color, 'min') else min(color),
                    cmax=color.max() if hasattr(color, 'max') else max(color),
                ),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
        )
        
        return fig
    
    def create_structure_property_plot(
        self,
        data: pd.DataFrame,
        x_property: str,
        y_property: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        show_confidence: bool = False,
        x_uncertainty: Optional[str] = None,
        y_uncertainty: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create structure-property relationship scatter plot.
        
        Args:
            data: DataFrame containing material properties
            x_property: Property for x-axis
            y_property: Property for y-axis
            color_by: Property to use for coloring (optional)
            size_by: Property to use for marker size (optional)
            show_confidence: Whether to show confidence intervals
            x_uncertainty: Column name for x uncertainty (optional)
            y_uncertainty: Column name for y uncertainty (optional)
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if x_property not in data.columns or y_property not in data.columns:
            raise ValueError(f"Properties not found in data")
        
        # Prepare hover data
        hover_data = [x_property, y_property]
        if color_by and color_by in data.columns:
            hover_data.append(color_by)
        if size_by and size_by in data.columns:
            hover_data.append(size_by)
        
        # Create scatter plot
        fig = px.scatter(
            data,
            x=x_property,
            y=y_property,
            color=color_by if color_by and color_by in data.columns else None,
            size=size_by if size_by and size_by in data.columns else None,
            hover_data=hover_data,
            template=self.theme,
        )
        
        # Add error bars if uncertainty data provided
        if show_confidence and x_uncertainty and x_uncertainty in data.columns:
            fig.update_traces(
                error_x=dict(
                    type='data',
                    array=data[x_uncertainty],
                    visible=True
                )
            )
        
        if show_confidence and y_uncertainty and y_uncertainty in data.columns:
            fig.update_traces(
                error_y=dict(
                    type='data',
                    array=data[y_uncertainty],
                    visible=True
                )
            )
        
        # Update layout
        if title is None:
            title = f"{y_property} vs {x_property}"
        
        fig.update_layout(
            title=title,
            xaxis_title=x_property,
            yaxis_title=y_property,
            height=600,
        )
        
        return fig
    
    def create_ml_performance_dashboard(
        self,
        learning_curves: Dict[str, List[float]],
        feature_importance: pd.DataFrame,
        residuals: Optional[pd.DataFrame] = None,
        title: str = "ML Model Performance Dashboard"
    ) -> go.Figure:
        """
        Create ML performance dashboard with learning curves and feature importance.
        
        Args:
            learning_curves: Dict with 'train_scores' and 'val_scores' lists
            feature_importance: DataFrame with 'feature' and 'importance' columns
            residuals: DataFrame with 'predicted' and 'residual' columns (optional)
            title: Dashboard title
            
        Returns:
            Plotly Figure object with subplots
        """
        # Determine number of subplots
        n_plots = 2 if residuals is None else 3
        
        # Create subplots
        if n_plots == 2:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Learning Curves", "Feature Importance"),
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Learning Curves", "Feature Importance", "Residual Plot"),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter", "colspan": 2}, None]]
            )
        
        # Learning curves
        epochs = list(range(1, len(learning_curves['train_scores']) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=learning_curves['train_scores'],
                mode='lines+markers',
                name='Training',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if 'val_scores' in learning_curves:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=learning_curves['val_scores'],
                    mode='lines+markers',
                    name='Validation',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # Feature importance
        top_features = feature_importance.nlargest(10, 'importance')
        fig.add_trace(
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Residual plot (if provided)
        if residuals is not None and n_plots == 3:
            fig.add_trace(
                go.Scatter(
                    x=residuals['predicted'],
                    y=residuals['residual'],
                    mode='markers',
                    marker=dict(color='purple', opacity=0.6),
                    name='Residuals'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Score (R²)", row=1, col=1)
        fig.update_xaxes(title_text="Importance", row=1, col=2)
        fig.update_yaxes(title_text="Feature", row=1, col=2)
        
        if residuals is not None and n_plots == 3:
            fig.update_xaxes(title_text="Predicted Value", row=2, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800 if n_plots == 3 else 500,
            showlegend=True,
        )
        
        return fig
    
    def create_parity_plot(
        self,
        data: pd.DataFrame,
        predicted_col: str,
        actual_col: str,
        show_confidence: bool = False,
        uncertainty_col: Optional[str] = None,
        title: str = "Predicted vs Actual"
    ) -> go.Figure:
        """
        Create parity plot comparing predicted vs actual values.
        
        Args:
            data: DataFrame with predictions and actual values
            predicted_col: Column name for predicted values
            actual_col: Column name for actual values
            show_confidence: Whether to show confidence intervals
            uncertainty_col: Column name for prediction uncertainty (optional)
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if predicted_col not in data.columns or actual_col not in data.columns:
            raise ValueError(f"Required columns not found in data")
        
        fig = go.Figure()
        
        # Add scatter plot
        scatter_kwargs = dict(
            x=data[actual_col],
            y=data[predicted_col],
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Predictions'
        )
        
        # Add error bars if uncertainty provided
        if show_confidence and uncertainty_col and uncertainty_col in data.columns:
            scatter_kwargs['error_y'] = dict(
                type='data',
                array=data[uncertainty_col],
                visible=True
            )
        
        fig.add_trace(go.Scatter(**scatter_kwargs))
        
        # Add perfect prediction line (y=x)
        min_val = min(data[actual_col].min(), data[predicted_col].min())
        max_val = max(data[actual_col].max(), data[predicted_col].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        # Calculate and display metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2 = r2_score(data[actual_col], data[predicted_col])
        mae = mean_absolute_error(data[actual_col], data[predicted_col])
        rmse = np.sqrt(mean_squared_error(data[actual_col], data[predicted_col]))
        
        metrics_text = f"R² = {r2:.3f}<br>MAE = {mae:.3f}<br>RMSE = {rmse:.3f}"
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=f"Actual {actual_col}",
            yaxis_title=f"Predicted {predicted_col}",
            template=self.theme,
            height=600,
        )
        
        return fig
    
    def create_concentration_trend_plot(
        self,
        data: pd.DataFrame,
        property_name: str,
        concentration_col: str = 'dopant_concentration',
        group_by: Optional[str] = None,
        show_confidence: bool = False,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create concentration-dependent trend plot.
        
        Args:
            data: DataFrame with material properties
            property_name: Property to plot
            concentration_col: Column name for concentration values
            group_by: Column to group by (e.g., 'base_composition')
            show_confidence: Whether to show confidence intervals
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if property_name not in data.columns or concentration_col not in data.columns:
            raise ValueError(f"Required columns not found in data")
        
        fig = go.Figure()
        
        if group_by and group_by in data.columns:
            # Plot separate lines for each group
            for group_name in data[group_by].unique():
                group_data = data[data[group_by] == group_name]
                
                # Sort by concentration
                group_data = group_data.sort_values(concentration_col)
                
                # Calculate mean and std for each concentration
                grouped = group_data.groupby(concentration_col)[property_name].agg(['mean', 'std', 'count'])
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped.index,
                        y=grouped['mean'],
                        mode='lines+markers',
                        name=str(group_name),
                    )
                )
                
                # Add confidence intervals if requested
                if show_confidence:
                    # Standard error
                    se = grouped['std'] / np.sqrt(grouped['count'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=grouped.index.tolist() + grouped.index.tolist()[::-1],
                            y=(grouped['mean'] + se).tolist() + (grouped['mean'] - se).tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,200,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f'{group_name} CI'
                        )
                    )
        else:
            # Single trend line
            sorted_data = data.sort_values(concentration_col)
            
            fig.add_trace(
                go.Scatter(
                    x=sorted_data[concentration_col],
                    y=sorted_data[property_name],
                    mode='markers',
                    marker=dict(color='blue', opacity=0.6),
                    name=property_name
                )
            )
            
            # Add trend line
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                sorted_data[concentration_col], 
                sorted_data[property_name]
            )
            
            x_trend = np.array([sorted_data[concentration_col].min(), sorted_data[concentration_col].max()])
            y_trend = slope * x_trend + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name=f'Trend (R²={r_value**2:.3f})'
                )
            )
        
        if title is None:
            title = f"{property_name} vs {concentration_col}"
        
        fig.update_layout(
            title=title,
            xaxis_title=concentration_col,
            yaxis_title=property_name,
            template=self.theme,
            height=600,
        )
        
        return fig
    
    def create_property_distribution_plot(
        self,
        data: pd.DataFrame,
        property_name: str,
        group_by: Optional[str] = None,
        plot_type: str = 'histogram',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create property distribution plot.
        
        Args:
            data: DataFrame with material properties
            property_name: Property to plot
            group_by: Column to group by (optional)
            plot_type: Type of plot ('histogram', 'box', 'violin')
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if property_name not in data.columns:
            raise ValueError(f"Property '{property_name}' not found in data")
        
        if plot_type == 'histogram':
            if group_by and group_by in data.columns:
                fig = px.histogram(
                    data,
                    x=property_name,
                    color=group_by,
                    barmode='overlay',
                    opacity=0.7,
                    template=self.theme,
                )
            else:
                fig = px.histogram(
                    data,
                    x=property_name,
                    template=self.theme,
                )
        
        elif plot_type == 'box':
            if group_by and group_by in data.columns:
                fig = px.box(
                    data,
                    x=group_by,
                    y=property_name,
                    template=self.theme,
                )
            else:
                fig = px.box(
                    data,
                    y=property_name,
                    template=self.theme,
                )
        
        elif plot_type == 'violin':
            if group_by and group_by in data.columns:
                fig = px.violin(
                    data,
                    x=group_by,
                    y=property_name,
                    box=True,
                    template=self.theme,
                )
            else:
                fig = px.violin(
                    data,
                    y=property_name,
                    box=True,
                    template=self.theme,
                )
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        if title is None:
            title = f"Distribution of {property_name}"
        
        fig.update_layout(
            title=title,
            height=600,
        )
        
        return fig
    
    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Property Correlation Matrix"
    ) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            xaxis={'side': 'bottom'},
        )
        
        return fig
