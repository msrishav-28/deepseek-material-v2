"""Interactive widgets for Jupyter notebooks."""

from typing import Callable, Dict, List, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


class MaterialExplorer:
    """Interactive widget for exploring material properties."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize material explorer.
        
        Args:
            data: DataFrame with material properties
        """
        self.data = data
        self.filtered_data = data.copy()
        
        # Create widgets
        self._create_widgets()
        
    def _create_widgets(self):
        """Create interactive widgets."""
        # Property selectors
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.x_axis = widgets.Dropdown(
            options=numeric_cols,
            value=numeric_cols[0] if numeric_cols else None,
            description='X-axis:',
            style={'description_width': '100px'}
        )
        
        self.y_axis = widgets.Dropdown(
            options=numeric_cols,
            value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
            description='Y-axis:',
            style={'description_width': '100px'}
        )
        
        self.color_by = widgets.Dropdown(
            options=['None'] + numeric_cols,
            value='None',
            description='Color by:',
            style={'description_width': '100px'}
        )
        
        # Filters
        self.stability_filter = widgets.FloatSlider(
            value=0.1,
            min=0,
            max=0.5,
            step=0.01,
            description='Max ΔE_hull:',
            style={'description_width': '100px'},
            continuous_update=False
        )
        
        # Plot button
        self.plot_button = widgets.Button(
            description='Update Plot',
            button_style='primary',
            icon='refresh'
        )
        self.plot_button.on_click(self._update_plot)
        
        # Output area
        self.output = widgets.Output()
        
    def display(self):
        """Display the widget interface."""
        # Layout
        controls = widgets.VBox([
            widgets.HBox([self.x_axis, self.y_axis]),
            widgets.HBox([self.color_by, self.stability_filter]),
            self.plot_button
        ])
        
        display(widgets.VBox([controls, self.output]))
        
        # Initial plot
        self._update_plot(None)
    
    def _update_plot(self, _):
        """Update the plot based on current selections."""
        with self.output:
            self.output.clear_output(wait=True)
            
            # Filter data
            self.filtered_data = self.data[
                self.data['energy_above_hull'] <= self.stability_filter.value
            ]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_data = self.filtered_data[self.x_axis.value]
            y_data = self.filtered_data[self.y_axis.value]
            
            if self.color_by.value != 'None':
                scatter = ax.scatter(
                    x_data, y_data,
                    c=self.filtered_data[self.color_by.value],
                    cmap='viridis',
                    alpha=0.6,
                    s=100
                )
                plt.colorbar(scatter, ax=ax, label=self.color_by.value)
            else:
                ax.scatter(x_data, y_data, alpha=0.6, s=100)
            
            ax.set_xlabel(self.x_axis.value)
            ax.set_ylabel(self.y_axis.value)
            ax.set_title(f'{self.y_axis.value} vs {self.x_axis.value}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Display statistics
            print(f"\nFiltered materials: {len(self.filtered_data)} / {len(self.data)}")
            print(f"X-axis range: [{x_data.min():.2f}, {x_data.max():.2f}]")
            print(f"Y-axis range: [{y_data.min():.2f}, {y_data.max():.2f}]")


class ScreeningConfigurator:
    """Interactive widget for configuring screening parameters."""
    
    def __init__(self, callback: Optional[Callable] = None):
        """Initialize screening configurator.
        
        Args:
            callback: Function to call when configuration is submitted
        """
        self.callback = callback
        self._create_widgets()
    
    def _create_widgets(self):
        """Create configuration widgets."""
        # Base system selector
        self.base_systems = widgets.SelectMultiple(
            options=['SiC', 'B4C', 'WC', 'TiC', 'Al2O3'],
            value=['SiC'],
            description='Base Systems:',
            style={'description_width': '120px'}
        )
        
        # Dopant input
        self.dopants = widgets.Textarea(
            value='Ti, Zr, Hf',
            description='Dopants:',
            placeholder='Enter comma-separated elements',
            style={'description_width': '120px'}
        )
        
        # Concentration selector
        self.concentrations = widgets.SelectMultiple(
            options=[0.01, 0.02, 0.03, 0.05, 0.10],
            value=[0.01, 0.05],
            description='Concentrations:',
            style={'description_width': '120px'}
        )
        
        # Stability threshold
        self.stability_threshold = widgets.FloatText(
            value=0.1,
            description='ΔE_hull threshold:',
            style={'description_width': '120px'}
        )
        
        # Parallel jobs
        self.parallel_jobs = widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            description='Parallel jobs:',
            style={'description_width': '120px'}
        )
        
        # Submit button
        self.submit_button = widgets.Button(
            description='Start Screening',
            button_style='success',
            icon='play'
        )
        self.submit_button.on_click(self._on_submit)
        
        # Output
        self.output = widgets.Output()
    
    def display(self):
        """Display the configuration interface."""
        layout = widgets.VBox([
            widgets.HTML('<h3>Screening Configuration</h3>'),
            self.base_systems,
            self.dopants,
            self.concentrations,
            self.stability_threshold,
            self.parallel_jobs,
            self.submit_button,
            self.output
        ])
        
        display(layout)
    
    def _on_submit(self, _):
        """Handle submit button click."""
        config = self.get_config()
        
        with self.output:
            self.output.clear_output()
            print("Configuration submitted:")
            print(f"  Base systems: {config['base_systems']}")
            print(f"  Dopants: {config['dopants']}")
            print(f"  Concentrations: {config['concentrations']}")
            print(f"  Stability threshold: {config['stability_threshold']} eV/atom")
            print(f"  Parallel jobs: {config['parallel_jobs']}")
            
            if self.callback:
                self.callback(config)
    
    def get_config(self) -> Dict:
        """Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'base_systems': list(self.base_systems.value),
            'dopants': [d.strip() for d in self.dopants.value.split(',')],
            'concentrations': list(self.concentrations.value),
            'stability_threshold': self.stability_threshold.value,
            'parallel_jobs': self.parallel_jobs.value
        }


class PropertyComparator:
    """Interactive widget for comparing material properties."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize property comparator.
        
        Args:
            data: DataFrame with material properties
        """
        self.data = data
        self._create_widgets()
    
    def _create_widgets(self):
        """Create comparison widgets."""
        # Material selectors
        materials = self.data['composition'].unique().tolist()
        
        self.material1 = widgets.Dropdown(
            options=materials,
            value=materials[0] if materials else None,
            description='Material 1:',
            style={'description_width': '100px'}
        )
        
        self.material2 = widgets.Dropdown(
            options=materials,
            value=materials[1] if len(materials) > 1 else materials[0],
            description='Material 2:',
            style={'description_width': '100px'}
        )
        
        # Property selector
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.properties = widgets.SelectMultiple(
            options=numeric_cols,
            value=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
            description='Properties:',
            style={'description_width': '100px'}
        )
        
        # Compare button
        self.compare_button = widgets.Button(
            description='Compare',
            button_style='info',
            icon='balance-scale'
        )
        self.compare_button.on_click(self._compare)
        
        # Output
        self.output = widgets.Output()
    
    def display(self):
        """Display the comparison interface."""
        layout = widgets.VBox([
            widgets.HTML('<h3>Material Property Comparison</h3>'),
            widgets.HBox([self.material1, self.material2]),
            self.properties,
            self.compare_button,
            self.output
        ])
        
        display(layout)
    
    def _compare(self, _):
        """Compare selected materials."""
        with self.output:
            self.output.clear_output(wait=True)
            
            # Get material data
            mat1_data = self.data[self.data['composition'] == self.material1.value].iloc[0]
            mat2_data = self.data[self.data['composition'] == self.material2.value].iloc[0]
            
            # Create comparison plot
            properties = list(self.properties.value)
            values1 = [mat1_data[prop] for prop in properties]
            values2 = [mat2_data[prop] for prop in properties]
            
            # Normalize for radar plot
            max_vals = [max(mat1_data[prop], mat2_data[prop]) for prop in properties]
            norm_values1 = [v / m if m != 0 else 0 for v, m in zip(values1, max_vals)]
            norm_values2 = [v / m if m != 0 else 0 for v, m in zip(values2, max_vals)]
            
            # Create radar plot
            angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
            norm_values1 += norm_values1[:1]
            norm_values2 += norm_values2[:1]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            ax.plot(angles, norm_values1, 'o-', linewidth=2, label=self.material1.value)
            ax.fill(angles, norm_values1, alpha=0.25)
            
            ax.plot(angles, norm_values2, 'o-', linewidth=2, label=self.material2.value)
            ax.fill(angles, norm_values2, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(properties)
            ax.set_ylim(0, 1)
            ax.set_title('Normalized Property Comparison', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Display table
            comparison_df = pd.DataFrame({
                self.material1.value: values1,
                self.material2.value: values2,
                'Difference': [v2 - v1 for v1, v2 in zip(values1, values2)],
                'Relative (%)': [(v2 - v1) / v1 * 100 if v1 != 0 else 0 
                                 for v1, v2 in zip(values1, values2)]
            }, index=properties)
            
            print("\nDetailed Comparison:")
            display(comparison_df.round(2))


class MLModelTuner:
    """Interactive widget for ML model hyperparameter tuning."""
    
    def __init__(self, callback: Optional[Callable] = None):
        """Initialize ML model tuner.
        
        Args:
            callback: Function to call when training is started
        """
        self.callback = callback
        self._create_widgets()
    
    def _create_widgets(self):
        """Create tuning widgets."""
        # Model type
        self.model_type = widgets.Dropdown(
            options=['Random Forest', 'XGBoost', 'SVM', 'Ensemble'],
            value='Ensemble',
            description='Model Type:',
            style={'description_width': '120px'}
        )
        
        # CV folds
        self.cv_folds = widgets.IntSlider(
            value=5,
            min=3,
            max=10,
            description='CV Folds:',
            style={'description_width': '120px'}
        )
        
        # Random seed
        self.random_seed = widgets.IntText(
            value=42,
            description='Random Seed:',
            style={'description_width': '120px'}
        )
        
        # Target R²
        self.target_r2 = widgets.FloatRangeSlider(
            value=[0.65, 0.75],
            min=0.5,
            max=1.0,
            step=0.05,
            description='Target R²:',
            style={'description_width': '120px'}
        )
        
        # Bootstrap samples
        self.bootstrap_samples = widgets.IntText(
            value=1000,
            description='Bootstrap:',
            style={'description_width': '120px'}
        )
        
        # Train button
        self.train_button = widgets.Button(
            description='Train Model',
            button_style='success',
            icon='cogs'
        )
        self.train_button.on_click(self._on_train)
        
        # Output
        self.output = widgets.Output()
    
    def display(self):
        """Display the tuning interface."""
        layout = widgets.VBox([
            widgets.HTML('<h3>ML Model Configuration</h3>'),
            self.model_type,
            self.cv_folds,
            self.random_seed,
            self.target_r2,
            self.bootstrap_samples,
            self.train_button,
            self.output
        ])
        
        display(layout)
    
    def _on_train(self, _):
        """Handle train button click."""
        config = self.get_config()
        
        with self.output:
            self.output.clear_output()
            print("Training configuration:")
            print(f"  Model: {config['model_type']}")
            print(f"  CV Folds: {config['cv_folds']}")
            print(f"  Random Seed: {config['random_seed']}")
            print(f"  Target R²: {config['target_r2']}")
            print(f"  Bootstrap Samples: {config['bootstrap_samples']}")
            
            if self.callback:
                self.callback(config)
    
    def get_config(self) -> Dict:
        """Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_type': self.model_type.value.lower().replace(' ', '_'),
            'cv_folds': self.cv_folds.value,
            'random_seed': self.random_seed.value,
            'target_r2': self.target_r2.value,
            'bootstrap_samples': self.bootstrap_samples.value
        }
