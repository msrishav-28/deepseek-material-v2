# Ceramic Armor Discovery - Jupyter Notebooks

This directory contains interactive Jupyter notebooks for exploring, analyzing, and visualizing ceramic armor materials data.

## Notebooks Overview

### 00_quickstart_tutorial.ipynb
**Quick introduction to the framework**
- Environment setup and configuration
- Basic screening workflow
- ML model training and prediction
- Results visualization and export

**Recommended for**: New users, quick demonstrations

### 01_exploratory_analysis.ipynb
**Detailed property analysis**
- Material property distributions
- Correlation analysis
- Statistical summaries
- Data quality assessment

**Recommended for**: Initial data exploration, understanding material properties

### 02_ml_model_training.ipynb
**Advanced machine learning workflows**
- Feature engineering
- Model training and validation
- Hyperparameter optimization
- Performance evaluation
- Uncertainty quantification

**Recommended for**: ML practitioners, model development

### 03_dopant_screening.ipynb
**Comprehensive dopant screening workflow**
- High-throughput screening setup
- DFT data collection and validation
- Thermodynamic stability analysis (ΔE_hull ≤ 0.1 eV/atom)
- Multi-objective candidate ranking
- Detailed results visualization

**Recommended for**: Materials researchers, screening campaigns

### 04_mechanism_analysis.ipynb
**Ballistic performance mechanism analysis**
- Feature importance analysis
- Thermal conductivity impact studies
- Property correlation networks
- Mechanistic hypothesis testing
- Dopant effect analysis

**Recommended for**: Understanding performance drivers, mechanism studies

### 05_interactive_exploration.ipynb
**Interactive widgets for data exploration**
- Material property explorer
- Screening configuration interface
- Property comparison tools
- ML model tuning interface
- Custom analysis widgets

**Recommended for**: Interactive exploration, presentations

## Getting Started

### Prerequisites

```bash
# Install Jupyter and required packages
conda install jupyter ipywidgets matplotlib seaborn plotly

# Enable widgets
jupyter nbextension enable --py widgetsnbextension
```

### Running Notebooks

```bash
# Start Jupyter from project root
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

### Recommended Order

1. Start with `00_quickstart_tutorial.ipynb` to understand the basics
2. Explore data with `01_exploratory_analysis.ipynb`
3. Run screening with `03_dopant_screening.ipynb`
4. Train models with `02_ml_model_training.ipynb`
5. Analyze mechanisms with `04_mechanism_analysis.ipynb`
6. Use interactive tools in `05_interactive_exploration.ipynb`

## Interactive Widgets

The notebooks use custom interactive widgets for enhanced exploration:

- **MaterialExplorer**: Interactive property visualization with filters
- **ScreeningConfigurator**: GUI for configuring screening parameters
- **PropertyComparator**: Side-by-side material comparison with radar plots
- **MLModelTuner**: Interactive ML model configuration

## Data Requirements

Most notebooks expect data in the following locations:

- Screening results: `./results/dopant_screening/`
- ML models: `./results/models/`
- Analysis outputs: `./results/analysis/`

If data files are not found, notebooks will either:
- Generate sample data for demonstration
- Provide instructions for running prerequisite workflows

## Tips for Best Results

### Performance
- Use `%matplotlib inline` for static plots
- Use `%matplotlib widget` for interactive plots
- Clear output regularly to manage memory

### Reproducibility
- Always set random seeds: `np.random.seed(42)`
- Document configuration parameters
- Save intermediate results

### Visualization
- Use high DPI for publication-quality figures: `plt.savefig('plot.png', dpi=300)`
- Export interactive plots: `fig.write_html('plot.html')`
- Use consistent color schemes across notebooks

## Customization

### Creating Custom Notebooks

```python
# Template structure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ceramic_discovery.config import config
from ceramic_discovery.screening.screening_engine import ScreeningEngine

# Your analysis code here
```

### Adding Custom Widgets

```python
import ipywidgets as widgets
from IPython.display import display

# Create custom widget
my_widget = widgets.IntSlider(value=5, min=0, max=10)
display(my_widget)
```

## Troubleshooting

### Widgets Not Displaying
```bash
# Reinstall and enable widgets
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Kernel Crashes
- Reduce data size for memory-intensive operations
- Use data chunking for large datasets
- Restart kernel and clear outputs

### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .
```

## Contributing

When adding new notebooks:

1. Follow the naming convention: `##_descriptive_name.ipynb`
2. Include a clear introduction and objectives
3. Add markdown documentation throughout
4. Test with sample data
5. Update this README

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [ipywidgets Documentation](https://ipywidgets.readthedocs.io/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Plotly Documentation](https://plotly.com/python/)

## Support

For issues or questions:
- Check notebook comments and markdown cells
- Review the main project README
- Consult the design document in `.kiro/specs/design.md`
