# Getting Started with Ceramic Armor Discovery Framework

Welcome! This guide will help you get started with the Ceramic Armor Discovery Framework, whether you're a materials scientist, computational researcher, or developer.

## Table of Contents

1. [What is This Framework?](#what-is-this-framework)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [First Steps](#first-steps)
5. [Your First Analysis](#your-first-analysis)
6. [Next Steps](#next-steps)
7. [Getting Help](#getting-help)

---

## What is This Framework?

The Ceramic Armor Discovery Framework is a computational platform that helps researchers:

- **Screen thousands of ceramic materials** to find promising armor candidates
- **Predict ballistic performance** (V₅₀) from fundamental material properties
- **Understand mechanisms** linking properties to performance
- **Accelerate discovery** by prioritizing materials for experimental testing

### Key Capabilities

✅ **DFT Data Collection** - Automated extraction from Materials Project  
✅ **Stability Analysis** - Thermodynamic viability screening  
✅ **Machine Learning** - Property-to-performance predictions  
✅ **High-Throughput Screening** - Parallel processing of large material sets  
✅ **Uncertainty Quantification** - Confidence intervals for all predictions  
✅ **Reproducibility** - Complete provenance tracking and environment management

### Who Should Use This?

- **Materials Scientists** researching ceramic armor materials
- **Computational Researchers** doing high-throughput materials screening
- **Graduate Students** working on materials discovery projects
- **R&D Engineers** evaluating ceramic armor candidates

---

## Prerequisites

### Required Knowledge

- **Basic Python** - You should be comfortable with Python scripts and notebooks
- **Materials Science Fundamentals** - Understanding of crystal structures, properties
- **Command Line** - Basic familiarity with terminal/command prompt

### System Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.11 or higher
- **RAM:** Minimum 8 GB (16 GB recommended for large datasets)
- **Disk Space:** At least 10 GB free space
- **Internet:** Required for Materials Project API access

### Required Accounts

- **Materials Project API Key** (free)
  - Sign up at: https://materialsproject.org/
  - Navigate to Dashboard → API to get your key
  - Keep this key secure - you'll need it during setup

---

## Installation

We provide three installation methods. Choose the one that fits your needs:

### Option 1: Quick Install (Recommended for Beginners)

This is the fastest way to get started:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Install Python dependencies
pip install -e .

# 3. Set up environment variables
cp .env.example .env

# 4. Edit .env file with your API key
# On Windows: notepad .env
# On Mac/Linux: nano .env

# 5. Verify installation
python -m ceramic_discovery.cli status
```

### Option 2: Conda Environment (Recommended for Research)

Use this for reproducible research environments:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate ceramic-armor-discovery

# 4. Install package
pip install -e .

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your API key

# 6. Verify installation
ceramic-discovery status
```

### Option 3: Docker (Full Stack with Database)

Use this for complete deployment with all services:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Set up environment
cp .env.example .env
# Edit .env with your API key

# 3. Start all services
docker-compose up -d

# 4. Access Jupyter Lab
# Open browser to: http://localhost:8888

# 5. Check service status
docker-compose ps
```

---

## First Steps

### 1. Verify Your Installation

Run the health check to ensure everything is configured correctly:

```bash
ceramic-discovery health
```

You should see:
```
✓ Configuration loaded successfully
✓ API key configured
✓ Data directories exist
✓ All dependencies installed
```

### 2. Understand the Project Structure

```
ceramic-armor-discovery/
├── src/ceramic_discovery/      # Main Python package
│   ├── dft/                    # DFT data collection and stability analysis
│   ├── ml/                     # Machine learning models
│   ├── ceramics/               # Ceramic systems and properties
│   ├── ballistics/             # Ballistic performance prediction
│   ├── screening/              # High-throughput screening workflows
│   ├── validation/             # Validation and reproducibility
│   └── analysis/               # Analysis and visualization
│
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   ├── 00_quickstart_tutorial.ipynb
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_ml_model_training.ipynb
│   ├── 03_dopant_screening.ipynb
│   └── 04_mechanism_analysis.ipynb
│
├── examples/                   # Example Python scripts
├── tests/                      # Test suite
├── data/                       # Data storage
├── results/                    # Analysis results
├── docs/                       # Documentation
└── .env                        # Your configuration (API keys, etc.)
```

### 3. Configure Your Environment

Edit the `.env` file with your settings:

```bash
# Required: Your Materials Project API key
MATERIALS_PROJECT_API_KEY=your_api_key_here

# Optional: Computational resources
MAX_WORKERS=4                    # Number of parallel workers
MEMORY_LIMIT_GB=16              # Memory limit

# Optional: Random seed for reproducibility
ML_RANDOM_SEED=42

# Optional: Database (if using Docker)
DATABASE_URL=postgresql://user:password@localhost:5432/ceramic_discovery
```

### 4. Test the Command-Line Interface

Try these commands to familiarize yourself with the CLI:

```bash
# Show version
ceramic-discovery --version

# Show configuration
ceramic-discovery status

# Show help
ceramic-discovery --help

# Check system health
ceramic-discovery health
```

---

## Your First Analysis

Let's run a simple analysis to predict properties for a ceramic material.

### Step 1: Start Jupyter Lab

```bash
# If using conda or pip installation:
jupyter lab

# If using Docker:
# Already running at http://localhost:8888
```

### Step 2: Open the Quickstart Tutorial

Navigate to `notebooks/00_quickstart_tutorial.ipynb` and follow along.

### Step 3: Run a Simple Prediction

Here's a minimal example you can run in a Python script or notebook:

```python
from ceramic_discovery.ceramics import CeramicSystemFactory
from ceramic_discovery.ballistics import BallisticPredictor

# Create a baseline SiC system
sic = CeramicSystemFactory.create_sic()

# Display properties
print(f"Material: {sic.formula}")
print(f"Hardness: {sic.mechanical.hardness} GPa")
print(f"Fracture Toughness: {sic.mechanical.fracture_toughness} MPa·m^0.5")
print(f"Density: {sic.mechanical.density} g/cm³")

# Load ballistic predictor (requires trained model)
predictor = BallisticPredictor()
predictor.load_model("./models/v50_predictor_v1.pkl")

# Predict V₅₀
prediction = predictor.predict_v50(sic.to_dict())
print(f"\nPredicted V₅₀: {prediction['v50']:.0f} m/s")
print(f"95% CI: [{prediction['confidence_interval'][0]:.0f}, {prediction['confidence_interval'][1]:.0f}] m/s")
print(f"Reliability: {prediction['reliability_score']:.2f}")
```

### Step 4: Explore the Notebooks

Work through the tutorial notebooks in order:

1. **00_quickstart_tutorial.ipynb** - Basic usage and concepts
2. **01_exploratory_analysis.ipynb** - Data exploration
3. **02_ml_model_training.ipynb** - Training ML models
4. **03_dopant_screening.ipynb** - Screening dopant candidates
5. **04_mechanism_analysis.ipynb** - Understanding mechanisms

---

## Next Steps

### Learn the Workflows

The framework supports several research workflows:

1. **Dopant Screening** - Screen dopants for a ceramic system
   - See: `docs/USER_GUIDE.md` → Workflow 1
   - Example: `examples/screening_workflow_example.py`

2. **Property Prediction** - Predict properties for new materials
   - See: `docs/USER_GUIDE.md` → Workflow 2
   - Example: `examples/ballistic_prediction_example.py`

3. **Mechanism Investigation** - Understand property-performance relationships
   - See: `docs/USER_GUIDE.md` → Workflow 4
   - Example: `examples/validation_example.py`

### Explore the Documentation

- **User Guide** (`docs/USER_GUIDE.md`) - Detailed workflow instructions
- **API Reference** (`docs/API_REFERENCE.md`) - Complete API documentation
- **Methodology** (`docs/METHODOLOGY.md`) - Scientific methodology and assumptions
- **Case Studies** (`docs/CASE_STUDIES.md`) - Real-world examples

### Run Example Scripts

Try the example scripts in the `examples/` directory:

```bash
# Screening workflow
python examples/screening_workflow_example.py

# Ballistic prediction
python examples/ballistic_prediction_example.py

# Performance optimization
python examples/performance_optimization_example.py

# Validation
python examples/validation_example.py
```

### Customize for Your Research

1. **Define your ceramic system** - Add new baseline systems
2. **Specify dopants** - Choose elements to screen
3. **Set screening criteria** - Define stability and property thresholds
4. **Run screening** - Execute high-throughput workflow
5. **Analyze results** - Visualize and interpret findings

---

## Getting Help

### Documentation

- **README.md** - Project overview and quick start
- **QUICKSTART.md** - Rapid installation guide
- **docs/USER_GUIDE.md** - Comprehensive user guide
- **docs/API_REFERENCE.md** - API documentation
- **docs/METHODOLOGY.md** - Scientific methodology

### Troubleshooting

Common issues and solutions:

#### "Module not found" errors
```bash
# Make sure you installed the package
pip install -e .
```

#### "API key not configured"
```bash
# Check your .env file
cat .env | grep MATERIALS_PROJECT_API_KEY

# Make sure it's set correctly
export MATERIALS_PROJECT_API_KEY=your_key_here  # Linux/Mac
set MATERIALS_PROJECT_API_KEY=your_key_here     # Windows
```

#### "Database connection failed"
```bash
# If using Docker, ensure services are running
docker-compose ps

# Restart services if needed
docker-compose restart
```

#### Memory errors
```python
# Reduce batch size or number of workers
from ceramic_discovery.config import Config
config = Config.load()
config.max_workers = 2  # Reduce from default
```

### Community Support

- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share ideas
- **Examples** - Check `examples/` directory for code samples
- **Tests** - Look at `tests/` for usage examples

### Running Tests

Verify your installation by running the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=ceramic_discovery --cov-report=html

# Run specific test file
pytest tests/test_ballistic_predictor.py -v
```

---

## Quick Reference

### Essential Commands

```bash
# Check system status
ceramic-discovery status

# Check health
ceramic-discovery health

# Run screening (when implemented)
ceramic-discovery screen --config config.yaml

# Make predictions (when implemented)
ceramic-discovery predict --model model.pkl --data data.csv
```

### Essential Python Imports

```python
# DFT and stability
from ceramic_discovery.dft import StabilityAnalyzer, MaterialsProjectClient

# Ceramic systems
from ceramic_discovery.ceramics import CeramicSystemFactory

# Machine learning
from ceramic_discovery.ml import ModelTrainer, FeatureEngineeringPipeline

# Ballistic prediction
from ceramic_discovery.ballistics import BallisticPredictor, MechanismAnalyzer

# Screening
from ceramic_discovery.screening import ScreeningEngine

# Validation
from ceramic_discovery.validation import PhysicalPlausibilityValidator
```

### Key Concepts

- **ΔE_hull** - Energy above convex hull (stability metric)
  - ≤ 0.1 eV/atom = viable for synthesis
  
- **V₅₀** - Ballistic limit velocity (performance metric)
  - Higher = better ballistic performance
  
- **Tier 1-2 Properties** - Fundamental properties used as ML inputs
  - Tier 1: DFT-derived (formation energy, band gap, density)
  - Tier 2: Measurable (hardness, fracture toughness)
  
- **Uncertainty Quantification** - All predictions include confidence intervals
  - 95% CI reported for all predictions
  - Reliability score (0-1) indicates confidence

---

## What's Next?

Now that you're set up, here are some suggested paths:

### For Materials Scientists
1. Work through `00_quickstart_tutorial.ipynb`
2. Read `docs/METHODOLOGY.md` to understand the science
3. Try `03_dopant_screening.ipynb` with your ceramic system
4. Explore `04_mechanism_analysis.ipynb` for insights

### For Computational Researchers
1. Review `docs/API_REFERENCE.md` for API details
2. Examine `examples/` for workflow patterns
3. Read `docs/METHODOLOGY.md` for computational details
4. Customize workflows for your research

### For Developers
1. Review the codebase structure in `src/ceramic_discovery/`
2. Run the test suite: `pytest tests/`
3. Read `CONTRIBUTING.md` (if available)
4. Check open issues on GitHub

---

## Success Checklist

You're ready to start when you can check all these boxes:

- [ ] Installation completed successfully
- [ ] `ceramic-discovery status` runs without errors
- [ ] `ceramic-discovery health` shows all checks passing
- [ ] `.env` file configured with your API key
- [ ] Jupyter Lab opens and notebooks load
- [ ] Example scripts run successfully
- [ ] Test suite passes: `pytest tests/`

---

## Additional Resources

### Scientific Background

- **Materials Project** - https://materialsproject.org/
- **DFT Basics** - Density Functional Theory fundamentals
- **Ceramic Armor** - Ballistic performance of ceramics

### Python and Data Science

- **Python Tutorial** - https://docs.python.org/3/tutorial/
- **NumPy** - https://numpy.org/doc/
- **Pandas** - https://pandas.pydata.org/docs/
- **Scikit-learn** - https://scikit-learn.org/

### Tools and Frameworks

- **Jupyter** - https://jupyter.org/
- **Docker** - https://docs.docker.com/
- **Conda** - https://docs.conda.io/

---

**Welcome to the Ceramic Armor Discovery Framework!**

We're excited to have you here. If you have questions, don't hesitate to:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Look at examples in `examples/` and `notebooks/`

Happy discovering! 🚀
