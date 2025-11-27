# Ceramic Armor Discovery Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A computational materials science platform for accelerating the discovery of high-performance ceramic armor materials using Density Functional Theory (DFT) and machine learning.

## 🎯 Overview

The Ceramic Armor Discovery Framework enables researchers to:

- **Screen thousands of ceramic materials** efficiently using high-throughput computational methods
- **Predict ballistic performance** (V₅₀) from fundamental material properties
- **Identify promising dopants** for ceramic armor systems (SiC, B₄C, WC, TiC, Al₂O₃)
- **Understand mechanisms** linking properties to performance
- **Accelerate experimental validation** by prioritizing the most promising candidates

### Key Capabilities

✅ **Multi-Source Data Integration** - Load from Materials Project, JARVIS-DFT, NIST-JANAF, and literature databases  
✅ **DFT Data Collection** - Automated extraction from Materials Project API  
✅ **Stability Analysis** - Thermodynamic viability screening (ΔE_hull ≤ 0.1 eV/atom)  
✅ **Advanced Feature Engineering** - Composition and structure-based descriptors for ML  
✅ **Machine Learning** - Property-to-performance predictions with uncertainty quantification  
✅ **Application-Specific Ranking** - Rank materials for aerospace, cutting tools, thermal barriers, and more  
✅ **Experimental Planning** - Automated synthesis and characterization protocol generation  
✅ **High-Throughput Screening** - Parallel processing for large material sets  
✅ **Reproducibility** - Complete provenance tracking and environment management  
✅ **Production Ready** - Docker deployment, Kubernetes support, CI/CD pipeline

## 🚀 Quick Start

### For First-Time Users

**New to the framework?** Start here:

1. **Read the [Getting Started Guide](GETTING_STARTED.md)** - Complete introduction for beginners
2. **Follow the [Installation Guide](INSTALLATION.md)** - Detailed setup instructions
3. **Try the [Quickstart Tutorial](notebooks/00_quickstart_tutorial.ipynb)** - Interactive introduction

### Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Install with conda (recommended)
conda env create -f environment.yml
conda activate ceramic-armor-discovery
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env with your Materials Project API key

# 4. Verify installation
ceramic-discovery health
```

**Get your Materials Project API key:** https://materialsproject.org/

### Prerequisites

- **Python** 3.11 or higher
- **Materials Project API key** (free registration)
- **8 GB RAM** minimum (16 GB recommended)
- **10 GB disk space** for data storage

**Optional:**
- Conda/Miniconda for environment management
- Docker for containerized deployment
- PostgreSQL for database features

## 📚 Documentation

### Essential Guides

- **[Getting Started](GETTING_STARTED.md)** - Complete guide for first-time users
- **[Installation](INSTALLATION.md)** - Detailed installation instructions for all platforms
- **[User Guide](docs/USER_GUIDE.md)** - Comprehensive workflows and examples
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[JARVIS Integration](docs/JARVIS_INTEGRATION.md)** - Guide to using JARVIS-DFT data
- **[Application Ranking](docs/APPLICATION_RANKING.md)** - Application-specific material ranking
- **[Methodology](docs/METHODOLOGY.md)** - Scientific methodology and assumptions
- **[Contributing](CONTRIBUTING.md)** - Guidelines for contributors

### Quick Links

- [Troubleshooting](INSTALLATION.md#troubleshooting)
- [Example Scripts](examples/)
- [Jupyter Notebooks](notebooks/)
- [Case Studies](docs/CASE_STUDIES.md)
- [Changelog](CHANGELOG.md)

## 💡 Usage Examples

### Example 1: Load Multi-Source Data

```python
from ceramic_discovery.dft import JarvisClient, DataCombiner, MaterialsProjectClient

# Load JARVIS-DFT carbides
jarvis_client = JarvisClient("./data/jdft_3d-7-7-2018.json")
carbides = jarvis_client.load_carbides(metal_elements={"Si", "Ti", "Zr", "Hf"})

# Load Materials Project data
mp_client = MaterialsProjectClient(api_key="your_api_key")
mp_data = mp_client.search_materials("SiC")

# Combine sources
combiner = DataCombiner()
combined = combiner.combine_sources(
    mp_data=mp_data,
    jarvis_data=carbides,
    nist_data={},
    literature_data={}
)

print(f"Combined {len(combined)} unique materials from multiple sources")
```

### Example 2: Application-Specific Ranking

```python
from ceramic_discovery.screening import ApplicationRanker

# Initialize ranker
ranker = ApplicationRanker()

# Rank materials for aerospace application
aerospace_ranking = ranker.rank_materials(
    materials=materials,
    application="aerospace_hypersonic"
)

# Display top candidates
for i, ranked in enumerate(aerospace_ranking[:5], 1):
    print(f"{i}. {ranked.formula}: Score {ranked.overall_score:.3f}")
```

### Example 3: Experimental Planning

```python
from ceramic_discovery.validation import ExperimentalPlanner

# Initialize planner
planner = ExperimentalPlanner()

# Design synthesis protocol
methods = planner.design_synthesis_protocol(
    formula="HfC",
    properties={"melting_point": 3900, "hardness": 28.0}
)

# Design characterization plan
techniques = planner.design_characterization_plan(
    formula="HfC",
    target_properties=["hardness", "thermal_conductivity"]
)

# Estimate resources
estimate = planner.estimate_resources(
    synthesis_methods=methods,
    characterization_plan=techniques
)

print(f"Timeline: {estimate.timeline_months:.1f} months")
print(f"Cost: ${estimate.estimated_cost_k_dollars:.1f}K")
```

### Example 4: Predict Ballistic Performance

```python
from ceramic_discovery.ceramics import CeramicSystemFactory
from ceramic_discovery.ballistics import BallisticPredictor

# Create baseline SiC system
sic = CeramicSystemFactory.create_sic()

# Load predictor
predictor = BallisticPredictor()
predictor.load_model("./models/v50_predictor_v1.pkl")

# Predict V₅₀
prediction = predictor.predict_v50(sic.to_dict())
print(f"Predicted V₅₀: {prediction['v50']:.0f} m/s")
print(f"95% CI: [{prediction['confidence_interval'][0]:.0f}, "
      f"{prediction['confidence_interval'][1]:.0f}] m/s")
```

## 🔬 Research Workflows

The framework supports several research workflows:

### 1. Dopant Screening Workflow
Screen dopant candidates for a ceramic system and identify promising materials.

**See:** [User Guide - Workflow 1](docs/USER_GUIDE.md#workflow-1-dopant-screening)  
**Example:** [screening_workflow_example.py](examples/screening_workflow_example.py)

### 2. Property Prediction Workflow
Predict properties for new material compositions using ML models.

**See:** [User Guide - Workflow 2](docs/USER_GUIDE.md#workflow-2-property-prediction)  
**Example:** [ballistic_prediction_example.py](examples/ballistic_prediction_example.py)

### 3. Ballistic Performance Analysis
Analyze ballistic performance predictions and identify key drivers.

**See:** [User Guide - Workflow 3](docs/USER_GUIDE.md#workflow-3-ballistic-performance-analysis)

### 4. Mechanism Investigation
Investigate physical mechanisms underlying property-performance relationships.

**See:** [User Guide - Workflow 4](docs/USER_GUIDE.md#workflow-4-mechanism-investigation)  
**Example:** [validation_example.py](examples/validation_example.py)

### 5. Multi-Source Data Integration
Load and combine data from JARVIS-DFT, NIST-JANAF, Materials Project, and literature.

**See:** [User Guide - Workflow 5](docs/USER_GUIDE.md#workflow-5-multi-source-data-integration)  
**Guide:** [JARVIS Integration](docs/JARVIS_INTEGRATION.md)  
**Example:** [jarvis_data_loading_example.py](examples/jarvis_data_loading_example.py)

### 6. Application-Specific Ranking
Rank materials for specific applications (aerospace, cutting tools, thermal barriers).

**See:** [User Guide - Workflow 6](docs/USER_GUIDE.md#workflow-6-application-specific-material-ranking)  
**Guide:** [Application Ranking](docs/APPLICATION_RANKING.md)  
**Example:** [application_ranking_example.py](examples/application_ranking_example.py)

### 7. Experimental Planning
Generate synthesis and characterization protocols for top candidates.

**See:** [User Guide - Workflow 7](docs/USER_GUIDE.md#workflow-7-experimental-planning)  
**Example:** [experimental_planning_example.py](examples/experimental_planning_example.py)

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API key

# 2. Start all services
docker-compose up -d

# 3. Access Jupyter Lab
# Open browser to: http://localhost:8888

# 4. Check status
docker-compose ps
```

### Services Included

- **Application** - Python environment with all dependencies
- **PostgreSQL** - Database with pgvector extension
- **Redis** - Caching layer
- **Jupyter Lab** - Interactive notebooks

### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

## 📁 Project Structure

```
ceramic-armor-discovery/
├── src/ceramic_discovery/          # Main Python package
│   ├── dft/                        # DFT data collection and stability analysis
│   ├── ml/                         # Machine learning models
│   ├── ceramics/                   # Ceramic systems and properties
│   ├── ballistics/                 # Ballistic performance prediction
│   ├── screening/                  # High-throughput screening
│   ├── validation/                 # Validation and reproducibility
│   ├── analysis/                   # Analysis and visualization
│   ├── monitoring/                 # Performance and quality monitoring
│   ├── cost_management/            # Resource and budget tracking
│   ├── reporting/                  # Report generation
│   ├── performance/                # Performance optimization
│   ├── config.py                   # Configuration management
│   ├── cli.py                      # Command-line interface
│   └── health.py                   # Health checks
│
├── notebooks/                      # Jupyter notebooks
│   ├── 00_quickstart_tutorial.ipynb
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_ml_model_training.ipynb
│   ├── 03_dopant_screening.ipynb
│   ├── 04_mechanism_analysis.ipynb
│   └── 05_interactive_exploration.ipynb
│
├── examples/                       # Example Python scripts
│   ├── screening_workflow_example.py
│   ├── ballistic_prediction_example.py
│   ├── performance_optimization_example.py
│   └── validation_example.py
│
├── tests/                          # Comprehensive test suite
├── docs/                           # Documentation
│   ├── USER_GUIDE.md
│   ├── API_REFERENCE.md
│   ├── METHODOLOGY.md
│   └── CASE_STUDIES.md
│
├── data/                           # Data storage
│   └── hdf5/                       # HDF5 datasets
├── results/                        # Analysis results
├── logs/                           # Log files
├── models/                         # Trained ML models
│
├── k8s/                            # Kubernetes manifests
├── .github/                        # CI/CD workflows
├── docker-compose.yml              # Docker services
├── Dockerfile                      # Container definition
├── environment.yml                 # Conda environment
└── setup.py                        # Package setup
```

## 🖥️ Command Line Interface

```bash
# System health and status
ceramic-discovery health              # Check system health
ceramic-discovery status              # Show configuration

# Workflows (when implemented)
ceramic-discovery screen --config config.yaml
ceramic-discovery predict --model model.pkl --data data.csv

# Database management
ceramic-discovery db-init             # Initialize database
ceramic-discovery db-migrate          # Run migrations
ceramic-discovery db-status           # Check database status

# Utilities
ceramic-discovery --version           # Show version
ceramic-discovery --help              # Show help
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ceramic_discovery --cov-report=html

# Run specific test file
pytest tests/test_ballistic_predictor.py -v

# Run tests matching pattern
pytest tests/ -k "stability" -v
```

### Test Coverage

Current test coverage: **90%+**

- Unit tests for all modules
- Integration tests for workflows
- System validation tests
- Performance tests
- Scientific accuracy tests

## 🔧 Development

### Setup Development Environment

```bash
# Fork and clone repository
git clone https://github.com/YOUR_USERNAME/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# Create environment
conda env create -f environment.yml
conda activate ceramic-armor-discovery

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
make lint
```

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code of conduct
- Development setup
- Coding standards
- Testing guidelines
- Pull request process

## 🔬 Scientific Methodology

### Key Design Principles

- **Correct DFT Criteria**: ΔE_hull ≤ 0.1 eV/atom for metastable compounds
- **Unit Standardization**: All fracture toughness in MPa·m^0.5
- **Realistic ML Targets**: R² = 0.65-0.75 with proper uncertainty quantification
- **Reproducibility**: Complete provenance tracking and environment pinning
- **Tier 1-2 Features**: Only fundamental properties used as ML inputs

### Assumptions and Limitations

- **DFT Calculations**: 0 K approximation, perfect crystals
- **ML Predictions**: Microstructure effects not explicitly modeled
- **Property Models**: Rule-of-mixtures approximations (±20-30% accuracy)
- **Ballistic Performance**: Standard 7.62mm AP threat

**See [METHODOLOGY.md](docs/METHODOLOGY.md) for complete details.**

## 📊 Performance

### Computational Efficiency

- **Parallel Processing**: Multi-core support for batch operations
- **Caching**: Redis-based caching for repeated computations
- **Memory Management**: Efficient handling of large datasets
- **HPC Integration**: SLURM support for cluster computing

### Benchmarks

- **Screening**: ~1000 materials/hour (4 cores)
- **ML Prediction**: ~10,000 predictions/second
- **DFT Data Collection**: Limited by Materials Project API rate

## 🚢 Deployment

### Docker Deployment

See [Docker Deployment](#-docker-deployment) section above.

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods -n ceramic-discovery

# View logs
kubectl logs -f deployment/ceramic-discovery-app
```

### CI/CD Pipeline

GitHub Actions workflow for:
- Automated testing
- Code quality checks
- Docker image building
- Documentation deployment

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{ceramic_armor_discovery,
  title = {Ceramic Armor Discovery Framework},
  author = {Research Team},
  year = {2025},
  url = {https://github.com/your-org/ceramic-armor-discovery},
  version = {0.1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

### Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Examples**: See [examples/](examples/) and [notebooks/](notebooks/)
- **Issues**: [GitHub Issues](https://github.com/your-org/ceramic-armor-discovery/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ceramic-armor-discovery/discussions)

### Troubleshooting

Common issues and solutions are documented in:
- [Installation Guide - Troubleshooting](INSTALLATION.md#troubleshooting)
- [User Guide - Troubleshooting](docs/USER_GUIDE.md#troubleshooting)

### Community

- Report bugs via GitHub Issues
- Request features via GitHub Issues
- Ask questions via GitHub Discussions
- Contribute via Pull Requests

## 🙏 Acknowledgments

- **Materials Project** for providing DFT data
- **Scientific Python Community** for excellent tools
- **All Contributors** to this project

## 📈 Project Status

**Current Version:** 0.1.0 (Alpha)  
**Status:** Active Development  
**Test Coverage:** 90%+  
**Documentation:** Complete

See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [CHANGELOG.md](CHANGELOG.md) for details.

---

**Ready to discover ceramic armor materials? Start with the [Getting Started Guide](GETTING_STARTED.md)!** 🚀
