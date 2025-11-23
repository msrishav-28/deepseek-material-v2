# Installation Guide

Complete installation instructions for the Ceramic Armor Discovery Framework.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Upgrading](#upgrading)
7. [Uninstallation](#uninstallation)

---

## System Requirements

### Operating Systems

- **Windows** 10/11 (64-bit)
- **macOS** 10.15 (Catalina) or later
- **Linux** Ubuntu 20.04+, CentOS 8+, or equivalent

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 10 GB free space
- Internet connection

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Disk: 50 GB free space (for data storage)
- SSD for better performance

### Software Prerequisites

**Required:**
- Python 3.11 or higher
- pip (Python package manager)
- Git

**Optional:**
- Conda/Miniconda (recommended for environment management)
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL 14+ (for database features)
- Redis (for caching)

---

## Installation Methods

### Method 1: Quick Install with pip

**Best for:** Quick testing, simple use cases

```bash
# 1. Clone repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Install package
pip install -e .

# 3. Install development dependencies (optional)
pip install -e ".[dev]"

# 4. Verify installation
python -m ceramic_discovery.cli --version
```

**Pros:**
- Fast and simple
- No additional tools required
- Works on all platforms

**Cons:**
- No environment isolation
- May conflict with other packages
- Less reproducible

---

### Method 2: Conda Environment (Recommended)

**Best for:** Research, reproducibility, avoiding conflicts

#### Step 1: Install Conda

If you don't have Conda installed:

**Windows:**
```bash
# Download Miniconda installer from:
# https://docs.conda.io/en/latest/miniconda.html
# Run the installer and follow prompts
```

**macOS:**
```bash
# Using Homebrew
brew install --cask miniconda

# Or download from:
# https://docs.conda.io/en/latest/miniconda.html
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Step 2: Create Environment

```bash
# 1. Clone repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Create conda environment from file
conda env create -f environment.yml

# 3. Activate environment
conda activate ceramic-armor-discovery

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
ceramic-discovery --version
```

#### Step 3: Configure Environment

```bash
# Set up environment variables
cp .env.example .env

# Edit .env file with your settings
# Windows: notepad .env
# macOS/Linux: nano .env
```

**Pros:**
- Complete environment isolation
- Reproducible across systems
- Includes all scientific libraries
- Easy to manage dependencies

**Cons:**
- Requires Conda installation
- Larger disk space usage
- Slightly slower initial setup

---

### Method 3: Docker Deployment

**Best for:** Full-stack deployment, production use, team collaboration

#### Step 1: Install Docker

**Windows:**
```bash
# Download Docker Desktop from:
# https://www.docker.com/products/docker-desktop
# Install and start Docker Desktop
```

**macOS:**
```bash
# Using Homebrew
brew install --cask docker

# Or download Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# CentOS/RHEL
sudo yum install docker docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

#### Step 2: Deploy with Docker Compose

```bash
# 1. Clone repository
git clone https://github.com/your-org/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 3. Start all services
docker-compose up -d

# 4. Check service status
docker-compose ps

# 5. View logs
docker-compose logs -f

# 6. Access Jupyter Lab
# Open browser to: http://localhost:8888
```

#### Services Included

- **Application** - Main Python environment
- **PostgreSQL** - Database with pgvector extension
- **Redis** - Caching layer
- **Jupyter Lab** - Interactive notebooks

#### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f [service_name]

# Execute command in container
docker-compose exec app python -m ceramic_discovery.cli status

# Rebuild containers
docker-compose build

# Remove all data (WARNING: deletes volumes)
docker-compose down -v
```

**Pros:**
- Complete isolated environment
- All services included
- Easy deployment
- Consistent across platforms
- Production-ready

**Cons:**
- Requires Docker installation
- Higher resource usage
- More complex troubleshooting

---

### Method 4: Development Installation

**Best for:** Contributing to the project, development work

```bash
# 1. Fork repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# 3. Add upstream remote
git remote add upstream https://github.com/original-org/ceramic-armor-discovery.git

# 4. Create conda environment
conda env create -f environment.yml
conda activate ceramic-armor-discovery

# 5. Install in development mode with dev dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# 7. Run tests to verify
pytest tests/

# 8. Check code quality
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# ============================================
# REQUIRED SETTINGS
# ============================================

# Materials Project API Key (REQUIRED)
# Get your key from: https://materialsproject.org/
MATERIALS_PROJECT_API_KEY=your_api_key_here

# ============================================
# OPTIONAL SETTINGS
# ============================================

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ceramic_discovery

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Data Storage
HDF5_DATA_PATH=./data/hdf5
RESULTS_DIR=./results
LOGS_DIR=./logs

# Computational Resources
MAX_WORKERS=4
MEMORY_LIMIT_GB=16
ENABLE_GPU=false

# Machine Learning
ML_RANDOM_SEED=42
ML_MODEL_DIR=./models

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed

# HPC Settings (if using SLURM)
SLURM_PARTITION=compute
SLURM_TIME_LIMIT=24:00:00
SLURM_NODES=1
SLURM_NTASKS_PER_NODE=16
```

### API Key Setup

#### Get Materials Project API Key

1. Go to https://materialsproject.org/
2. Create account or log in
3. Navigate to Dashboard → API
4. Copy your API key
5. Add to `.env` file:
   ```bash
   MATERIALS_PROJECT_API_KEY=your_key_here
   ```

#### Verify API Key

```bash
# Test API connection
python -c "from ceramic_discovery.dft import MaterialsProjectClient; client = MaterialsProjectClient(); print('API key valid!')"
```

### Directory Structure

The framework will create these directories automatically:

```
ceramic-armor-discovery/
├── data/
│   └── hdf5/          # HDF5 data storage
├── results/           # Analysis results
│   ├── experiments/
│   ├── checkpoints/
│   └── reproducibility_packages/
├── logs/              # Log files
│   ├── alerts/
│   ├── data_quality/
│   ├── performance/
│   └── progress/
└── models/            # Trained ML models
```

To create manually:

```bash
mkdir -p data/hdf5 results logs models
```

---

## Verification

### Basic Verification

```bash
# 1. Check version
ceramic-discovery --version

# 2. Check configuration
ceramic-discovery status

# 3. Run health check
ceramic-discovery health
```

Expected output:
```
✓ Configuration loaded successfully
✓ API key configured
✓ Data directories exist
✓ All dependencies installed
✓ Database connection: OK (if configured)
✓ Redis connection: OK (if configured)
```

### Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ceramic_discovery --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run tests matching pattern
pytest tests/ -k "stability" -v
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or if using Docker
# Already running at http://localhost:8888

# Open and run: notebooks/00_quickstart_tutorial.ipynb
```

### Example Scripts

```bash
# Run example scripts
python examples/screening_workflow_example.py
python examples/ballistic_prediction_example.py
```

---

## Troubleshooting

### Common Issues

#### Issue: "Module not found: ceramic_discovery"

**Solution:**
```bash
# Make sure you installed the package
pip install -e .

# Check if package is installed
pip list | grep ceramic

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Issue: "API key not configured"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Check API key is set
cat .env | grep MATERIALS_PROJECT_API_KEY

# Set environment variable manually (temporary)
export MATERIALS_PROJECT_API_KEY=your_key_here  # Linux/Mac
set MATERIALS_PROJECT_API_KEY=your_key_here     # Windows
```

#### Issue: "Permission denied" errors

**Solution:**
```bash
# Linux/Mac: Fix permissions
chmod +x scripts/*.sh
chmod -R u+w data/ results/ logs/

# Windows: Run as administrator or check folder permissions
```

#### Issue: "Database connection failed"

**Solution:**
```bash
# Check if PostgreSQL is running
# Docker:
docker-compose ps

# Local installation:
# Linux: sudo systemctl status postgresql
# Mac: brew services list
# Windows: Check Services app

# Test connection
psql -h localhost -U user -d ceramic_discovery

# Check DATABASE_URL in .env
```

#### Issue: "Out of memory" errors

**Solution:**
```python
# Reduce computational resources in .env
MAX_WORKERS=2
MEMORY_LIMIT_GB=8

# Or in Python:
from ceramic_discovery.config import Config
config = Config.load()
config.max_workers = 2
```

#### Issue: Conda environment conflicts

**Solution:**
```bash
# Remove and recreate environment
conda deactivate
conda env remove -n ceramic-armor-discovery
conda env create -f environment.yml
conda activate ceramic-armor-discovery
pip install -e .
```

#### Issue: Docker containers won't start

**Solution:**
```bash
# Check Docker is running
docker ps

# View logs
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check port conflicts
# Change ports in docker-compose.yml if needed
```

### Platform-Specific Issues

#### Windows

**Issue:** Long path errors
```bash
# Enable long paths in Windows
# Run as Administrator:
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Issue:** Line ending issues
```bash
# Configure Git to handle line endings
git config --global core.autocrlf true
```

#### macOS

**Issue:** SSL certificate errors
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command
```

**Issue:** Command not found after conda install
```bash
# Initialize conda for your shell
conda init zsh  # or bash
# Restart terminal
```

#### Linux

**Issue:** Missing system dependencies
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

---

## Upgrading

### Upgrade Package

```bash
# Pull latest changes
git pull origin main

# Update conda environment
conda env update -f environment.yml

# Reinstall package
pip install -e . --upgrade

# Run tests
pytest tests/
```

### Upgrade Dependencies

```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Or with conda
conda update --all
```

### Database Migrations

```bash
# If database schema changed
ceramic-discovery db-migrate

# Or manually
psql -h localhost -U user -d ceramic_discovery -f scripts/migrations/001_update_schema.sql
```

---

## Uninstallation

### Remove Package

```bash
# Uninstall package
pip uninstall ceramic-armor-discovery

# Remove conda environment
conda deactivate
conda env remove -n ceramic-armor-discovery
```

### Remove Docker Containers

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker rmi ceramic-armor-discovery_app
```

### Clean Up Files

```bash
# Remove data (optional)
rm -rf data/ results/ logs/

# Remove repository
cd ..
rm -rf ceramic-armor-discovery/
```

---

## Next Steps

After successful installation:

1. **Read the Getting Started guide** - `GETTING_STARTED.md`
2. **Try the quickstart tutorial** - `notebooks/00_quickstart_tutorial.ipynb`
3. **Explore the documentation** - `docs/USER_GUIDE.md`
4. **Run example scripts** - `examples/`

---

## Support

If you encounter issues not covered here:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Open a new issue with:
   - Your operating system
   - Python version
   - Installation method used
   - Complete error message
   - Steps to reproduce

---

**Installation complete! Ready to discover ceramic armor materials! 🚀**
