# Contributing to Ceramic Armor Discovery Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Acknowledge different viewpoints and experiences
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Git for version control
- Familiarity with materials science concepts
- Basic understanding of DFT and machine learning

### Find an Issue

Good first contributions:

1. **Documentation improvements** - Fix typos, clarify explanations
2. **Bug fixes** - Address issues labeled "bug"
3. **Tests** - Add test coverage for existing code
4. **Examples** - Create new example scripts or notebooks

Check the issue tracker for:
- `good first issue` - Suitable for newcomers
- `help wanted` - Community contributions welcome
- `documentation` - Documentation improvements needed

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ceramic-armor-discovery.git
cd ceramic-armor-discovery

# Add upstream remote
git remote add upstream https://github.com/original-org/ceramic-armor-discovery.git
```

### 2. Create Development Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ceramic-armor-discovery

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

---

## How to Contribute

### Reporting Bugs

When reporting bugs, include:

1. **Description** - Clear description of the bug
2. **Steps to reproduce** - Minimal example to reproduce
3. **Expected behavior** - What should happen
4. **Actual behavior** - What actually happens
5. **Environment** - OS, Python version, package versions
6. **Error messages** - Full error traceback

Example bug report:

```markdown
**Description**
StabilityAnalyzer crashes when energy_above_hull is None

**Steps to Reproduce**
```python
from ceramic_discovery.dft import StabilityAnalyzer
analyzer = StabilityAnalyzer()
result = analyzer.analyze_stability("mp-149", "SiC", None, -0.65)
```

**Expected Behavior**
Should handle None gracefully or raise informative error

**Actual Behavior**
TypeError: unsupported operand type(s) for <=: 'NoneType' and 'float'

**Environment**
- OS: Windows 11
- Python: 3.11.5
- Package version: 0.1.0
```

### Suggesting Enhancements

When suggesting enhancements, include:

1. **Use case** - Why is this enhancement needed?
2. **Proposed solution** - How should it work?
3. **Alternatives** - Other approaches considered
4. **Impact** - Who benefits from this enhancement?

### Submitting Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link related issues

---

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public functions

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Linting

We use **Flake8** for linting:

```bash
# Run linter
flake8 src/ tests/

# Configuration in setup.cfg or pyproject.toml
```

### Type Checking

We use **MyPy** for static type checking:

```bash
# Run type checker
mypy src/

# Configuration in pyproject.toml
```

### Example Code Style

```python
"""Module for analyzing material stability.

This module provides tools for analyzing thermodynamic stability
of materials using DFT data from the Materials Project.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class StabilityResult:
    """Result of stability analysis.
    
    Attributes:
        material_id: Unique identifier for the material
        formula: Chemical formula
        energy_above_hull: Energy above convex hull (eV/atom)
        is_viable: Whether material is synthetically viable
    """
    material_id: str
    formula: str
    energy_above_hull: float
    is_viable: bool


class StabilityAnalyzer:
    """Analyzer for material thermodynamic stability.
    
    This class analyzes the thermodynamic stability of materials
    using energy above convex hull calculations.
    
    Args:
        metastable_threshold: Maximum energy above hull for viability (eV/atom)
        
    Example:
        >>> analyzer = StabilityAnalyzer(metastable_threshold=0.1)
        >>> result = analyzer.analyze_stability("mp-149", "SiC", 0.0, -0.65)
        >>> print(result.is_viable)
        True
    """
    
    def __init__(self, metastable_threshold: float = 0.1) -> None:
        """Initialize the stability analyzer.
        
        Args:
            metastable_threshold: Maximum ΔE_hull for viable materials (eV/atom)
        """
        self.metastable_threshold = metastable_threshold
        
    def analyze_stability(
        self,
        material_id: str,
        formula: str,
        energy_above_hull: float,
        formation_energy_per_atom: float
    ) -> StabilityResult:
        """Analyze stability of a material.
        
        Args:
            material_id: Unique identifier
            formula: Chemical formula
            energy_above_hull: Energy above convex hull (eV/atom)
            formation_energy_per_atom: Formation energy (eV/atom)
            
        Returns:
            StabilityResult containing analysis results
            
        Raises:
            ValueError: If energy_above_hull is negative
        """
        if energy_above_hull < 0:
            raise ValueError("Energy above hull cannot be negative")
            
        is_viable = energy_above_hull <= self.metastable_threshold
        
        return StabilityResult(
            material_id=material_id,
            formula=formula,
            energy_above_hull=energy_above_hull,
            is_viable=is_viable
        )
```

---

## Testing Guidelines

### Test Structure

Tests are organized by module:

```
tests/
├── test_dft/
│   ├── test_stability_analyzer.py
│   └── test_materials_project_client.py
├── test_ml/
│   ├── test_model_trainer.py
│   └── test_feature_engineering.py
└── conftest.py  # Shared fixtures
```

### Writing Tests

Use **pytest** for all tests:

```python
"""Tests for StabilityAnalyzer."""

import pytest
from ceramic_discovery.dft import StabilityAnalyzer, StabilityResult


class TestStabilityAnalyzer:
    """Test suite for StabilityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return StabilityAnalyzer(metastable_threshold=0.1)
    
    def test_stable_material(self, analyzer):
        """Test analysis of stable material (ΔE_hull = 0)."""
        result = analyzer.analyze_stability(
            material_id="mp-149",
            formula="SiC",
            energy_above_hull=0.0,
            formation_energy_per_atom=-0.65
        )
        
        assert result.is_viable is True
        assert result.energy_above_hull == 0.0
        
    def test_metastable_material(self, analyzer):
        """Test analysis of metastable material (0 < ΔE_hull ≤ 0.1)."""
        result = analyzer.analyze_stability(
            material_id="test-001",
            formula="Si0.98B0.02C",
            energy_above_hull=0.05,
            formation_energy_per_atom=-0.63
        )
        
        assert result.is_viable is True
        
    def test_unstable_material(self, analyzer):
        """Test analysis of unstable material (ΔE_hull > 0.1)."""
        result = analyzer.analyze_stability(
            material_id="test-002",
            formula="SiC2",
            energy_above_hull=0.5,
            formation_energy_per_atom=-0.3
        )
        
        assert result.is_viable is False
        
    def test_negative_energy_raises_error(self, analyzer):
        """Test that negative energy above hull raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            analyzer.analyze_stability(
                material_id="test-003",
                formula="SiC",
                energy_above_hull=-0.1,
                formation_energy_per_atom=-0.65
            )
```

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Target coverage**: 90%+ for critical modules
- **Run coverage**: `pytest --cov=ceramic_discovery --cov-report=html`

### Test Categories

1. **Unit tests** - Test individual functions/classes
2. **Integration tests** - Test module interactions
3. **System tests** - Test complete workflows
4. **Performance tests** - Test computational efficiency

### Fixtures

Use fixtures for common test data:

```python
# conftest.py
import pytest
from ceramic_discovery.ceramics import CeramicSystemFactory


@pytest.fixture
def sic_system():
    """Baseline SiC system for testing."""
    return CeramicSystemFactory.create_sic()


@pytest.fixture
def sample_dft_data():
    """Sample DFT data for testing."""
    return {
        "material_id": "mp-149",
        "formula": "SiC",
        "energy_above_hull": 0.0,
        "formation_energy_per_atom": -0.65,
        "band_gap": 2.4,
        "density": 3.21
    }
```

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def predict_v50(
    self,
    properties: Dict[str, float],
    return_uncertainty: bool = True
) -> Dict[str, float]:
    """Predict ballistic limit velocity (V₅₀).
    
    Predicts the V₅₀ ballistic performance metric from material
    properties using a trained machine learning model.
    
    Args:
        properties: Dictionary of material properties including:
            - hardness (GPa)
            - fracture_toughness (MPa·m^0.5)
            - density (g/cm³)
            - thermal_conductivity_1000C (W/(m·K))
        return_uncertainty: Whether to include uncertainty quantification
        
    Returns:
        Dictionary containing:
            - v50: Predicted V₅₀ (m/s)
            - confidence_interval: 95% CI as (lower, upper) tuple
            - reliability_score: Reliability score (0-1)
            
    Raises:
        ValueError: If required properties are missing
        ModelNotFoundError: If model not loaded
        
    Example:
        >>> predictor = BallisticPredictor()
        >>> predictor.load_model("model.pkl")
        >>> properties = {"hardness": 28.0, "fracture_toughness": 4.5, ...}
        >>> result = predictor.predict_v50(properties)
        >>> print(f"V₅₀: {result['v50']:.0f} m/s")
        V₅₀: 850 m/s
        
    Note:
        Predictions are most reliable for materials similar to the
        training data. Check reliability_score before using predictions.
    """
```

### Documentation Files

Update relevant documentation:

- **README.md** - If changing core functionality
- **docs/USER_GUIDE.md** - If adding user-facing features
- **docs/API_REFERENCE.md** - If changing API
- **docs/METHODOLOGY.md** - If changing scientific methodology

### Jupyter Notebooks

For tutorial notebooks:

1. **Clear outputs** before committing
2. **Add markdown cells** explaining each step
3. **Include visualizations** where helpful
4. **Test notebooks** end-to-end before committing

---

## Pull Request Process

### Before Submitting

Checklist before opening a PR:

- [ ] Code follows style guidelines
- [ ] All tests pass: `pytest tests/`
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings written
- [ ] No merge conflicts with main branch

### PR Template

Use this template for your PR description:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Related Issues
Fixes #123
Related to #456

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe testing performed:
- Unit tests added/updated
- Integration tests passed
- Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
Add screenshots for UI changes
```

### Review Process

1. **Automated checks** - CI/CD runs tests and linting
2. **Code review** - Maintainer reviews code
3. **Feedback** - Address review comments
4. **Approval** - Maintainer approves PR
5. **Merge** - PR merged to main branch

### After Merge

- Delete your feature branch
- Update your local repository
- Close related issues

---

## Release Process

### Versioning

We use Semantic Versioning (SemVer):

- **MAJOR** (1.0.0) - Incompatible API changes
- **MINOR** (0.1.0) - New features, backwards compatible
- **PATCH** (0.0.1) - Bug fixes, backwards compatible

### Release Checklist

1. Update version in `setup.py` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build documentation
5. Create release tag
6. Publish to PyPI (if applicable)
7. Create GitHub release with notes

---

## Development Workflow

### Typical Workflow

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Run tests
pytest tests/
black src/ tests/
flake8 src/ tests/

# 5. Commit changes
git add .
git commit -m "Add feature: description"

# 6. Push to fork
git push origin feature/my-feature

# 7. Open PR on GitHub
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

---

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with your question
4. Join our community discussions

Thank you for contributing to the Ceramic Armor Discovery Framework! 🎉
