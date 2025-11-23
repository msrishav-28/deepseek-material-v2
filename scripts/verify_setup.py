#!/usr/bin/env python
"""Verification script for Task 1 completion."""

import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False


def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False


def main():
    """Run verification checks."""
    print("=" * 70)
    print("Task 1 Verification: Project Setup and Research Infrastructure")
    print("=" * 70)
    print()

    checks = []

    # Core configuration files
    print("Core Configuration Files:")
    checks.append(check_file_exists("environment.yml", "Conda environment"))
    checks.append(check_file_exists("Dockerfile", "Docker container"))
    checks.append(check_file_exists("docker-compose.yml", "Docker Compose"))
    checks.append(check_file_exists(".env.example", "Environment template"))
    checks.append(check_file_exists("setup.py", "Package setup"))
    checks.append(check_file_exists("pyproject.toml", "Project config"))
    print()

    # Source code structure
    print("Source Code Structure:")
    checks.append(check_file_exists("src/ceramic_discovery/__init__.py", "Main package"))
    checks.append(check_file_exists("src/ceramic_discovery/config.py", "Configuration"))
    checks.append(check_file_exists("src/ceramic_discovery/cli.py", "CLI"))
    checks.append(check_directory_exists("src/ceramic_discovery/dft", "DFT module"))
    checks.append(check_directory_exists("src/ceramic_discovery/ml", "ML module"))
    checks.append(check_directory_exists("src/ceramic_discovery/ceramics", "Ceramics module"))
    checks.append(check_directory_exists("src/ceramic_discovery/screening", "Screening module"))
    checks.append(check_directory_exists("src/ceramic_discovery/validation", "Validation module"))
    checks.append(check_directory_exists("src/ceramic_discovery/utils", "Utils module"))
    print()

    # Database and scripts
    print("Database and Scripts:")
    checks.append(check_file_exists("scripts/init_db.sql", "Database init script"))
    checks.append(check_file_exists("scripts/submit_slurm.sh", "HPC submission script"))
    checks.append(check_file_exists("scripts/create_conda_lock.sh", "Conda lock script"))
    print()

    # Jupyter notebooks
    print("Jupyter Notebooks:")
    checks.append(check_file_exists("notebooks/01_exploratory_analysis.ipynb", "Exploratory analysis"))
    checks.append(check_file_exists("notebooks/02_ml_model_training.ipynb", "ML training"))
    print()

    # Testing infrastructure
    print("Testing Infrastructure:")
    checks.append(check_file_exists("tests/__init__.py", "Test package"))
    checks.append(check_file_exists("tests/conftest.py", "Test fixtures"))
    checks.append(check_file_exists("tests/test_config.py", "Config tests"))
    print()

    # Data directories
    print("Data Directories:")
    checks.append(check_directory_exists("data/hdf5", "HDF5 storage"))
    checks.append(check_directory_exists("results", "Results directory"))
    checks.append(check_directory_exists("logs", "Logs directory"))
    print()

    # Documentation
    print("Documentation:")
    checks.append(check_file_exists("README.md", "Main README"))
    checks.append(check_file_exists("QUICKSTART.md", "Quick start guide"))
    checks.append(check_file_exists("PROJECT_STATUS.md", "Project status"))
    print()

    # Summary
    print("=" * 70)
    total = len(checks)
    passed = sum(checks)
    print(f"Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ Task 1 is COMPLETE - All infrastructure is in place!")
        return 0
    else:
        print(f"✗ Task 1 is INCOMPLETE - {total - passed} items missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
