#!/bin/bash
# Create conda-lock file for exact reproducibility

echo "Creating conda-lock file for reproducibility..."

# Install conda-lock if not present
conda install -c conda-forge conda-lock -y

# Generate lock file for multiple platforms
conda-lock -f environment.yml -p linux-64 -p osx-64 -p win-64

echo "✓ conda-lock.yml created"
echo "To create environment from lock file:"
echo "  conda-lock install -n ceramic-armor-discovery conda-lock.yml"
