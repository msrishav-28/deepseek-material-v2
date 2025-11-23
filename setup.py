from setuptools import setup, find_packages

setup(
    name="ceramic-armor-discovery",
    version="0.1.0",
    description="Computational framework for ceramic armor materials discovery using DFT and ML",
    author="Research Team",
    author_email="research@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "pandas>=2.0.3",
        "ase>=3.22.1",
        "pymatgen>=2023.7.20",
        "mp-api>=0.37.5",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "optuna>=3.3.0",
        "h5py>=3.9.0",
        "sqlalchemy>=2.0.19",
        "psycopg2>=2.9.7",
        "pgvector>=0.2.2",
        "prefect>=2.11.5",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.16.1",
        "click>=8.1.7",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
        "redis>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "ceramic-discovery=ceramic_discovery.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.11",
    ],
)
