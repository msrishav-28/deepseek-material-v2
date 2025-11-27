"""
Ceramic Armor Discovery Framework

A computational materials science platform for high-throughput screening
of ceramic armor materials using DFT and machine learning.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from ceramic_discovery.config import (
    Config,
    DataSourcesConfig,
    FeatureEngineeringConfig,
    ApplicationRankingConfig,
    ExperimentalPlanningConfig,
)

__all__ = [
    "Config",
    "DataSourcesConfig",
    "FeatureEngineeringConfig",
    "ApplicationRankingConfig",
    "ExperimentalPlanningConfig",
]
