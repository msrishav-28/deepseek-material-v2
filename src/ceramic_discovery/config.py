"""Configuration management for the Ceramic Armor Discovery Framework."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://ceramic_user:ceramic_dev_password@localhost:5432/ceramic_materials",
        )
    )
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    """Redis cache configuration."""

    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ttl_seconds: int = 3600


@dataclass
class MaterialsProjectConfig:
    """Materials Project API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("MATERIALS_PROJECT_API_KEY", ""))
    rate_limit_per_second: int = 5
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class HDF5Config:
    """HDF5 storage configuration."""

    data_path: Path = field(
        default_factory=lambda: Path(os.getenv("HDF5_DATA_PATH", "./data/hdf5"))
    )
    compression: str = "gzip"
    compression_level: int = 4


@dataclass
class ComputationalConfig:
    """Computational settings."""

    max_parallel_jobs: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARALLEL_JOBS", "4"))
    )
    dft_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("DFT_TIMEOUT_SECONDS", "3600"))
    )
    random_seed: int = field(default_factory=lambda: int(os.getenv("ML_RANDOM_SEED", "42")))


@dataclass
class MLConfig:
    """Machine learning configuration."""

    random_seed: int = field(default_factory=lambda: int(os.getenv("ML_RANDOM_SEED", "42")))
    cross_validation_folds: int = field(
        default_factory=lambda: int(os.getenv("ML_CROSS_VALIDATION_FOLDS", "5"))
    )
    target_r2_min: float = 0.65
    target_r2_max: float = 0.75
    bootstrap_samples: int = 1000


@dataclass
class HPCConfig:
    """HPC cluster configuration."""

    scheduler: str = field(default_factory=lambda: os.getenv("HPC_SCHEDULER", "slurm"))
    partition: str = field(default_factory=lambda: os.getenv("HPC_PARTITION", "compute"))
    max_nodes: int = field(default_factory=lambda: int(os.getenv("HPC_MAX_NODES", "4")))


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_FILE", "./logs/ceramic_discovery.log"))
    )


@dataclass
class Config:
    """Main configuration class."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    materials_project: MaterialsProjectConfig = field(default_factory=MaterialsProjectConfig)
    hdf5: HDF5Config = field(default_factory=HDF5Config)
    computational: ComputationalConfig = field(default_factory=ComputationalConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Create necessary directories."""
        self.hdf5.data_path.mkdir(parents=True, exist_ok=True)
        self.logging.file.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration."""
        if not self.materials_project.api_key:
            raise ValueError("MATERIALS_PROJECT_API_KEY is required")

        if not self.database.url:
            raise ValueError("DATABASE_URL is required")


# Global configuration instance
config = Config()
