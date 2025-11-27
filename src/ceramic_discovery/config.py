"""Configuration management for the Ceramic Armor Discovery Framework."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
class JarvisConfig:
    """JARVIS-DFT data source configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("JARVIS_ENABLED", "true").lower() == "true")
    file_path: Path = field(
        default_factory=lambda: Path(os.getenv("JARVIS_FILE_PATH", "./data/jdft_3d-7-7-2018.json"))
    )
    metal_elements: Set[str] = field(
        default_factory=lambda: {
            "Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr", "B", "Al"
        }
    )


@dataclass
class NISTConfig:
    """NIST-JANAF data source configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("NIST_ENABLED", "true").lower() == "true")
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("NIST_DATA_DIR", "./data/nist"))
    )
    default_temperature_K: float = 298.15


@dataclass
class LiteratureConfig:
    """Literature database configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("LITERATURE_ENABLED", "true").lower() == "true")


@dataclass
class DataSourcesConfig:
    """Multi-source data integration configuration."""

    jarvis: JarvisConfig = field(default_factory=JarvisConfig)
    nist: NISTConfig = field(default_factory=NISTConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    
    # Data combination settings
    source_priority: List[str] = field(
        default_factory=lambda: ["literature", "materials_project", "jarvis", "nist"]
    )
    enable_deduplication: bool = True
    conflict_resolution_strategy: str = "priority"  # "priority", "average", "newest"


@dataclass
class CompositionDescriptorsConfig:
    """Composition descriptor calculation configuration."""

    enabled: bool = True
    calculate_entropy: bool = True
    calculate_electronegativity: bool = True
    calculate_atomic_properties: bool = True


@dataclass
class StructureDescriptorsConfig:
    """Structure descriptor calculation configuration."""

    enabled: bool = True
    calculate_pugh_ratio: bool = True
    calculate_elastic_properties: bool = True
    calculate_energy_density: bool = True
    validate_physical_bounds: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""

    composition_descriptors: CompositionDescriptorsConfig = field(
        default_factory=CompositionDescriptorsConfig
    )
    structure_descriptors: StructureDescriptorsConfig = field(
        default_factory=StructureDescriptorsConfig
    )
    
    # Feature selection
    min_feature_variance: float = 0.01
    remove_correlated_features: bool = False
    correlation_threshold: float = 0.95


@dataclass
class ApplicationRankingConfig:
    """Application-specific ranking configuration."""

    enabled: bool = True
    
    # Applications to use for ranking
    applications: List[str] = field(
        default_factory=lambda: [
            "aerospace_hypersonic",
            "cutting_tools",
            "thermal_barriers",
            "wear_resistant",
            "electronic"
        ]
    )
    
    # Scoring settings
    allow_partial_scoring: bool = True
    min_properties_for_ranking: int = 2
    confidence_threshold: float = 0.5


@dataclass
class SynthesisConfig:
    """Synthesis method configuration."""

    default_methods: List[str] = field(
        default_factory=lambda: [
            "Hot Pressing",
            "Solid State Sintering",
            "Spark Plasma Sintering"
        ]
    )
    
    # Cost and time multipliers
    cost_multiplier: float = 1.0
    time_multiplier: float = 1.0


@dataclass
class CharacterizationConfig:
    """Characterization technique configuration."""

    default_techniques: List[str] = field(
        default_factory=lambda: [
            "XRD",
            "SEM",
            "TEM",
            "Hardness Testing",
            "Thermal Conductivity",
            "Density Measurement"
        ]
    )
    
    # Cost and time multipliers
    cost_multiplier: float = 1.0
    time_multiplier: float = 1.0


@dataclass
class ExperimentalPlanningConfig:
    """Experimental validation planning configuration."""

    enabled: bool = True
    
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    characterization: CharacterizationConfig = field(default_factory=CharacterizationConfig)
    
    # Resource estimation settings
    default_timeline_months: float = 6.0
    default_cost_k_dollars: float = 50.0
    confidence_level: float = 0.8


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
    
    # SiC Alloy Designer Integration configurations
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    application_ranking: ApplicationRankingConfig = field(default_factory=ApplicationRankingConfig)
    experimental_planning: ExperimentalPlanningConfig = field(default_factory=ExperimentalPlanningConfig)

    def __post_init__(self) -> None:
        """Create necessary directories."""
        self.hdf5.data_path.mkdir(parents=True, exist_ok=True)
        self.logging.file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create data source directories if enabled
        if self.data_sources.nist.enabled:
            self.data_sources.nist.data_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration."""
        if not self.materials_project.api_key:
            raise ValueError("MATERIALS_PROJECT_API_KEY is required")

        if not self.database.url:
            raise ValueError("DATABASE_URL is required")
        
        # Validate data sources
        if self.data_sources.jarvis.enabled:
            if not self.data_sources.jarvis.file_path.exists():
                logger.warning(
                    f"JARVIS file not found: {self.data_sources.jarvis.file_path}. "
                    "JARVIS data loading will be skipped."
                )
        
        # Validate application ranking
        if self.application_ranking.enabled:
            if not self.application_ranking.applications:
                raise ValueError("At least one application must be specified for ranking")
        
        # Validate feature engineering
        if self.feature_engineering.correlation_threshold < 0 or self.feature_engineering.correlation_threshold > 1:
            raise ValueError("Correlation threshold must be between 0 and 1")


# Global configuration instance
config = Config()
