"""PostgreSQL database schema for material properties."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Material(Base):
    """Core material table."""

    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(String(50), unique=True, nullable=False, index=True)
    formula = Column(String(200), nullable=False, index=True)
    formula_pretty = Column(String(200))

    # Composition
    base_system = Column(String(50), index=True)  # e.g., 'SiC', 'B4C'
    dopant_element = Column(String(10))
    dopant_concentration = Column(Float)  # atomic fraction
    is_composite = Column(Boolean, default=False)
    composite_ratio = Column(JSON)  # For composite systems

    # Metadata
    source = Column(String(50), default="Materials Project")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)

    # Relationships
    properties = relationship("MaterialProperty", back_populates="material", cascade="all, delete-orphan")
    dft_calculations = relationship("DFTCalculation", back_populates="material", cascade="all, delete-orphan")


class MaterialProperty(Base):
    """Material properties with uncertainty quantification."""

    __tablename__ = "material_properties"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False, index=True)

    # Property identification
    property_name = Column(String(100), nullable=False, index=True)
    property_category = Column(String(50), index=True)  # structural, mechanical, thermal, etc.
    property_tier = Column(Integer)  # 1, 2, or 3

    # Property value
    value = Column(Float)
    unit = Column(String(50), nullable=False)
    unit_standardized = Column(String(50))  # Standardized unit

    # Uncertainty quantification
    uncertainty = Column(Float)
    uncertainty_type = Column(String(50))  # 'standard_deviation', 'confidence_interval', etc.

    # Data quality
    source = Column(String(50), nullable=False)  # 'dft', 'experimental', 'predicted', 'calculated'
    quality_score = Column(Float)  # 0-1 scale
    is_derived = Column(Boolean, default=False)  # True if calculated from other properties

    # Provenance
    calculation_method = Column(String(200))
    reference = Column(Text)
    notes = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)

    # Relationships
    material = relationship("Material", back_populates="properties")


class DFTCalculation(Base):
    """DFT calculation results and metadata."""

    __tablename__ = "dft_calculations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False, index=True)

    # Thermodynamic properties
    formation_energy_per_atom = Column(Float, nullable=False)
    energy_above_hull = Column(Float, nullable=False, index=True)
    decomposition_energy = Column(Float)
    is_stable = Column(Boolean, index=True)

    # Electronic properties
    band_gap = Column(Float)
    is_metal = Column(Boolean)
    is_magnetic = Column(Boolean)
    total_magnetization = Column(Float)
    efermi = Column(Float)

    # Structural properties
    crystal_structure = Column(JSON)  # Full structure data
    space_group_number = Column(Integer)
    nsites = Column(Integer)
    volume = Column(Float)
    density = Column(Float)

    # Elastic properties (if available)
    bulk_modulus_vrh = Column(Float)
    shear_modulus_vrh = Column(Float)
    elastic_tensor = Column(JSON)

    # Calculation metadata
    functional = Column(String(50))  # e.g., 'PBE', 'HSE06'
    pseudopotential = Column(String(100))
    cutoff_energy = Column(Float)
    k_point_density = Column(Float)

    # Provenance
    calculation_id = Column(String(100))
    source_database = Column(String(50), default="Materials Project")
    retrieved_at = Column(DateTime)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    material = relationship("Material", back_populates="dft_calculations")


class PropertyProvenance(Base):
    """Track provenance and version history of properties."""

    __tablename__ = "property_provenance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    property_id = Column(Integer, ForeignKey("material_properties.id"), nullable=False)

    # Version tracking
    version = Column(Integer, nullable=False)
    previous_value = Column(Float)
    new_value = Column(Float)
    change_reason = Column(Text)

    # Provenance
    changed_by = Column(String(100))
    changed_at = Column(DateTime, default=datetime.utcnow)
    source_reference = Column(Text)

    # Computational details
    calculation_parameters = Column(JSON)
    software_versions = Column(JSON)


class ExperimentalData(Base):
    """Experimental validation data from literature."""

    __tablename__ = "experimental_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_formula = Column(String(200), nullable=False, index=True)

    # Property measurement
    property_name = Column(String(100), nullable=False)
    measured_value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    uncertainty = Column(Float)

    # Experimental conditions
    temperature = Column(Float)  # K
    pressure = Column(Float)  # GPa
    synthesis_method = Column(String(200))
    sample_preparation = Column(Text)

    # Literature reference
    doi = Column(String(100))
    citation = Column(Text, nullable=False)
    year = Column(Integer)
    authors = Column(Text)

    # Data quality
    quality_score = Column(Float)  # 0-1 scale
    reliability_notes = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class ScreeningResult(Base):
    """Results from high-throughput screening workflows."""

    __tablename__ = "screening_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(100), nullable=False, index=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False)

    # Screening criteria
    passes_stability_filter = Column(Boolean, index=True)
    passes_performance_filter = Column(Boolean)
    passes_cost_filter = Column(Boolean)

    # Rankings
    overall_rank = Column(Integer)
    stability_score = Column(Float)
    performance_score = Column(Float)
    cost_score = Column(Float)

    # Predictions
    predicted_v50 = Column(Float)
    prediction_confidence = Column(Float)
    prediction_uncertainty = Column(Float)

    # Workflow metadata
    screening_date = Column(DateTime, default=datetime.utcnow)
    ml_model_version = Column(String(50))
    parameters = Column(JSON)


def create_database_schema(database_url: str) -> None:
    """
    Create all database tables.

    Args:
        database_url: PostgreSQL connection URL
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)


def get_session(database_url: str):
    """
    Get database session.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        SQLAlchemy session
    """
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def drop_all_tables(database_url: str) -> None:
    """
    Drop all database tables (use with caution!).

    Args:
        database_url: PostgreSQL connection URL
    """
    engine = create_engine(database_url)
    Base.metadata.drop_all(engine)
