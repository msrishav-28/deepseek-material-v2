-- Initialize Ceramic Armor Discovery Database
-- This script sets up the PostgreSQL database with pgvector extension

-- Enable pgvector extension for material embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Materials table: Core material compositions
CREATE TABLE IF NOT EXISTS materials (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) UNIQUE NOT NULL,
    base_composition VARCHAR(50) NOT NULL,
    dopant_element VARCHAR(10),
    dopant_concentration FLOAT,
    composite_ratio_1 FLOAT,
    composite_ratio_2 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DFT calculations table
CREATE TABLE IF NOT EXISTS dft_calculations (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) REFERENCES materials(material_id),
    energy_above_hull FLOAT NOT NULL,
    formation_energy FLOAT,
    phase_stability VARCHAR(20),
    confidence FLOAT,
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    software_version VARCHAR(50),
    functional VARCHAR(50),
    convergence_criteria JSONB
);

-- Material properties table (58 properties with uncertainties)
CREATE TABLE IF NOT EXISTS material_properties (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) REFERENCES materials(material_id),
    property_name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    uncertainty FLOAT,
    unit VARCHAR(50),
    source VARCHAR(50),
    quality_score FLOAT,
    temperature_c FLOAT,
    measurement_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML models table
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),
    r2_score FLOAT,
    r2_confidence_lower FLOAT,
    r2_confidence_upper FLOAT,
    feature_importance JSONB,
    hyperparameters JSONB,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    random_seed INTEGER,
    model_file_path VARCHAR(255)
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) REFERENCES materials(material_id),
    model_id INTEGER REFERENCES ml_models(id),
    predicted_value FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    reliability_score FLOAT,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow runs table for reproducibility
CREATE TABLE IF NOT EXISTS workflow_runs (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(100) UNIQUE NOT NULL,
    workflow_type VARCHAR(50),
    parameters JSONB,
    random_seed INTEGER,
    software_versions JSONB,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),
    results_summary JSONB
);

-- Experimental data table for validation
CREATE TABLE IF NOT EXISTS experimental_data (
    id SERIAL PRIMARY KEY,
    material_composition VARCHAR(100),
    property_name VARCHAR(100),
    value FLOAT,
    uncertainty FLOAT,
    unit VARCHAR(50),
    source VARCHAR(255),
    doi VARCHAR(100),
    measurement_conditions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Material embeddings table (for similarity search)
CREATE TABLE IF NOT EXISTS material_embeddings (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) REFERENCES materials(material_id),
    embedding vector(512),
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_materials_base_composition ON materials(base_composition);
CREATE INDEX IF NOT EXISTS idx_materials_dopant ON materials(dopant_element);
CREATE INDEX IF NOT EXISTS idx_dft_energy_hull ON dft_calculations(energy_above_hull);
CREATE INDEX IF NOT EXISTS idx_properties_material ON material_properties(material_id);
CREATE INDEX IF NOT EXISTS idx_properties_name ON material_properties(property_name);
CREATE INDEX IF NOT EXISTS idx_predictions_material ON predictions(material_id);
CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_runs(status);

-- Create vector similarity search index
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON material_embeddings 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ceramic_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ceramic_user;
