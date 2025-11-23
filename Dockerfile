FROM continuumio/miniconda3:23.5.2-0

# Set working directory
WORKDIR /opt/ceramic-armor-discovery

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate environment in shell
SHELL ["conda", "run", "-n", "ceramic-armor-discovery", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Install project in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONDA_DEFAULT_ENV=ceramic-armor-discovery

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type liveness --json || exit 1

# Expose ports
EXPOSE 8888

# Default command
CMD ["conda", "run", "-n", "ceramic-armor-discovery", "python", "-m", "ceramic_discovery.cli"]
