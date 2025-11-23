#!/bin/bash
#SBATCH --job-name=ceramic_screening
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@example.com

# Load required modules
module load python/3.11
module load postgresql/14

# Activate conda environment
source activate ceramic-armor-discovery

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Run screening workflow
echo "Starting ceramic armor screening workflow"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

python -m ceramic_discovery.screening \
    --materials materials.json \
    --output results/run_${SLURM_JOB_ID} \
    --parallel-jobs ${SLURM_NTASKS}

echo "Workflow completed"
