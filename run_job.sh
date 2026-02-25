#!/bin/bash
# PBS Batch Script for Habitat Project
# Usage: qsub run_job.sh

#PBS -N Habitat_Batch_Job
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=04:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o habitat_run.log

# 1. Load Modules (Adjust based on your HPC environment)
module load anaconda3/2023.03 2>/dev/null || echo "Module load failed or not needed"
module load cuda/11.8 2>/dev/null || echo "CUDA load failed or not needed"

# 2. Activate Environment
# Assuming 'habitat' is your conda env name
source activate habitat 2>/dev/null || conda activate habitat

# 3. Environment Variables
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# Force Headless Mode for Habitat
export DISPLAY=:0
export MAGNUM_TARGET_HEADLESS=1

# 4. Run the Script
echo "Starting Habitat Simulation at $(date)"
echo "Running on host: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Avoid conflicts with user installed packages
export PYTHONNOUSERSITE=1

# Run the driver script
python driver.py

echo "Job finished at $(date)"
