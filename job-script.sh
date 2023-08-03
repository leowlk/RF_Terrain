#!/bin/bash

#SBATCH --job-name="RF_terrain"
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=compute
#SBATCH --account=Education-ABE-MSc-G

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /scratch/lwlkan/.conda/RF_Terrain
srun python main.py > rf_job_log.log

conda deactivate