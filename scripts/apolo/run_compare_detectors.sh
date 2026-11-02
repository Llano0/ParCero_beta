#!/bin/bash
#SBATCH --job-name=compare-det
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Runs the notebook top to bottom and saves the outputs inside it (like the EDA).
# If a cell fails the job stops and the notebook on disk is left untouched.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/compare_detectors.ipynb
