#!/bin/bash
#SBATCH --job-name=cnn-combos
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Round 2 of the CNN search (5 combinations, 40 epochs each, one after another).
# Finished runs in dl/outputs/cnn_combos/ are reused, so it can be submitted again after a cut.
# Time per epoch on GPU is unknown: adjust --time from the command line if needed.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_combos.ipynb
