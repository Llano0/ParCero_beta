#!/bin/bash
#SBATCH --job-name=cnn-base
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Runs the notebook top to bottom (the 10 trainings included, one after another) and
# saves the outputs inside it. Finished runs in dl/outputs/cnn/ are reused, so it can be
# submitted again after a cut. Time per epoch is unknown until the first run: adjust
# --time from the command line if needed (sbatch --time=HH:MM:SS dl/scripts/run_cnn_base.sh).
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_base.ipynb
