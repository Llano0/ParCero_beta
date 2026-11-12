#!/bin/bash
#SBATCH --job-name=yolo-3class
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Runs the notebook top to bottom (training included) and saves the outputs inside it.
# If a cell fails the job stops and the notebook on disk is left untouched.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/yolo_3class.ipynb
