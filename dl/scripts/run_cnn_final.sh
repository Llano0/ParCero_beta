#!/bin/bash
#SBATCH --job-name=cnn-final
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_final.ipynb
