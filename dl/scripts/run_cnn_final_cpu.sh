#!/bin/bash
#SBATCH --job-name=cnn-final-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# With TRAINING = False (the default) the notebook only reads the saved results and runs
# inference (valid, robustness, external photos), so one hour is plenty. If you set
# TRAINING = True to train more seeds, raise --time (about an hour per seed on CPU in
# cnn_combos.ipynb): sbatch --time=04:00:00 dl/scripts/run_cnn_final_cpu.sh
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_final.ipynb
