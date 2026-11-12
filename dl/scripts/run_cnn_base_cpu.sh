#!/bin/bash
#SBATCH --job-name=cnn-base-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Same notebook as run_cnn_base.sh, but on bigmem (no GPU requested): the notebook falls
# back to CPU on its own. Finished runs in dl/outputs/cnn/ are reused, so it can be
# submitted again after a cut. Time per epoch is unknown until the first run: adjust
# --time from the command line if needed.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_base.ipynb
