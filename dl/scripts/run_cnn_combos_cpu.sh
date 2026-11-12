#!/bin/bash
#SBATCH --job-name=cnn-combos-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Same notebook as run_cnn_combos.sh, but on bigmem (no GPU requested): the notebook falls
# back to CPU on its own. Round 1 took 17-31 min per run at 25 epochs; 40 epochs and the
# 5-block nets will take longer. Finished runs are reused, so it can be submitted again.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/cnn_combos.ipynb
