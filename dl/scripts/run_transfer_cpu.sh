#!/bin/bash
#SBATCH --job-name=transfer-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Same as run_transfer.sh but on CPU, for when the GPU queue is busy. VGG16 is heavy on CPU:
# the light network is the realistic choice here
#   sbatch --export=ALL,ARCH=efficientnet_b0 dl/scripts/run_transfer_cpu.sh
# With TRAINING = False in the notebook it only reads the saved runs and redoes the analysis.
# ImageNet weights must already be in ~/.cache/torch/hub/checkpoints/ (see run_transfer.sh).
# VGG16 runs dl/notebooks/transfer_learning.ipynb itself. Any other ARCH runs a fresh copy,
# dl/notebooks/transfer_learning_<arch>.ipynb (regenerated from the VGG16 one on every run),
# so both networks can train at the same time, one GPU each, without overwriting each other:
#   sbatch --export=ALL,ARCH=efficientnet_b0 dl/scripts/run_transfer_cpu.sh
export ARCH=${ARCH:-vgg16}
NOTEBOOK=dl/notebooks/transfer_learning.ipynb
if [ "$ARCH" != vgg16 ]; then
  cp "$NOTEBOOK" "dl/notebooks/transfer_learning_${ARCH}.ipynb"
  NOTEBOOK="dl/notebooks/transfer_learning_${ARCH}.ipynb"
fi
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  "$NOTEBOOK"
