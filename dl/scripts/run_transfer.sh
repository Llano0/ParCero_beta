#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Fine-tunes the ImageNet network (VGG16 by default) in two phases for 3 seeds (42, 43, 44),
# one after another, and then runs the whole analysis, including the comparison with the
# CNN base of dl/outputs/cnn_final/. Each finished seed is saved in
# dl/outputs/transfer_<arch>/seed<N>/ and reused, so after a cut you can submit it again.
#
# BEFORE THE FIRST RUN, on the login node (compute nodes have no internet), download the
# ImageNet weights into ~/.cache/torch/hub/checkpoints/:
#   python -c "from torchvision import models; models.vgg16(weights='IMAGENET1K_V1'); models.efficientnet_b0(weights='IMAGENET1K_V1')"
#
# VGG16 runs dl/notebooks/transfer_learning.ipynb itself. Any other ARCH runs a fresh copy,
# dl/notebooks/transfer_learning_<arch>.ipynb (regenerated from the VGG16 one on every run),
# so both networks can train at the same time, one GPU each, without overwriting each other:
#   sbatch --export=ALL,ARCH=efficientnet_b0 dl/scripts/run_transfer.sh
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
