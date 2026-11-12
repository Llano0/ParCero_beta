#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Compares the CNN base, VGG16 and EfficientNet-B0 (and YOLO 3-class as baseline) with the saved
# weights: nothing is retrained. Needs dl/outputs/{cnn_final,transfer_vgg16,transfer_efficientnet_b0,yolo}/.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/comparison.ipynb
