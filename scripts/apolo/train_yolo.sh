#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

yolo detect train model=artifacts/weights/yolo11n.pt data=data/raw/data.yaml \
  imgsz=640 batch=32 seed=42 deterministic=True workers=8 \
  fliplr=0.5 flipud=0.5 degrees=0 \
  project="$PWD/artifacts/yolo" "$@"
