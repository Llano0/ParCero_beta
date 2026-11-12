#!/bin/bash
#SBATCH --job-name=yolo-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/detector/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Same settings as train_yolo.sh, but on CPU (no GPU requested). Anything passed
# after the script name is appended last and overrides the defaults below.
yolo detect train model=detector/outputs/weights/yolo11n.pt data=data/raw/data.yaml \
  device=cpu imgsz=640 batch=32 seed=42 deterministic=True workers=16 \
  fliplr=0.5 flipud=0.5 degrees=0 \
  project="$PWD/detector/outputs/yolo" "$@"
