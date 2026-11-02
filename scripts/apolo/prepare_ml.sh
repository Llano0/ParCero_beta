#!/bin/bash
#SBATCH --job-name=prep-ml
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/%x-%j.out

set -e
module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Detector used to cut the crops; override with: sbatch --export=ALL,WEIGHTS=... prepare_ml.sh
WEIGHTS=${WEIGHTS:-artifacts/yolo/h_single_aug/weights/best.pt}
MODEL=dinov2_vitb14

# 1. Crops from the hand-drawn boxes: the classifiers are trained on these (train split)
python src/data/crop_eggs.py --raw data/raw --out data/crops --margin 0.15
python src/embeddings/extract_embeddings.py --crops-csv data/crops/crops.csv \
  --model $MODEL --out-dir artifacts/embeddings

# 2. Crops from the detector on valid: measures what the detector's errors cost the classifier
python src/detection/detect_crops.py --weights "$WEIGHTS" --images data/raw/valid/images \
  --raw data/raw --split valid --out artifacts/detected/valid --conf 0.10
python src/embeddings/extract_embeddings.py --crops-csv artifacts/detected/valid/crops.csv \
  --model $MODEL --out-dir artifacts/embeddings_detected/valid

# 3. Crops from the external photos (no ground truth)
python src/detection/detect_crops.py --weights "$WEIGHTS" --images data/external \
  --split external --out artifacts/detected/external --conf 0.10
python src/embeddings/extract_embeddings.py --crops-csv artifacts/detected/external/crops.csv \
  --model $MODEL --out-dir artifacts/embeddings_detected/external
