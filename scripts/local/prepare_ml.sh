#!/bin/bash
# Local (laptop) version of scripts/apolo/prepare_ml.sh: everything model_development.ipynb needs.
# Run from the repo root after downloading Chula (README step 1). No GPU or internet needed once the
# DINOv2 weights are in the torch.hub cache. Steps whose output already exists are skipped.
#   bash scripts/local/prepare_ml.sh
set -e
PY=${PY:-python}
WEIGHTS=${WEIGHTS:-artifacts/yolo/h_single_aug/weights/best.pt}
MODEL=dinov2_vitb14

[ -f "$WEIGHTS" ] || { echo "missing detector weights: $WEIGHTS"; exit 1; }
[ -f data/raw/data.yaml ] || $PY src/data/filter_classes.py --src data/roboflow --dst data/raw

# 1. Crops from the hand-drawn boxes and their embeddings (the classifiers learn from these)
[ -f data/crops/crops.csv ] || $PY src/data/crop_eggs.py --raw data/raw --out data/crops --margin 0.15
[ -f artifacts/embeddings/${MODEL}_224.npz ] || $PY src/embeddings/extract_embeddings.py \
  --crops-csv data/crops/crops.csv --model $MODEL --out-dir artifacts/embeddings

# 2. Crops from the detector: valid (model selection, rejection thresholds) and test (final evaluation only)
for SPLIT in valid test; do
  FINAL=""; [ "$SPLIT" = test ] && FINAL="--final"
  [ -f artifacts/detected/$SPLIT/crops.csv ] || $PY src/detection/detect_crops.py --weights "$WEIGHTS" \
    --images data/raw/$SPLIT/images --raw data/raw --split $SPLIT --out artifacts/detected/$SPLIT --conf 0.10 $FINAL
  [ -f artifacts/embeddings_detected/$SPLIT/${MODEL}_224.npz ] || $PY src/embeddings/extract_embeddings.py \
    --crops-csv artifacts/detected/$SPLIT/crops.csv --model $MODEL --out-dir artifacts/embeddings_detected/$SPLIT
done
echo "Ready: jupyter nbconvert --to notebook --execute --inplace notebooks/model_development.ipynb"
