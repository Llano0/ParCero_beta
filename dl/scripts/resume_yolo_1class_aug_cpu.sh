#!/bin/bash
#SBATCH --job-name=yolo-1class-resume
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Resumes the yolo_1class_aug training that was cut by the time limit (it stopped after
# epoch 72 of 100). Ultralytics reads the settings, the optimizer state and the epoch from
# last.pt and keeps writing in the same folder and the same results.csv.
LAST=dl/outputs/yolo/yolo_1class_aug/weights/last.pt
if [[ ! -f "$LAST" ]]; then
    echo "missing $LAST: nothing to resume" >&2
    exit 1
fi

yolo detect train resume model="$LAST"
