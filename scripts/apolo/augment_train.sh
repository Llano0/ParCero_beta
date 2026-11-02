#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

python -I src/data/augment_train.py --raw data/raw --out data/aug "$@"
