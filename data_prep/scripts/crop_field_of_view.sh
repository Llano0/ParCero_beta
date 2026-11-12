#!/bin/bash
#SBATCH --job-name=ext-fov
#SBATCH --partition=accel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/data_prep/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# No GPU is requested (the partition is used only because it is the one with free nodes).
# Crops the circular field of view of the external photos into data/external_fov/ (the originals in
# data/external/ are not touched) and writes data/external_fov_preview.png to check the result by eye.
#   sbatch data_prep/scripts/crop_field_of_view.sh
#   sbatch data_prep/scripts/crop_field_of_view.sh --limit 6      -> quick test on 6 photos
python -I data_prep/src/crop_field_of_view.py --input data/external --output data/external_fov --preview "$@"
