#!/bin/bash
#SBATCH --job-name=ext-csv
#SBATCH --partition=accel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/data_prep/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# No GPU is requested (the partition is used only because it is the one with free nodes):
# the script just copies a few photos and writes a CSV.
#   sbatch data_prep/scripts/build_external_csv.sh            -> dry run, changes nothing
#   sbatch data_prep/scripts/build_external_csv.sh --apply    -> copies the photos, writes the CSV
python -I data_prep/src/build_external_csv.py "$@"
