#!/bin/bash
#SBATCH --job-name=yolo-1class-cpu
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH --output=logs/dl/%x-%j.out

module load miniconda3/25.5.1
source ~/.bashrc
conda activate geohelmintos
cd "$SLURM_SUBMIT_DIR"

# Same notebook as run_yolo_1class.sh, but on bigmem (no GPU requested): the notebook
# falls back to CPU on its own (torch.cuda.is_available() is False here). Each of the
# two trainings takes ~4h on CPU, like g_single_plain/h_single_aug.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=geohelmintos \
  --ExecutePreprocessor.timeout=-1 \
  dl/notebooks/yolo_1class.ipynb
