#!/bin/bash
#SBATCH --account=otros
#SBATCH --partition=compute-gpu-h100
#SBATCH --qos=h100
#SBATCH --job-name=TSP_duplicates_tour_30000
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scratch/ainfanta_otros/ANYWCSP
#SBATCH --output=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.out
#SBATCH --error=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

RUN_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
MODEL_DIR="models/${RUN_ID}"

test ! -e "results/models_${RUN_ID}_epochs.json"
test ! -e "results/assigment_models_${RUN_ID}_epochs.json"
mkdir -p models results
mkdir "$MODEL_DIR"
cp configs/TSP/TSP_duplicates_tour_19000.json "$MODEL_DIR/config.json"

# Set the epoch budget in this run's configuration snapshot.
.venv-anywcsp/bin/python - "$MODEL_DIR/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
config.update(epochs=30000)
path.write_text(json.dumps(config, indent=4) + "\n")
PY

exec srun .venv-anywcsp/bin/python -u train.py \
    --config "$MODEL_DIR/config.json" \
    --model_dir "$MODEL_DIR" \
    --results_dir results \
    --seed 0 --num_workers 0 \
    --logging_steps 10 --checkpoint_steps 5000
