#!/bin/bash
#SBATCH --account=otros
#SBATCH --job-name=TSP_eval_n100_909674
#SBATCH --partition=compute-gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/scratch/ainfanta_otros/ANYWCSP
#SBATCH --output=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.out
#SBATCH --error=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Override with sbatch --export=ALL,CHECKPOINT_NAME=last,NETWORK_STEPS=1000 ...
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best}"
NETWORK_STEPS="${NETWORK_STEPS:-200}"
OUTPUT="results/evaluation_TSP_duplicates_tour_19000_909674_n100_${CHECKPOINT_NAME}_${SLURM_JOB_ID}.json"

mkdir -p results
exec srun .venv-anywcsp/bin/python -u evaluate_tsp.py \
    --model_dir models/TSP_duplicates_tour_19000_909674 \
    --data_path 'data/TravellingSalesman-m1-n100/*.xml' \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --network_steps "$NETWORK_STEPS" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --device cuda \
    --output "$OUTPUT"
