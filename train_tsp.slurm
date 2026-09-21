#!/bin/bash
#SBATCH --account=otros
#SBATCH --job-name=tsp_train
#SBATCH --partition=compute-gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.out
#SBATCH --error=/scratch/ainfanta_otros/ANYWCSP/logs/%u_%x_%j.err

set -euxo pipefail

cd /scratch/ainfanta_otros/ANYWCSP 2>/dev/null || cd /nfs_ssd/ainfanta_otros/ANYWCSP
source .venv-anywcsp/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

hostname
pwd
which python
nvidia-smi
python -u test_cuda.py
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch no detecta CUDA; se detiene el job"'

srun python -u train.py \
  --config configs/TSP/TSP_default.json \
  --model_dir models/TSP \
  --num_workers 0 \
  --logging_steps 10 \
  --checkpoint_steps 5000
