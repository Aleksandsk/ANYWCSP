#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --partition=compute-gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%u_%x_%j.out
#SBATCH --error=logs/%u_%x_%j.err

cd /nfs_ssd/ainfanta_otros/ANYWCSP
source .venv-anywcsp/bin/activate

echo "PWD: $(pwd)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi
python -u test_cuda.py
