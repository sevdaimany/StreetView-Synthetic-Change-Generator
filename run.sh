#!/bin/bash
#SBATCH --job-name=pipeline3
#SBATCH --partition=jean-zellou
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=0
#SBATCH -e ./logs20/err-%j.err
#SBATCH -o ./logs20/log-%j.out

export HF_HOME=/mnt/common/hdd/home/SImany/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export CUDA_HOME=$CONDA_PREFIX

echo "HF_HOME is set to: $HF_HOME"

eval "$(conda shell.bash hook)"

# Activate your environment
conda activate pipeline3

python main.py