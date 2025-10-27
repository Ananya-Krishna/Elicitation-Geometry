#!/bin/bash
#SBATCH --job-name=parallel_falcon_base
#SBATCH --output=data/logs/parallel_falcon_base_%j.out
#SBATCH --error=data/logs/parallel_falcon_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Create logs directory
mkdir -p logs

# Load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN_FILE=/home/ark89/.cache/huggingface/token

# Change to project directory
cd /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/data

# Run parallel data-split for falcon_base
echo "Starting PARALLEL DATA-SPLIT for falcon_base..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: H200 (NVIDIA H200)"
echo "Time limit: 4 hours"

~/.conda/envs/elicitation/bin/python data_curation_h200_parallel_data.py falcon_base 2 6

echo "PARALLEL DATA-SPLIT for falcon_base completed!"
