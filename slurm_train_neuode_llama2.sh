#!/bin/bash
#SBATCH --job-name=neuode_llama2
#SBATCH --output=logs/neuode_llama2_%j.out
#SBATCH --error=logs/neuode_llama2_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- Setup ---

# Go to project root
cd /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry

mkdir -p logs

# Activate conda env
module load miniconda
conda activate elicitation

echo "Starting Neural-ODE training for LLaMA2..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python model/train_neur_ode_realtox.py \
  --sae_root new_sae_outputs \
  --data_dir data/numpy_activations/llama2_base \
  --model_family llama2 \
  --output_root new_approach/ode_output_ultra/llama2 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-4

echo "Neural-ODE training for LLaMA2 DONE."
