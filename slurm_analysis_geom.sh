#!/bin/bash
#SBATCH --job-name=geom_all_models
#SBATCH --output=logs/geom_all_%j.out
#SBATCH --error=logs/geom_all_%j.err
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ------------------- Setup -------------------

# Go to project root
cd /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry

# Make sure logs dir exists
mkdir -p logs

# Load conda + activate env
module load miniconda
conda activate elicitation

echo "Starting all-model geometric analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"

python new_approach/analysis/analysis_all_models_geom_numpy.py \
  --base_path /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry \
  --sae_subdir new_sae_outputs \
  --ode_subdir final/ode \
  --data_dir final/prompts \
  --output_dir final/geom_analysis/newfixedtoxicity \
  --models llama2 falcon mistral \
  --processes 3

echo "All-model geometric analysis DONE."
