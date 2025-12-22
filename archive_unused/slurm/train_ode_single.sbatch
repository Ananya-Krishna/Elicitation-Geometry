#!/bin/bash
#SBATCH --job-name=ode_train
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ode_%j.out
#SBATCH --error=logs/ode_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Train Neural-ODE for a single model family
# Usage: sbatch train_ode_single.sbatch <model_family>
# Example: sbatch train_ode_single.sbatch llama2
# ============================================================================

set -e

# Get model family from command line or use default
MODEL_FAMILY="${1:-llama2}"

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Create directories
mkdir -p logs ode_output

# Load modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Setup Python
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Directories
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output"
ODE_OUTPUT_DIR="${PROJECT_DIR}/ode_output"

echo "============================================================================"
echo "Training Neural-ODE for ${MODEL_FAMILY}"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================================"

# Check if SAE outputs exist
if [ ! -f "${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy" ]; then
    echo "ERROR: SAE outputs not found for ${MODEL_FAMILY}"
    echo "Please run train_sae.py first or use train_single_model.sbatch"
    exit 1
fi

# Train Neural-ODE
python "${SCRIPT_DIR}/train_neural_ode.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ODE_OUTPUT_DIR}" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --time_dependent

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ Neural-ODE training complete for ${MODEL_FAMILY}!"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "✗ Neural-ODE training failed for ${MODEL_FAMILY}"
    echo "============================================================================"
    exit 1
fi
