#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sae_%j.out
#SBATCH --error=logs/sae_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Train SAE for a single model family
# Usage: sbatch train_single_model.sbatch <model_family>
# Example: sbatch train_single_model.sbatch llama2
# ============================================================================

set -e

# Get model family from command line or use default
MODEL_FAMILY="${1:-llama2}"

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Create directories
mkdir -p logs sae_output

# Load modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Setup Python
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Directories
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output"

echo "============================================================================"
echo "Training SAE for ${MODEL_FAMILY}"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================================"

# Train SAE
python "${SCRIPT_DIR}/train_sae.py" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${SAE_OUTPUT_DIR}" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden_dim 16384 \
    --l1_coef 1e-4 \
    --train_on_base

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ SAE training complete for ${MODEL_FAMILY}!"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "✗ SAE training failed for ${MODEL_FAMILY}"
    echo "============================================================================"
    exit 1
fi
