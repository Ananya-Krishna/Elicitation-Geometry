#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_train_%j.out
#SBATCH --error=logs/test_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Test Training Script - Single model to verify GPU usage
# ============================================================================

# Setup - Use absolute path
PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
cd "${PROJECT_DIR}"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Use conda Python directly
PYTHON_CMD=~/.conda/envs/elicitation/bin/python

# Install missing packages if needed
echo "Checking/installing required packages..."
${PYTHON_CMD} -m pip install --quiet --user torchdiffeq POT matplotlib seaborn 2>&1 | grep -v "already satisfied" || true

# Setup environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN_FILE=/home/ark89/.cache/huggingface/token

# Directories
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output"
MODEL_FAMILY="llama2"

echo "============================================================================"
echo "TEST: Training SAE for ${MODEL_FAMILY}"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_CMD}"
echo "============================================================================"

# Test SAE training with just 2 epochs and smaller dataset
echo ""
echo "STEP 1: Testing SAE training for ${MODEL_FAMILY}..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_sae.py" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${SAE_OUTPUT_DIR}" \
    --epochs 2 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden_dim 16384 \
    --l1_coef 1e-4 \
    --train_on_base \
    --max_samples 100

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ TEST COMPLETE - Training works correctly!"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "✗ TEST FAILED - Check logs for errors"
    echo "============================================================================"
    exit 1
fi

