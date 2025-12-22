#!/bin/bash
#SBATCH --job-name=from_encoded
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_from_encoded_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_from_encoded_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# TRAIN FROM EXISTING ENCODED DATA
# ============================================================================
# This script SKIPS SAE training/encoding and uses existing encoded files
# from job 3407975 (sae_output_realtox/). Just does:
# 1. Neural-ODE training (with larger batch size)
# 2. Optimal Transport analysis
# ============================================================================

set -u

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output_realtox"  # Use existing encoded files
ODE_OUTPUT_DIR="${PROJECT_DIR}/ode_output_from_encoded"
ANALYSIS_OUTPUT_DIR="${PROJECT_DIR}/analysis_output_from_encoded"

# Create output directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${ODE_OUTPUT_DIR}"
mkdir -p "${ANALYSIS_OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# Model families
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY="${MODEL_FAMILIES[${SLURM_ARRAY_TASK_ID}]}"

# Python command
PYTHON_CMD="${HOME}/.conda/envs/elicitation/bin/python"

# Install dependencies if needed
pip install --quiet --user torchdiffeq POT matplotlib seaborn 2>/dev/null || true

echo "============================================================================"
echo "Training from EXISTING ENCODED DATA: ${MODEL_FAMILY}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Partition: gpu_h200"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "SAE Output Dir: ${SAE_OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Python: ${PYTHON_CMD}"
echo "============================================================================"

# Check if encoded files exist
BASE_ENCODED="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy"
ALIGNED_ENCODED="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_aligned_sae_encoded.npy"

if [ ! -f "${BASE_ENCODED}" ] || [ ! -f "${ALIGNED_ENCODED}" ]; then
    echo ""
    echo "ERROR: Encoded files not found!"
    echo "  Looking for: ${BASE_ENCODED}"
    echo "  Looking for: ${ALIGNED_ENCODED}"
    echo ""
    echo "Job 3407975 may still be encoding. Please wait for it to complete."
    echo "Or run the full pipeline with train_ultra_fast.sbatch"
    exit 1
fi

echo ""
echo "✓ Found encoded files:"
echo "  Base: ${BASE_ENCODED} ($(du -h "${BASE_ENCODED}" | cut -f1))"
echo "  Aligned: ${ALIGNED_ENCODED} ($(du -h "${ALIGNED_ENCODED}" | cut -f1))"
echo ""

# STEP 1: Train Neural-ODE (FAST with larger batch size)
echo ""
echo "STEP 1: Training Neural-ODE for ${MODEL_FAMILY} (FAST MODE, large batch)..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_neural_ode.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ODE_OUTPUT_DIR}" \
    --epochs 15 \
    --batch_size 512 \
    --lr 1e-4 \
    --time_dependent

if [ $? -ne 0 ]; then
    echo "ERROR: Neural-ODE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo "✓ Neural-ODE training complete for ${MODEL_FAMILY}"

# STEP 2: Optimal Transport Analysis (FAST)
echo ""
echo "STEP 2: Computing Optimal Transport for ${MODEL_FAMILY} (FAST MODE)..."
mkdir -p "${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
${PYTHON_CMD} "${SCRIPT_DIR}/optimal_transport.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --ode_dir "${ODE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}" \
    --ot_samples 2000 \
    --ot_reg 0.01

if [ $? -ne 0 ]; then
    echo "ERROR: Optimal Transport failed for ${MODEL_FAMILY}"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Complete pipeline finished for ${MODEL_FAMILY} (from existing encoded data)!"
echo "============================================================================"
echo ""
echo "Outputs saved to:"
echo "  Neural-ODE: ${ODE_OUTPUT_DIR}"
echo "  Analysis: ${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
echo "============================================================================"

