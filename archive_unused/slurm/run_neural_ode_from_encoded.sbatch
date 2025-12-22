#!/bin/bash
#SBATCH --job-name=neuode_from_encoded
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/neuode_from_encoded_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/neuode_from_encoded_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# RUN NEURAL-ODE FROM EXISTING ENCODINGS
# ============================================================================
# This script runs Neural-ODE training using existing SAE encodings.
# Uses the encodings in sae_output_ultra/ (2,250 samples each, first half).
# ============================================================================

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/new_approach/sae_output_ultra"
ODE_OUTPUT_BASE="${PROJECT_DIR}/new_approach/ode_output_ultra"

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

# Model families (array task maps to index)
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY="${MODEL_FAMILIES[$SLURM_ARRAY_TASK_ID]}"

# Create model-specific output directory
ODE_OUTPUT_DIR="${ODE_OUTPUT_BASE}/${MODEL_FAMILY}"
mkdir -p "${ODE_OUTPUT_DIR}"

module reset
module load miniconda
conda activate elicitation

PYTHON_CMD="${HOME}/.conda/envs/elicitation/bin/python"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN_FILE=/home/ark89/.cache/huggingface/token

echo "============================================================================"
echo "NEURAL-ODE TRAINING FROM EXISTING ENCODINGS: ${MODEL_FAMILY}"
echo "============================================================================"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================================"

# Verify encoded files exist
BASE_FILE="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy"
ALIGNED_FILE="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_aligned_sae_encoded.npy"

if [ ! -f "${BASE_FILE}" ] || [ ! -f "${ALIGNED_FILE}" ]; then
    echo "ERROR: Encoded files not found!"
    echo "  Base: ${BASE_FILE}"
    echo "  Aligned: ${ALIGNED_FILE}"
    exit 1
fi

echo "✓ Found encoded files:"
echo "  Base: ${BASE_FILE}"
echo "  Aligned: ${ALIGNED_FILE}"

# Run Neural-ODE training
echo ""
echo "Training Neural-ODE for ${MODEL_FAMILY}..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_neuode_realtox.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ODE_OUTPUT_DIR}" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "ERROR: Neural-ODE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Neural-ODE training complete for ${MODEL_FAMILY}!"
echo "============================================================================"
echo ""
echo "Outputs saved to: ${ODE_OUTPUT_DIR}"
echo "  - best_neural_ode.pt: Trained Neural-ODE model checkpoint"
echo "  - neural_ode_history.json: Training metrics"
echo "  - ${MODEL_FAMILY}_actions.npy: Action values A(X) for each sample"
echo "  - gap_analysis.json: Toxicity-based analysis"
echo "============================================================================"

