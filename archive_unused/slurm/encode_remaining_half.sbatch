#!/bin/bash
#SBATCH --job-name=encode_remaining
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/encode_remaining_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/encode_remaining_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# ENCODE REMAINING 50% OF DATA
# ============================================================================
# This script encodes the second half of data (samples 2250-4499) and combines
# with existing encodings (samples 0-2249) to create full encodings (4500 samples).
# 
# IMPORTANT: Does NOT overwrite existing encodings until new ones are verified.
# ============================================================================

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/new_approach/sae_output_ultra"
OUTPUT_DIR="${PROJECT_DIR}/new_approach/sae_output_ultra"

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

# Model families (array task maps to index)
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY="${MODEL_FAMILIES[$SLURM_ARRAY_TASK_ID]}"

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
echo "ENCODING REMAINING 50% FOR ${MODEL_FAMILY}"
echo "============================================================================"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================================"

# Verify existing encodings exist
BASE_FILE="${OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy"
ALIGNED_FILE="${OUTPUT_DIR}/${MODEL_FAMILY}_aligned_sae_encoded.npy"

# Check for model-specific SAE first, then generic
MODEL_SAE_FILE="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_best_sae.pt"
GENERIC_SAE_FILE="${SAE_OUTPUT_DIR}/best_sae.pt"

if [ ! -f "${BASE_FILE}" ] || [ ! -f "${ALIGNED_FILE}" ]; then
    echo "ERROR: Existing encodings not found!"
    echo "  Base: ${BASE_FILE}"
    echo "  Aligned: ${ALIGNED_FILE}"
    exit 1
fi

# Check for SAE file (model-specific or generic)
if [ -f "${MODEL_SAE_FILE}" ]; then
    echo "✓ Found model-specific SAE: ${MODEL_SAE_FILE}"
elif [ -f "${GENERIC_SAE_FILE}" ]; then
    echo "⚠️  Using generic SAE (may cause dimension errors): ${GENERIC_SAE_FILE}"
    echo "   If encoding fails, ensure ${MODEL_SAE_FILE} exists"
else
    echo "ERROR: SAE checkpoint not found!"
    echo "  Tried: ${MODEL_SAE_FILE}"
    echo "  Tried: ${GENERIC_SAE_FILE}"
    exit 1
fi

echo "✓ Found existing encodings"

# Run encoding script
echo ""
echo "Encoding remaining 50% and combining with existing encodings..."
${PYTHON_CMD} "${SCRIPT_DIR}/encode_remaining_half.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 256 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Encoding failed for ${MODEL_FAMILY}"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Complete: Full encodings (4500 samples) saved for ${MODEL_FAMILY}!"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  ${BASE_FILE}: 4,500 samples (was 2,250)"
echo "  ${ALIGNED_FILE}: 4,500 samples (was 2,250)"
echo "============================================================================"

