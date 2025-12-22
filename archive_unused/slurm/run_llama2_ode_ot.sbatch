#!/bin/bash
#SBATCH --job-name=llama2_ode_ot
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00 # 1 hour should be enough for Neural-ODE + OT
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/llama2_ode_ot_%j.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/llama2_ode_ot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# Setup - Use absolute path
PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
cd "${PROJECT_DIR}"

# Create directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/ode_output_ultra"
mkdir -p "${PROJECT_DIR}/analysis_output_ultra/llama2"

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

echo "============================================================================"
echo "Neural-ODE + Optimal Transport for llama2"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_CMD}"
echo "============================================================================"

# Check if encoded files exist
BASE_FILE="${PROJECT_DIR}/sae_output_ultra/llama2_base_sae_encoded.npy"
ALIGNED_FILE="${PROJECT_DIR}/sae_output_ultra/llama2_aligned_sae_encoded.npy"

if [ ! -f "${BASE_FILE}" ] || [ ! -f "${ALIGNED_FILE}" ]; then
    echo "ERROR: Encoded files not found!"
    echo "  Base: ${BASE_FILE}"
    echo "  Aligned: ${ALIGNED_FILE}"
    exit 1
fi

echo "✓ Found encoded files"

# STEP 1: Train Neural-ODE
echo ""
echo "STEP 1: Training Neural-ODE for llama2..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_neural_ode.py" \
    --sae_dir "${PROJECT_DIR}/sae_output_ultra" \
    --data_dir "${PROJECT_DIR}" \
    --model_family llama2 \
    --output_dir "${PROJECT_DIR}/ode_output_ultra" \
    --epochs 15 \
    --batch_size 256 \
    --lr 1e-4 \
    --time_dependent

if [ $? -ne 0 ]; then
    echo "ERROR: Neural-ODE training failed for llama2"
    exit 1
fi

echo "✓ Neural-ODE training complete for llama2"

# STEP 2: Optimal Transport Analysis
echo ""
echo "STEP 2: Computing Optimal Transport for llama2..."
${PYTHON_CMD} "${SCRIPT_DIR}/optimal_transport.py" \
    --sae_dir "${PROJECT_DIR}/sae_output_ultra" \
    --ode_dir "${PROJECT_DIR}/ode_output_ultra" \
    --data_dir "${PROJECT_DIR}" \
    --model_family llama2 \
    --output_dir "${PROJECT_DIR}/analysis_output_ultra/llama2" \
    --ot_samples 2000 \
    --ot_reg 0.01

if [ $? -ne 0 ]; then
    echo "ERROR: Optimal Transport failed for llama2"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Complete pipeline (Neural-ODE + OT) finished for llama2!"
echo "============================================================================"
echo ""
echo "Outputs saved to:"
echo "  Neural-ODE: ${PROJECT_DIR}/ode_output_ultra"
echo "  Analysis: ${PROJECT_DIR}/analysis_output_ultra/llama2"
echo "============================================================================"



