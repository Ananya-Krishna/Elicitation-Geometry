#!/bin/bash
#SBATCH --job-name=realtox_train
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_realtox_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_realtox_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Train on RealToxicityPrompts Data
# Runs one model family per array task
# ============================================================================

# Setup - Use absolute path (same as data collection scripts)
PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
cd "${PROJECT_DIR}"

# Create directories in project root (must be done before any output)
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/sae_output_realtox"
mkdir -p "${PROJECT_DIR}/ode_output_realtox"
mkdir -p "${PROJECT_DIR}/analysis_output_realtox"

# Load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Use conda Python directly (same as data collection scripts)
PYTHON_CMD=~/.conda/envs/elicitation/bin/python

# Install missing packages if needed
echo "Checking/installing required packages..."
${PYTHON_CMD} -m pip install --quiet --user torchdiffeq POT matplotlib seaborn 2>/dev/null || true

# Setup environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN_FILE=/home/ark89/.cache/huggingface/token

# Model families (array task maps to index)
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY=${MODEL_FAMILIES[${SLURM_ARRAY_TASK_ID}]}

# Directories (separate from old data training)
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output_realtox"
ODE_OUTPUT_DIR="${PROJECT_DIR}/ode_output_realtox"
ANALYSIS_OUTPUT_DIR="${PROJECT_DIR}/analysis_output_realtox"

echo "============================================================================"
echo "Training on RealToxicityPrompts Data: ${MODEL_FAMILY}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Partition: scavenge_gpu (preemptable)"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_CMD}"
echo "============================================================================"

# STEP 1: Train SAE
echo ""
echo "STEP 1: Training SAE for ${MODEL_FAMILY} on RealToxicityPrompts..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_sae.py" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${SAE_OUTPUT_DIR}" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden_dim 16384 \
    --l1_coef 1e-4 \
    --train_on_base

if [ $? -ne 0 ]; then
    echo "ERROR: SAE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo "✓ SAE training complete for ${MODEL_FAMILY}"

# STEP 2: Train Neural-ODE
echo ""
echo "STEP 2: Training Neural-ODE for ${MODEL_FAMILY}..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_neural_ode.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ODE_OUTPUT_DIR}" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --time_dependent

if [ $? -ne 0 ]; then
    echo "ERROR: Neural-ODE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo "✓ Neural-ODE training complete for ${MODEL_FAMILY}"

# STEP 3: Optimal Transport Analysis
echo ""
echo "STEP 3: Computing Optimal Transport for ${MODEL_FAMILY}..."
mkdir -p "${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
${PYTHON_CMD} "${SCRIPT_DIR}/optimal_transport.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --ode_dir "${ODE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}" \
    --ot_samples 5000 \
    --ot_reg 0.01

if [ $? -ne 0 ]; then
    echo "ERROR: Optimal Transport failed for ${MODEL_FAMILY}"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Complete pipeline finished for ${MODEL_FAMILY} on RealToxicityPrompts!"
echo "============================================================================"
echo ""
echo "Outputs saved to:"
echo "  SAE: ${SAE_OUTPUT_DIR}"
echo "  Neural-ODE: ${ODE_OUTPUT_DIR}"
echo "  Analysis: ${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
echo "============================================================================"

