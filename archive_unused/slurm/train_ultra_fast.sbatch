#!/bin/bash
#SBATCH --job-name=ultra_fast
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_ultra_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/train_ultra_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# ULTRA-FAST TRAINING - 2 HOUR TARGET
# ============================================================================
# Optimizations:
# 1. Uses gpu_h200 partition (better I/O than scavenge)
# 2. Keeps H5 files open (no open/close overhead)
# 3. Reduced epochs (5 instead of 15)
# 4. Larger batch size (512 instead of 256)
# 5. Subsamples data (50% instead of 100%)
# 6. Higher learning rate for faster convergence
# ============================================================================

# set -u  # Exit on undefined variables (but not on errors)

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
DATA_DIR="${PROJECT_DIR}/old_data_backup"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output_ultra"
ODE_OUTPUT_DIR="${PROJECT_DIR}/ode_output_ultra"
ANALYSIS_OUTPUT_DIR="${PROJECT_DIR}/analysis_output_ultra"

# Create output directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${SAE_OUTPUT_DIR}"
mkdir -p "${ODE_OUTPUT_DIR}"
mkdir -p "${ANALYSIS_OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# Model families
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY="${MODEL_FAMILIES[$SLURM_ARRAY_TASK_ID]}"

module reset
module load miniconda
conda activate elicitation


echo "============================================================================"
echo "ULTRA-FAST Training on RealToxicityPrompts Data: ${MODEL_FAMILY}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Partition: gpu_h200 (better I/O)"
echo "Optimized: Open files, reduced epochs, subsampled data"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================================"

# STEP 1: Train SAE (ULTRA-FAST)
echo ""
echo "STEP 1: Training SAE for ${MODEL_FAMILY} on RealToxicityPrompts (ULTRA-FAST MODE)..."
python "${SCRIPT_DIR}/train_sae_ultra_fast.py" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${SAE_OUTPUT_DIR}" \
    --epochs 3 \
    --batch_size 512 \
    --lr 2e-3 \
    --hidden_dim 16384 \
    --l1_coef 1e-4 \
    --subsample 0.1 \
    --encode_subsample 0.05 \
    --train_on_base

if [ $? -ne 0 ]; then
    echo "ERROR: SAE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo "✓ SAE training complete for ${MODEL_FAMILY}"

# STEP 2: Train Neural-ODE (FAST)
echo ""
echo "STEP 2: Training Neural-ODE for ${MODEL_FAMILY} (FAST MODE)..."
python "${SCRIPT_DIR}/train_neural_ode.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ODE_OUTPUT_DIR}" \
    --epochs 15 \
    --batch_size 256 \
    --lr 1e-4 \
    --time_dependent

if [ $? -ne 0 ]; then
    echo "ERROR: Neural-ODE training failed for ${MODEL_FAMILY}"
    exit 1
fi

echo "✓ Neural-ODE training complete for ${MODEL_FAMILY}"

# STEP 3: Optimal Transport Analysis (FAST)
echo ""
echo "STEP 3: Computing Optimal Transport for ${MODEL_FAMILY} (FAST MODE)..."
mkdir -p "${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
python "${SCRIPT_DIR}/optimal_transport.py" \
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
echo "✓ Complete pipeline finished for ${MODEL_FAMILY} on RealToxicityPrompts (ULTRA-FAST MODE)!"
echo "============================================================================"
echo ""
echo "Outputs saved to:"
echo "  SAE: ${SAE_OUTPUT_DIR}"
echo "  Neural-ODE: ${ODE_OUTPUT_DIR}"
echo "  Analysis: ${ANALYSIS_OUTPUT_DIR}/${MODEL_FAMILY}"
echo "============================================================================"

