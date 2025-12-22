#!/bin/bash
#SBATCH --job-name=ot_from_ode
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/ot_from_ode_%A_%a.out
#SBATCH --error=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry/logs/ot_from_ode_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# RUN OPTIMAL TRANSPORT ANALYSIS FROM ODE OUTPUTS
# ============================================================================
# This script runs Optimal Transport analysis using existing Neural-ODE outputs.
# Uses the ODE outputs in ode_output_ultra/ and SAE encodings in sae_output_ultra/.
# ============================================================================

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
SCRIPT_DIR="${PROJECT_DIR}/model"
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/new_approach/sae_output_ultra"
ODE_OUTPUT_DIR="${PROJECT_DIR}/new_approach/ode_output_ultra"
ANALYSIS_OUTPUT_BASE="${PROJECT_DIR}/new_approach/analysis_output_ultra"

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

# Model families (array task maps to index)
MODEL_FAMILIES=("llama2" "falcon" "mistral")
MODEL_FAMILY="${MODEL_FAMILIES[$SLURM_ARRAY_TASK_ID]}"

# Create model-specific output directory
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_BASE}/${MODEL_FAMILY}"
mkdir -p "${ANALYSIS_OUTPUT_DIR}"

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
echo "OPTIMAL TRANSPORT ANALYSIS: ${MODEL_FAMILY}"
echo "============================================================================"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================================"

# Verify required files exist
SAE_BASE_FILE="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy"
SAE_ALIGNED_FILE="${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_aligned_sae_encoded.npy"
ODE_ACTIONS_FILE="${ODE_OUTPUT_DIR}/${MODEL_FAMILY}/${MODEL_FAMILY}_actions.npy"

if [ ! -f "${SAE_BASE_FILE}" ] || [ ! -f "${SAE_ALIGNED_FILE}" ]; then
    echo "ERROR: SAE encoded files not found!"
    echo "  Base: ${SAE_BASE_FILE}"
    echo "  Aligned: ${SAE_ALIGNED_FILE}"
    exit 1
fi

if [ ! -f "${ODE_ACTIONS_FILE}" ]; then
    echo "ERROR: Neural-ODE actions file not found!"
    echo "  Actions: ${ODE_ACTIONS_FILE}"
    exit 1
fi

echo "✓ Found required files:"
echo "  SAE Base: ${SAE_BASE_FILE}"
echo "  SAE Aligned: ${SAE_ALIGNED_FILE}"
echo "  ODE Actions: ${ODE_ACTIONS_FILE}"

# Run Optimal Transport analysis
# Note: train_ot_realtox.py expects actions file at ode_dir/{model_family}_actions.npy
# But we have it at ode_dir/{model_family}/{model_family}_actions.npy
# So we pass the model-specific subdirectory as ode_dir
echo ""
echo "Running Optimal Transport analysis for ${MODEL_FAMILY}..."
${PYTHON_CMD} "${SCRIPT_DIR}/train_ot_realtox.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --ode_dir "${ODE_OUTPUT_DIR}/${MODEL_FAMILY}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --ot_samples 2000 \
    --ot_reg 0.01

if [ $? -ne 0 ]; then
    echo "ERROR: Optimal Transport analysis failed for ${MODEL_FAMILY}"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ Optimal Transport analysis complete for ${MODEL_FAMILY}!"
echo "============================================================================"
echo ""
echo "Outputs saved to: ${ANALYSIS_OUTPUT_DIR}"
echo "  - ot_results.json: Wasserstein-2 distance"
echo "  - ot_comparison.json: OT vs Action comparison"
echo "  - domain_analysis.json: Toxicity-based analysis"
echo "  - regression_calibration.json: Refusal prediction model"
echo "  - final_report.txt: Summary report"
echo "  - visualizations/*.png: t-SNE and analysis plots"
echo "============================================================================"

