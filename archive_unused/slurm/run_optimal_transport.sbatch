#!/bin/bash
#SBATCH --job-name=ot_analysis
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ot_%j.out
#SBATCH --error=logs/ot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Run Optimal Transport Analysis for a single model family
# Usage: sbatch run_optimal_transport.sbatch <model_family>
# Example: sbatch run_optimal_transport.sbatch llama2
# ============================================================================

set -e

# Get model family from command line or use default
MODEL_FAMILY="${1:-llama2}"

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Create directories
mkdir -p logs analysis_output

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
ANALYSIS_OUTPUT_DIR="${PROJECT_DIR}/analysis_output/${MODEL_FAMILY}"

mkdir -p "${ANALYSIS_OUTPUT_DIR}"

echo "============================================================================"
echo "Optimal Transport Analysis for ${MODEL_FAMILY}"
echo "============================================================================"
echo "Project: ${PROJECT_DIR}"
echo "Model Family: ${MODEL_FAMILY}"
echo "============================================================================"

# Check if required inputs exist
if [ ! -f "${SAE_OUTPUT_DIR}/${MODEL_FAMILY}_base_sae_encoded.npy" ]; then
    echo "ERROR: SAE outputs not found for ${MODEL_FAMILY}"
    exit 1
fi

if [ ! -f "${ODE_OUTPUT_DIR}/${MODEL_FAMILY}_actions.npy" ]; then
    echo "ERROR: Neural-ODE outputs not found for ${MODEL_FAMILY}"
    exit 1
fi

# Run Optimal Transport
python "${SCRIPT_DIR}/optimal_transport.py" \
    --sae_dir "${SAE_OUTPUT_DIR}" \
    --ode_dir "${ODE_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --model_family "${MODEL_FAMILY}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --ot_samples 5000 \
    --ot_reg 0.01

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ Optimal Transport analysis complete for ${MODEL_FAMILY}!"
    echo "============================================================================"
    echo "Results saved to: ${ANALYSIS_OUTPUT_DIR}"
else
    echo ""
    echo "============================================================================"
    echo "✗ Optimal Transport analysis failed for ${MODEL_FAMILY}"
    echo "============================================================================"
    exit 1
fi
