#!/bin/bash
#SBATCH --job-name=geometric_analysis
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ark89@yale.edu

# ============================================================================
# Geometric Analysis Training Pipeline
# Trains SAE, Neural-ODE, and Optimal Transport models for all model families
# ============================================================================

set -e  # Exit on error

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Create necessary directories
mkdir -p logs
mkdir -p sae_output
mkdir -p ode_output
mkdir -p analysis_output

# Load modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Setup Python environment
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Model families to process
MODEL_FAMILIES=("llama2" "falcon" "mistral")

# Data directory - adjust if your data is in a different location
DATA_DIR="${PROJECT_DIR}"
SAE_OUTPUT_DIR="${PROJECT_DIR}/sae_output"
ODE_OUTPUT_DIR="${PROJECT_DIR}/ode_output"
ANALYSIS_OUTPUT_DIR="${PROJECT_DIR}/analysis_output"

echo "============================================================================"
echo "Geometric Analysis Training Pipeline"
echo "============================================================================"
echo "Project Directory: ${PROJECT_DIR}"
echo "Data Directory: ${DATA_DIR}"
echo "SAE Output: ${SAE_OUTPUT_DIR}"
echo "ODE Output: ${ODE_OUTPUT_DIR}"
echo "Analysis Output: ${ANALYSIS_OUTPUT_DIR}"
echo "============================================================================"
echo ""

# Function to check if SAE training is complete
check_sae_complete() {
    local model_family=$1
    local sae_file="${SAE_OUTPUT_DIR}/${model_family}_base_sae_encoded.npy"
    [ -f "$sae_file" ] && [ -f "${SAE_OUTPUT_DIR}/${model_family}_aligned_sae_encoded.npy" ]
}

# Function to check if ODE training is complete
check_ode_complete() {
    local model_family=$1
    local actions_file="${ODE_OUTPUT_DIR}/${model_family}_actions.npy"
    [ -f "$actions_file" ]
}

# STEP 1: Train SAE for all model families
echo "============================================================================"
echo "STEP 1: Training Sparse Autoencoders (SAE)"
echo "============================================================================"

for model_family in "${MODEL_FAMILIES[@]}"; do
    echo ""
    echo "Training SAE for ${model_family}..."
    
    if check_sae_complete "$model_family"; then
        echo "  ✓ SAE already trained for ${model_family}, skipping..."
    else
        echo "  → Training SAE on ${model_family}..."
        python "${SCRIPT_DIR}/train_sae.py" \
            --data_dir "${DATA_DIR}" \
            --model_family "${model_family}" \
            --output_dir "${SAE_OUTPUT_DIR}" \
            --epochs 20 \
            --batch_size 256 \
            --lr 1e-3 \
            --hidden_dim 16384 \
            --l1_coef 1e-4 \
            --train_on_base
        
        if [ $? -eq 0 ]; then
            echo "  ✓ SAE training completed for ${model_family}"
        else
            echo "  ✗ SAE training failed for ${model_family}"
            exit 1
        fi
    fi
done

echo ""
echo "✓ All SAE training complete!"
echo ""

# STEP 2: Train Neural-ODE for all model families
echo "============================================================================"
echo "STEP 2: Training Neural-ODE Flow"
echo "============================================================================"

for model_family in "${MODEL_FAMILIES[@]}"; do
    echo ""
    echo "Training Neural-ODE for ${model_family}..."
    
    if check_ode_complete "$model_family"; then
        echo "  ✓ Neural-ODE already trained for ${model_family}, skipping..."
    else
        # Check if SAE outputs exist
        if ! check_sae_complete "$model_family"; then
            echo "  ✗ SAE outputs not found for ${model_family}, skipping Neural-ODE..."
            continue
        fi
        
        echo "  → Training Neural-ODE on ${model_family}..."
        python "${SCRIPT_DIR}/train_neural_ode.py" \
            --sae_dir "${SAE_OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --model_family "${model_family}" \
            --output_dir "${ODE_OUTPUT_DIR}" \
            --epochs 50 \
            --batch_size 128 \
            --lr 1e-4 \
            --time_dependent
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Neural-ODE training completed for ${model_family}"
        else
            echo "  ✗ Neural-ODE training failed for ${model_family}"
            exit 1
        fi
    fi
done

echo ""
echo "✓ All Neural-ODE training complete!"
echo ""

# STEP 3: Compute Optimal Transport and Analysis
echo "============================================================================"
echo "STEP 3: Computing Optimal Transport and Final Analysis"
echo "============================================================================"

for model_family in "${MODEL_FAMILIES[@]}"; do
    echo ""
    echo "Computing Optimal Transport for ${model_family}..."
    
    # Check if ODE outputs exist
    if ! check_ode_complete "$model_family"; then
        echo "  ✗ Neural-ODE outputs not found for ${model_family}, skipping Optimal Transport..."
        continue
    fi
    
    echo "  → Computing Optimal Transport for ${model_family}..."
    python "${SCRIPT_DIR}/optimal_transport.py" \
        --sae_dir "${SAE_OUTPUT_DIR}" \
        --ode_dir "${ODE_OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        --model_family "${model_family}" \
        --output_dir "${ANALYSIS_OUTPUT_DIR}/${model_family}" \
        --ot_samples 5000 \
        --ot_reg 0.01
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Optimal Transport analysis completed for ${model_family}"
    else
        echo "  ✗ Optimal Transport analysis failed for ${model_family}"
        exit 1
    fi
done

echo ""
echo "============================================================================"
echo "✓ ALL TRAINING AND ANALYSIS COMPLETE!"
echo "============================================================================"
echo ""
echo "Output Summary:"
echo "  SAE Models: ${SAE_OUTPUT_DIR}"
echo "  Neural-ODE Models: ${ODE_OUTPUT_DIR}"
echo "  Analysis Results: ${ANALYSIS_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review analysis results in ${ANALYSIS_OUTPUT_DIR}"
echo "  2. Check visualizations in analysis_output/*/visualizations/"
echo "  3. Review final reports in analysis_output/*/final_report.txt"
echo ""
