#!/bin/bash
# ============================================================================
# Submit Training Jobs to Bouchet HPC
# 
# This script submits parallel training jobs for all model families.
# Each model family runs independently in parallel using SLURM job arrays.
#
# Usage:
#   bash submit_training.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================================"
echo "Submitting Geometric Analysis Training Jobs"
echo "============================================================================"
echo ""

# Create logs directory
mkdir -p logs

# Submit parallel training job (runs all 3 model families in parallel)
echo "Submitting parallel training job..."
JOB_ID=$(sbatch train_parallel.sbatch | grep -oP '\d+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "✓ Job submitted successfully!"
echo ""
echo "Job ID: ${JOB_ID}"
echo "Job Name: geometric_train"
echo ""
echo "This job will run 3 parallel tasks (one per model family):"
echo "  - Task 0: llama2"
echo "  - Task 1: falcon"
echo "  - Task 2: mistral"
echo ""
echo "Monitor job status:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "View output:"
echo "  tail -f logs/train_${JOB_ID}_0.out  # llama2"
echo "  tail -f logs/train_${JOB_ID}_1.out  # falcon"
echo "  tail -f logs/train_${JOB_ID}_2.out  # mistral"
echo ""
echo "Cancel job if needed:"
echo "  scancel ${JOB_ID}"
echo ""
echo "============================================================================"
