#!/bin/bash
# ============================================================================
# Check if RealToxicityPrompts data collection is complete
# If complete, automatically submit training jobs
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================================"
echo "Checking RealToxicityPrompts Data Collection Status"
echo "============================================================================"

# Check for all required activation files
REQUIRED_MODELS=(
    "llama2_base" "llama2_aligned"
    "falcon_base" "falcon_aligned"
    "mistral_base" "mistral_aligned"
)

ALL_COMPLETE=true
MISSING_FILES=()

for model in "${REQUIRED_MODELS[@]}"; do
    # Check for at least one split file for each model
    found=$(find elicitation_data_* -name "${model}_split_*_activations.h5" -type f 2>/dev/null | wc -l)
    
    if [ "$found" -eq 0 ]; then
        ALL_COMPLETE=false
        MISSING_FILES+=("${model}")
        echo "  ✗ Missing: ${model} (0 files found)"
    else
        echo "  ✓ Found: ${model} (${found} split files)"
    fi
done

echo ""

if [ "$ALL_COMPLETE" = true ]; then
    echo "============================================================================"
    echo "✅ All RealToxicityPrompts data collection complete!"
    echo "============================================================================"
    echo ""
    echo "Submitting training jobs..."
    
    # Submit training job
    cd "${PROJECT_DIR}/model"
    JOB_ID=$(sbatch train_realtoxicity.sbatch | grep -oP '\d+')
    
    echo ""
    echo "============================================================================"
    echo "✅ Training jobs submitted!"
    echo "============================================================================"
    echo "  Job ID: ${JOB_ID}"
    echo "  Array tasks: 0-2 (llama2, falcon, mistral)"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/train_realtox_${JOB_ID}_*.out"
    echo "============================================================================"
else
    echo "============================================================================"
    echo "⏳ Data collection still in progress..."
    echo "============================================================================"
    echo "Missing models:"
    for model in "${MISSING_FILES[@]}"; do
        echo "  - ${model}"
    done
    echo ""
    echo "Run this script again once data collection is complete."
    echo "============================================================================"
    exit 1
fi

