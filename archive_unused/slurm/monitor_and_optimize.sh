#!/bin/bash
# Monitor job progress and automatically optimize/resubmit if needed

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
JOB_ID="${1:-3422245}"  # Default to current job, or pass as argument
CHECK_INTERVAL=300  # Check every 5 minutes
MAX_RUNTIME_HOURS=20  # If estimated completion > this, optimize
MAX_OPTIMIZATION_ATTEMPTS=15  # Maximum number of optimization attempts

cd "${PROJECT_DIR}"

# Track optimization attempts across resubmissions
OPT_COUNTER_FILE="${PROJECT_DIR}/.optimization_counter_${JOB_ID}"
if [ -f "${OPT_COUNTER_FILE}" ]; then
    OPTIMIZATION_COUNT=$(cat "${OPT_COUNTER_FILE}")
else
    OPTIMIZATION_COUNT=0
    echo "0" > "${OPT_COUNTER_FILE}"
fi

echo "============================================================================"
echo "MONITORING JOB ${JOB_ID}"
echo "============================================================================"
echo "Checking every ${CHECK_INTERVAL} seconds (5 minutes)"
echo "Will optimize/resubmit if estimated completion > ${MAX_RUNTIME_HOURS} hours"
echo "Optimization attempts so far: ${OPTIMIZATION_COUNT}/${MAX_OPTIMIZATION_ATTEMPTS}"
echo "============================================================================"
echo ""

# Function to check job status
check_job_status() {
    local job_id=$1
    squeue -j "${job_id}" -h --format="%.8T" 2>/dev/null || echo "COMPLETED"
}

# Function to estimate time remaining
estimate_completion() {
    local task_id=$1
    local log_file="${PROJECT_DIR}/logs/train_ultra_${JOB_ID}_${task_id}.out"
    
    if [ ! -f "${log_file}" ]; then
        echo "N/A"
        return
    fi
    
    # Check current phase
    local phase=$(grep -E "STEP|Pre-loading|Training|Encoding|Computing" "${log_file}" | tail -1 | sed 's/.*STEP \([0-9]\):.*/\1/' | head -1)
    
    # If pre-loading, estimate from progress
    if grep -q "Pre-loading" "${log_file}" | tail -1; then
        local progress_line=$(grep -E "\[.*/[0-9]+\]" "${log_file}" | tail -1)
        if [ -n "${progress_line}" ]; then
            local current=$(echo "${progress_line}" | sed -n 's/.*\[\([0-9]*\)\/.*/\1/p')
            local total=$(echo "${progress_line}" | sed -n 's/.*\/\([0-9]*\)\].*/\1/p')
            if [ -n "${current}" ] && [ -n "${total}" ] && [ "${total}" -gt 0 ]; then
                local elapsed=$(squeue -j "${JOB_ID}_${task_id}" -h --format="%.10M" 2>/dev/null | head -1)
                if [ -n "${elapsed}" ]; then
                    # Convert elapsed time to seconds
                    local elapsed_sec=$(echo "${elapsed}" | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}')
                    if [ -n "${elapsed_sec}" ] && [ "${elapsed_sec}" -gt 0 ] && [ "${current}" -gt 0 ]; then
                        local rate=$(echo "scale=2; ${elapsed_sec} / ${current}" | bc 2>/dev/null)
                        local remaining=$(echo "scale=0; (${total} - ${current}) * ${rate}" | bc 2>/dev/null)
                        if [ -n "${remaining}" ] && [ "${remaining}" -gt 0 ]; then
                            local hours=$(echo "scale=1; ${remaining} / 3600" | bc 2>/dev/null)
                            echo "${hours}"
                            return
                        fi
                    fi
                fi
            fi
        fi
    fi
    
    # Default estimate based on phase
    case "${phase}" in
        1) echo "8" ;;  # SAE training
        2) echo "4" ;;  # Neural-ODE
        3) echo "1" ;;  # Optimal Transport
        *) echo "12" ;; # Unknown/default
    esac
}

# Function to optimize and resubmit
optimize_and_resubmit() {
    local current_job=$1
    local reason=$2
    
    # Increment optimization counter
    OPTIMIZATION_COUNT=$((OPTIMIZATION_COUNT + 1))
    echo "${OPTIMIZATION_COUNT}" > "${OPT_COUNTER_FILE}"
    
    # Check if we've exceeded the limit
    if [ "${OPTIMIZATION_COUNT}" -gt "${MAX_OPTIMIZATION_ATTEMPTS}" ]; then
        echo ""
        echo "============================================================================"
        echo "❌ MAXIMUM OPTIMIZATION ATTEMPTS EXCEEDED"
        echo "============================================================================"
        echo "Optimization attempts: ${OPTIMIZATION_COUNT} (max: ${MAX_OPTIMIZATION_ATTEMPTS})"
        echo "Reason: ${reason}"
        echo ""
        echo "Cancelling job ${current_job} and stopping monitoring..."
        scancel "${current_job}"
        echo ""
        echo "Job cancelled. Please investigate the root cause of the slowdown."
        echo "============================================================================"
        exit 1
    fi
    
    echo ""
    echo "============================================================================"
    echo "⚠️  OPTIMIZING AND RESUBMITTING (Attempt ${OPTIMIZATION_COUNT}/${MAX_OPTIMIZATION_ATTEMPTS})"
    echo "============================================================================"
    echo "Reason: ${reason}"
    echo "Cancelling job ${current_job}..."
    
    scancel "${current_job}"
    sleep 5
    
    # Further reduce subsampling (progressive reduction)
    if [ "${OPTIMIZATION_COUNT}" -le 5 ]; then
        # First 5 attempts: reduce to 30%
        echo "Reducing subsampling to 30% for faster completion..."
        sed -i 's/--subsample 0\.[0-9]*/--subsample 0.3/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/--encode_subsample 0\.[0-9]*/--encode_subsample 0.3/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/.*Training on [0-9]*%/OPTIMIZED: Training on 30%/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
    elif [ "${OPTIMIZATION_COUNT}" -le 10 ]; then
        # Next 5 attempts: reduce to 20%
        echo "Reducing subsampling to 20% for faster completion..."
        sed -i 's/--subsample 0\.[0-9]*/--subsample 0.2/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/--encode_subsample 0\.[0-9]*/--encode_subsample 0.2/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/.*Training on [0-9]*%/OPTIMIZED: Training on 20%/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
    else
        # Final attempts: reduce to 10%
        echo "Reducing subsampling to 10% for faster completion..."
        sed -i 's/--subsample 0\.[0-9]*/--subsample 0.1/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/--encode_subsample 0\.[0-9]*/--encode_subsample 0.1/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/.*Training on [0-9]*%/OPTIMIZED: Training on 10%/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
    fi
    
    # Reduce epochs progressively
    if [ "${OPTIMIZATION_COUNT}" -le 5 ]; then
        sed -i 's/--epochs 5/--epochs 4/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
        sed -i 's/--epochs 20/--epochs 15/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch"
    elif [ "${OPTIMIZATION_COUNT}" -le 10 ]; then
        sed -i 's/--epochs [0-9]*/--epochs 3/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch" | grep -E "train_sawe" || true
        sed -i 's/--epochs [0-9]*/--epochs 10/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch" | grep -E "train_neuode" || true
    else
        sed -i 's/--epochs [0-9]*/--epochs 2/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch" | grep -E "train_sawe" || true
        sed -i 's/--epochs [0-9]*/--epochs 8/g' "${PROJECT_DIR}/model/train_uf_realtox.sbatch" | grep -E "train_neuode" || true
    fi
    
    echo "Submitting optimized job..."
    local new_job=$(sbatch "${PROJECT_DIR}/model/train_uf_realtox.sbatch" | grep -o '[0-9]*')
    
    # Update counter file with new job ID
    echo "${OPTIMIZATION_COUNT}" > "${PROJECT_DIR}/.optimization_counter_${new_job}"
    
    echo ""
    echo "✓ New job submitted: ${new_job}"
    echo "  - Optimization attempt: ${OPTIMIZATION_COUNT}/${MAX_OPTIMIZATION_ATTEMPTS}"
    echo ""
    echo "Updating monitor to track new job..."
    echo "${new_job}" > "${PROJECT_DIR}/.current_monitored_job"
    
    return 0
}

# Main monitoring loop
check_count=0
while true; do
    check_count=$((check_count + 1))
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "[${current_time}] Check #${check_count}"
    echo "--------------------------------------------------------------------------------"
    
    # Check if job is still running
    status=$(check_job_status "${JOB_ID}")
    
    if [ "${status}" = "COMPLETED" ] || [ -z "${status}" ]; then
        echo "Job ${JOB_ID} has completed or is not found."
        echo "Checking final status..."
        
        # Check if all tasks completed successfully
        all_success=true
        for task in 0 1 2; do
            log_file="${PROJECT_DIR}/logs/train_ultra_${JOB_ID}_${task}.out"
            if [ -f "${log_file}" ]; then
                if grep -q "Complete pipeline finished" "${log_file}"; then
                    echo "  ✓ Task ${task} completed successfully"
                else
                    echo "  ✗ Task ${task} may have failed or is incomplete"
                    all_success=false
                fi
            fi
        done
        
        if [ "${all_success}" = true ]; then
            echo ""
            echo "============================================================================"
            echo "✓ ALL TASKS COMPLETED SUCCESSFULLY!"
            echo "============================================================================"
            exit 0
        else
            echo ""
            echo "Some tasks may have failed. Check logs for details."
            exit 1
        fi
    fi
    
    echo "Status: ${status}"
    
    # Check each task
    needs_optimization=false
    max_remaining_hours=0
    
    for task in 0 1 2; do
        task_job="${JOB_ID}_${task}"
        task_status=$(check_job_status "${task_job}")
        
        if [ "${task_status}" != "COMPLETED" ] && [ -n "${task_status}" ]; then
            echo ""
            echo "Task ${task}:"
            log_file="${PROJECT_DIR}/logs/train_ultra_${JOB_ID}_${task}.out"
            
            if [ -f "${log_file}" ]; then
                # Show current phase
                current_phase=$(grep -E "STEP [0-9]:" "${log_file}" | tail -1 | sed 's/.*STEP \([0-9]\):.*/\1/')
                if [ -n "${current_phase}" ]; then
                    case "${current_phase}" in
                        1) echo "  Phase: SAE Training" ;;
                        2) echo "  Phase: Neural-ODE Training" ;;
                        3) echo "  Phase: Optimal Transport" ;;
                        *) echo "  Phase: Initialization" ;;
                    esac
                fi
                
                # Show progress if pre-loading
                if tail -20 "${log_file}" | grep -q "Pre-loading"; then
                    progress=$(grep -E "\[.*/[0-9]+\]" "${log_file}" | tail -1)
                    if [ -n "${progress}" ]; then
                        echo "  Progress: ${progress}"
                    fi
                fi
                
                # Estimate remaining time
                remaining=$(estimate_completion "${task}")
                if [ "${remaining}" != "N/A" ]; then
                    remaining_float=$(echo "${remaining}" | bc 2>/dev/null)
                    if [ -n "${remaining_float}" ]; then
                        remaining_int=$(echo "scale=0; ${remaining_float}/1" | bc 2>/dev/null)
                        echo "  Estimated remaining: ~${remaining_float} hours"
                        
                        # Check if exceeds max runtime
                        if [ -n "${remaining_int}" ] && [ "${remaining_int}" -gt "${MAX_RUNTIME_HOURS}" ]; then
                            needs_optimization=true
                            if [ "${remaining_int}" -gt "${max_remaining_hours}" ]; then
                                max_remaining_hours=${remaining_int}
                            fi
                        fi
                    fi
                fi
            else
                echo "  Status: Waiting to start (no log file yet)"
            fi
        fi
    done
    
    # Optimize if needed
    if [ "${needs_optimization}" = true ]; then
        echo ""
        echo "⚠️  WARNING: Estimated completion > ${MAX_RUNTIME_HOURS} hours"
        echo "   Max remaining: ${max_remaining_hours} hours"
        echo "   Current optimization count: ${OPTIMIZATION_COUNT}/${MAX_OPTIMIZATION_ATTEMPTS}"
        echo ""
        
        # Check counter for this job (in case it was resubmitted)
        if [ -f "${PROJECT_DIR}/.optimization_counter_${JOB_ID}" ]; then
            OPTIMIZATION_COUNT=$(cat "${PROJECT_DIR}/.optimization_counter_${JOB_ID}")
        fi
        
        if [ "${OPTIMIZATION_COUNT}" -ge "${MAX_OPTIMIZATION_ATTEMPTS}" ]; then
            echo "❌ Maximum optimization attempts (${MAX_OPTIMIZATION_ATTEMPTS}) reached."
            echo "Cancelling job ${JOB_ID}..."
            scancel "${JOB_ID}"
            echo ""
            echo "Job cancelled. Please investigate the root cause of the slowdown."
            exit 1
        fi
        
        echo "Optimizing and resubmitting..."
        optimize_and_resubmit "${JOB_ID}" "Estimated completion (${max_remaining_hours}h) exceeds limit (${MAX_RUNTIME_HOURS}h)"
        JOB_ID=$(cat "${PROJECT_DIR}/.current_monitored_job" 2>/dev/null || echo "${JOB_ID}")
        
        # Reload counter for new job
        if [ -f "${PROJECT_DIR}/.optimization_counter_${JOB_ID}" ]; then
            OPTIMIZATION_COUNT=$(cat "${PROJECT_DIR}/.optimization_counter_${JOB_ID}")
        else
            OPTIMIZATION_COUNT=0
            echo "0" > "${PROJECT_DIR}/.optimization_counter_${JOB_ID}"
        fi
        
        echo "Now monitoring job: ${JOB_ID} (optimization count: ${OPTIMIZATION_COUNT})"
    fi
    
    echo ""
    echo "Next check in ${CHECK_INTERVAL} seconds (5 minutes)..."
    sleep "${CHECK_INTERVAL}"
done

