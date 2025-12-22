#!/bin/bash
# Monitor ultra-fast training jobs and check if on pace for 2-hour completion
# Usage: ./monitor_ultra_fast.sh [JOBID]
# If JOBID not provided, will find the most recent ultra_fast job

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
LOG_DIR="${PROJECT_DIR}/logs"

# Find job ID if not provided
if [ -z "$1" ]; then
    JOBID=$(squeue -u $USER -o "%i %j" | grep "ultra_fast" | head -1 | awk '{print $1}' | cut -d'_' -f1)
    if [ -z "$JOBID" ]; then
        echo "❌ No ultra_fast job found running"
        echo "   Please provide job ID: ./monitor_ultra_fast.sh [JOBID]"
        exit 1
    fi
    echo "📊 Found job: ${JOBID}"
else
    JOBID=$1
fi

echo "============================================================================"
echo "MONITORING ULTRA-FAST TRAINING JOB: ${JOBID}"
echo "============================================================================"
echo "Checking every 10 minutes..."
echo "Press Ctrl+C to stop"
echo "============================================================================"
echo ""

while true; do
    clear
    echo "============================================================================"
    echo "ULTRA-FAST TRAINING PROGRESS CHECK - $(date '+%H:%M:%S')"
    echo "============================================================================"
    
    # Check job status
    JOB_STATUS=$(squeue -j ${JOBID}_0 -o "%T" -h 2>/dev/null | head -1)
    if [ -z "$JOB_STATUS" ]; then
        echo "❌ Job ${JOBID} not found (may have completed or failed)"
        break
    fi
    
    # Get job info
    JOB_INFO=$(scontrol show job ${JOBID}_0 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "❌ Could not get job info"
        sleep 10
        continue
    fi
    
    # Extract start time and runtime
    START_TIME=$(echo "$JOB_INFO" | grep "StartTime=" | sed 's/.*StartTime=\([^ ]*\).*/\1/')
    RUNTIME=$(echo "$JOB_INFO" | grep "RunTime=" | sed 's/.*RunTime=\([^ ]*\).*/\1/')
    STATE=$(echo "$JOB_INFO" | grep "JobState=" | sed 's/.*JobState=\([^ ]*\).*/\1/')
    
    echo "📊 Job Status: ${STATE}"
    echo "⏰ Runtime: ${RUNTIME}"
    echo "🕐 Started: ${START_TIME}"
    echo ""
    
    # Check all array tasks
    echo "📋 Array Tasks:"
    for task in 0 1 2; do
        TASK_JOB="${JOBID}_${task}"
        TASK_STATUS=$(squeue -j ${TASK_JOB} -o "%T" -h 2>/dev/null)
        
        if [ -n "$TASK_STATUS" ]; then
            # Check log file
            LOG_FILE="${LOG_DIR}/train_ultra_${TASK_JOB}.out"
            ERR_FILE="${LOG_DIR}/train_ultra_${TASK_JOB}.err"
            
            if [ -f "$LOG_FILE" ]; then
                # Determine phase
                if grep -q "ENCODING ACTIVATIONS" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Encoding (SAE done)"
                elif grep -q "Training Neural-ODE" "$LOG_FILE" 2>/dev/null || grep -q "NEURAL-ODE" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Neural-ODE"
                elif grep -q "Computing Optimal Transport" "$LOG_FILE" 2>/dev/null || grep -q "OPTIMAL TRANSPORT" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Optimal Transport"
                elif grep -q "Epoch" "$ERR_FILE" 2>/dev/null; then
                    # Extract epoch progress from error log (tqdm writes there)
                    LAST_EPOCH=$(grep "Epoch" "$ERR_FILE" 2>/dev/null | tail -1 | sed -n 's/.*Epoch \([0-9]*\)\/[0-9]*.*/\1/p')
                    if [ -n "$LAST_EPOCH" ]; then
                        PHASE="SAE Epoch ${LAST_EPOCH}/5"
                    else
                        PHASE="SAE Training"
                    fi
                else
                    PHASE="Initializing"
                fi
                
                # Get model family
                MODEL=$(grep "Model Family:" "$LOG_FILE" 2>/dev/null | head -1 | sed 's/.*Model Family: \([^ ]*\).*/\1/')
                if [ -z "$MODEL" ]; then
                    MODELS=("llama2" "falcon" "mistral")
                    MODEL=${MODELS[$task]}
                fi
                
                echo "  Task ${task} (${MODEL}): ${PHASE}"
            else
                echo "  Task ${task}: Waiting to start"
            fi
        else
            echo "  Task ${task}: Not running"
        fi
    done
    
    echo ""
    echo "============================================================================"
    echo "PACE CHECK (Target: 2 hours)"
    echo "============================================================================"
    
    # Parse runtime (format: HH:MM:SS or D-HH:MM:SS)
    if [[ "$RUNTIME" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        DAYS=${BASH_REMATCH[1]}
        HOURS=${BASH_REMATCH[2]}
        MINS=${BASH_REMATCH[3]}
        SECS=${BASH_REMATCH[4]}
        TOTAL_SECS=$((DAYS*86400 + HOURS*3600 + MINS*60 + SECS))
    elif [[ "$RUNTIME" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        HOURS=${BASH_REMATCH[1]}
        MINS=${BASH_REMATCH[2]}
        SECS=${BASH_REMATCH[3]}
        TOTAL_SECS=$((HOURS*3600 + MINS*60 + SECS))
    else
        TOTAL_SECS=0
    fi
    
    TARGET_SECS=7200  # 2 hours
    ELAPSED_PCT=$((TOTAL_SECS * 100 / TARGET_SECS))
    
    if [ $TOTAL_SECS -gt 0 ]; then
        # Estimate completion based on current progress
        # Check which phase we're in
        ESTIMATED_REMAINING=0
        
        # Try to estimate from logs
        for task in 0 1 2; do
            LOG_FILE="${LOG_DIR}/train_ultra_${JOBID}_${task}.out"
            ERR_FILE="${LOG_DIR}/train_ultra_${JOBID}_${task}.err"
            
            if [ -f "$ERR_FILE" ]; then
                # Check if SAE is done
                if grep -q "ENCODING ACTIVATIONS" "$LOG_FILE" 2>/dev/null; then
                    # SAE done, estimate Neural-ODE + OT
                    ESTIMATED_REMAINING=$((40*60 + 20*60))  # 60 min
                elif grep -q "Training Neural-ODE" "$LOG_FILE" 2>/dev/null; then
                    # In Neural-ODE, estimate remaining
                    ESTIMATED_REMAINING=$((30*60))  # 30 min
                elif grep -q "Epoch" "$ERR_FILE" 2>/dev/null; then
                    # In SAE, estimate based on epochs
                    LAST_EPOCH=$(grep "Epoch" "$ERR_FILE" 2>/dev/null | tail -1 | sed -n 's/.*Epoch \([0-9]*\)\/5.*/\1/p')
                    if [ -n "$LAST_EPOCH" ] && [ "$LAST_EPOCH" -gt 0 ]; then
                        # Estimate: (5-epoch) * 6 min + Neural-ODE + OT
                        EPOCHS_REMAINING=$((5 - LAST_EPOCH))
                        ESTIMATED_REMAINING=$((EPOCHS_REMAINING*6*60 + 40*60 + 20*60))
                    else
                        ESTIMATED_REMAINING=$((30*60 + 40*60 + 20*60))  # 90 min
                    fi
                else
                    ESTIMATED_REMAINING=$((30*60 + 40*60 + 20*60))  # 90 min
                fi
                break
            fi
        done
        
        if [ $ESTIMATED_REMAINING -eq 0 ]; then
            ESTIMATED_REMAINING=$((30*60 + 40*60 + 20*60))  # Default: 90 min
        fi
        
        ESTIMATED_TOTAL=$((TOTAL_SECS + ESTIMATED_REMAINING))
        ESTIMATED_HOURS=$((ESTIMATED_TOTAL / 3600))
        ESTIMATED_MINS=$(((ESTIMATED_TOTAL % 3600) / 60))
        
        echo "⏱️  Elapsed: ${RUNTIME} (${ELAPSED_PCT}% of 2-hour target)"
        echo "📊 Estimated remaining: ~$((ESTIMATED_REMAINING / 60)) minutes"
        echo "🎯 Estimated total: ~${ESTIMATED_HOURS}h ${ESTIMATED_MINS}m"
        
        if [ $ESTIMATED_TOTAL -le $TARGET_SECS ]; then
            echo "✅ ON PACE for 2-hour completion!"
        else
            OVERRUN=$((ESTIMATED_TOTAL - TARGET_SECS))
            OVERRUN_MINS=$((OVERRUN / 60))
            echo "⚠️  May exceed 2 hours by ~${OVERRUN_MINS} minutes"
        fi
    else
        echo "⏱️  Just started, checking progress..."
    fi
    
    echo ""
    echo "📝 Recent log output (last 5 lines):"
    echo "---------------------------------------------------------------------------"
    for task in 0 1 2; do
        LOG_FILE="${LOG_DIR}/train_ultra_${JOBID}_${task}.out"
        if [ -f "$LOG_FILE" ]; then
            MODELS=("llama2" "falcon" "mistral")
            echo "Task ${task} (${MODELS[$task]}):"
            tail -3 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
        fi
    done
    echo "============================================================================"
    echo ""
    echo "Next check in 10 minutes... (Press Ctrl+C to stop)"
    
    sleep 600  # 10 minutes
done

