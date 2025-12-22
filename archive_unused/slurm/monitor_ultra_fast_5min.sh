#!/bin/bash
# Monitor ultra-fast training job every 5 minutes
# Usage: ./monitor_ultra_fast_5min.sh [JOBID]

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
LOG_DIR="${PROJECT_DIR}/logs"

# Find job ID if not provided
if [ -z "$1" ]; then
    JOBID=$(squeue -u $USER -o "%i %j" | grep "ultra_fast" | head -1 | awk '{print $1}' | cut -d'_' -f1)
    if [ -z "$JOBID" ]; then
        echo "❌ No ultra_fast job found running"
        echo "   Please provide job ID: ./monitor_ultra_fast_5min.sh [JOBID]"
        exit 1
    fi
    echo "📊 Found job: ${JOBID}"
else
    JOBID=$1
fi

echo "============================================================================"
echo "MONITORING ULTRA-FAST TRAINING JOB: ${JOBID} (Every 5 minutes)"
echo "============================================================================"
echo "Press Ctrl+C to stop"
echo "============================================================================"
echo ""

while true; do
    clear
    echo "============================================================================"
    echo "ULTRA-FAST TRAINING PROGRESS - $(date '+%H:%M:%S')"
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
        sleep 300
        continue
    fi
    
    # Extract info
    START_TIME=$(echo "$JOB_INFO" | grep "StartTime=" | sed 's/.*StartTime=\([^ ]*\).*/\1/')
    RUNTIME=$(echo "$JOB_INFO" | grep "RunTime=" | sed 's/.*RunTime=\([^ ]*\).*/\1/')
    STATE=$(echo "$JOB_INFO" | grep "JobState=" | sed 's/.*JobState=\([^ ]*\).*/\1/')
    TIME_LIMIT=$(echo "$JOB_INFO" | grep "TimeLimit=" | sed 's/.*TimeLimit=\([^ ]*\).*/\1/')
    
    echo "📊 Job Status: ${STATE}"
    echo "⏰ Runtime: ${RUNTIME}"
    echo "🕐 Started: ${START_TIME}"
    echo "⏱️  Time Limit: ${TIME_LIMIT}"
    echo ""
    
    # Check all array tasks
    echo "📋 Array Tasks Progress:"
    for task in 0 1 2; do
        TASK_JOB="${JOBID}_${task}"
        TASK_STATUS=$(squeue -j ${TASK_JOB} -o "%T" -h 2>/dev/null)
        
        if [ -n "$TASK_STATUS" ]; then
            LOG_FILE="${LOG_DIR}/train_ultra_${TASK_JOB}.out"
            ERR_FILE="${LOG_DIR}/train_ultra_${TASK_JOB}.err"
            
            PHASE="Not started"
            EPOCH_PROGRESS=""
            BATCH_INFO=""
            
            if [ -f "$LOG_FILE" ]; then
                if grep -q "Complete pipeline finished" "$LOG_FILE" 2>/dev/null; then
                    PHASE="✅ COMPLETE"
                elif grep -q "Optimal Transport" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Optimal Transport"
                elif grep -q "Neural-ODE" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Neural-ODE Training"
                elif grep -q "ENCODING ACTIVATIONS" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Encoding (SAE done)"
                elif grep -q "Pre-loading ALL" "$LOG_FILE" 2>/dev/null; then
                    PHASE="Pre-loading data"
                elif grep -q "TRAINING" "$LOG_FILE" 2>/dev/null; then
                    PHASE="SAE Training"
                fi
            fi
            
            if [ -f "$ERR_FILE" ]; then
                # Extract epoch progress
                LAST_LINE=$(tail -1 "$ERR_FILE" 2>/dev/null)
                if echo "$LAST_LINE" | grep -q "Epoch"; then
                    EPOCH_NUM=$(echo "$LAST_LINE" | sed -n 's/.*Epoch \([0-9]*\)\/5.*/\1/p')
                    BATCH_NUM=$(echo "$LAST_LINE" | sed -n 's/.*\([0-9]*\)\/\([0-9]*\).*/\1\/\2/p' | head -1)
                    if [ -n "$EPOCH_NUM" ]; then
                        EPOCH_PROGRESS="Epoch ${EPOCH_NUM}/5"
                    fi
                    if [ -n "$BATCH_NUM" ]; then
                        BATCH_INFO="Batch ${BATCH_NUM}"
                    fi
                fi
            fi
            
            MODELS=("llama2" "falcon" "mistral")
            echo "  Task ${task} (${MODELS[$task]}): ${TASK_STATUS}"
            echo "    Phase: ${PHASE}"
            if [ -n "$EPOCH_PROGRESS" ]; then
                echo "    Progress: ${EPOCH_PROGRESS}"
            fi
            if [ -n "$BATCH_INFO" ]; then
                echo "    ${BATCH_INFO}"
            fi
        else
            MODELS=("llama2" "falcon" "mistral")
            echo "  Task ${task} (${MODELS[$task]}): Not running"
        fi
    done
    
    echo ""
    echo "============================================================================"
    echo "PACE CHECK"
    echo "============================================================================"
    
    # Parse runtime
    if [[ "$RUNTIME" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        DAYS=${BASH_REMATCH[1]}
        HOURS=${BASH_REMATCH[2]}
        MINS=${BASH_REMATCH[3]}
        TOTAL_SECS=$((DAYS*86400 + HOURS*3600 + MINS*60))
    elif [[ "$RUNTIME" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        HOURS=${BASH_REMATCH[1]}
        MINS=${BASH_REMATCH[2]}
        TOTAL_SECS=$((HOURS*3600 + MINS*60))
    else
        TOTAL_SECS=0
    fi
    
    if [ $TOTAL_SECS -gt 0 ]; then
        ELAPSED_MIN=$((TOTAL_SECS / 60))
        echo "⏱️  Elapsed: ${ELAPSED_MIN} minutes"
        
        # Estimate based on current phase
        if [ "$PHASE" = "✅ COMPLETE" ]; then
            echo "✅ JOB COMPLETE!"
        elif [ "$PHASE" = "Optimal Transport" ]; then
            echo "📊 Phase: Optimal Transport (final stage)"
            echo "   Estimated remaining: ~20 minutes"
        elif [ "$PHASE" = "Neural-ODE Training" ]; then
            echo "📊 Phase: Neural-ODE Training"
            echo "   Estimated remaining: ~40-60 minutes"
        elif [ "$PHASE" = "Encoding (SAE done)" ] || [ "$PHASE" = "SAE Training" ]; then
            if [ -n "$EPOCH_NUM" ]; then
                EPOCHS_REMAINING=$((5 - EPOCH_NUM))
                SAE_REMAINING=$((EPOCHS_REMAINING * 10))  # ~10 min per epoch
                ODE_TIME=40
                OT_TIME=20
                TOTAL_REMAINING=$((SAE_REMAINING + ODE_TIME + OT_TIME))
                echo "📊 Phase: SAE Training (Epoch ${EPOCH_NUM}/5)"
                echo "   Estimated remaining: ~${TOTAL_REMAINING} minutes"
            else
                echo "📊 Phase: SAE Training (starting)"
                echo "   Estimated remaining: ~90 minutes"
            fi
        elif [ "$PHASE" = "Pre-loading data" ]; then
            echo "📊 Phase: Pre-loading data"
            echo "   Estimated remaining: ~90 minutes"
        fi
    fi
    
    echo ""
    echo "📝 Recent log output:"
    echo "---------------------------------------------------------------------------"
    for task in 0 1 2; do
        LOG_FILE="${LOG_DIR}/train_ultra_${JOBID}_${task}.out"
        if [ -f "$LOG_FILE" ]; then
            MODELS=("llama2" "falcon" "mistral")
            echo "Task ${task} (${MODELS[$task]}):"
            tail -2 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
        fi
    done
    echo "============================================================================"
    echo ""
    echo "Next check in 5 minutes... (Press Ctrl+C to stop)"
    
    sleep 300  # 5 minutes
done

