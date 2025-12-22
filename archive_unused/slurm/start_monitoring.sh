#!/bin/bash
# Start monitoring in background and log output

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
JOB_ID="${1:-3422245}"

cd "${PROJECT_DIR}"

echo "Starting monitoring for job ${JOB_ID}..."
echo "Monitor will run in background and log to: ${PROJECT_DIR}/logs/monitor_${JOB_ID}.log"

nohup bash "${PROJECT_DIR}/model/monitor_and_optimize.sh" "${JOB_ID}" > "${PROJECT_DIR}/logs/monitor_${JOB_ID}.log" 2>&1 &

MONITOR_PID=$!
echo "${MONITOR_PID}" > "${PROJECT_DIR}/.monitor_pid_${JOB_ID}"

echo "Monitor started (PID: ${MONITOR_PID})"
echo "To stop monitoring: kill \$(cat ${PROJECT_DIR}/.monitor_pid_${JOB_ID})"
echo "To view logs: tail -f ${PROJECT_DIR}/logs/monitor_${JOB_ID}.log"



