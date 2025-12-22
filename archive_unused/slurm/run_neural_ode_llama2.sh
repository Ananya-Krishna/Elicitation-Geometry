#!/bin/bash
# Quick script to run Neural-ODE for llama2 (interactive or SLURM)

PROJECT_DIR="/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry"
cd "${PROJECT_DIR}"

PYTHON_CMD="${HOME}/.conda/envs/elicitation/bin/python"

echo "Training Neural-ODE for llama2..."
${PYTHON_CMD} model/train_neural_ode.py \
    --sae_dir sae_output_ultra \
    --data_dir "${PROJECT_DIR}" \
    --model_family llama2 \
    --output_dir ode_output_ultra \
    --epochs 15 \
    --batch_size 256 \
    --lr 1e-4 \
    --time_dependent

echo "Done!"
