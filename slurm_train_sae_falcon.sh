#!/bin/bash
#SBATCH --job-name=sae_falcon
#SBATCH --output=logs/sae_falcon_%j.out
#SBATCH --error=logs/sae_falcon_%j.err

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

echo "Starting SAE training for falcon on Bouchet"
date
hostname
nvidia-smi || echo "nvidia-smi not available?"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN_FILE=/home/ark89/.cache/huggingface/token

BASE_DIR=/home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry
DATA_ROOT="${BASE_DIR}/data/numpy_activations"
OUT_ROOT="${BASE_DIR}/new_sae_outputs"

mkdir -p "${BASE_DIR}/logs"
mkdir -p "${OUT_ROOT}"

cd "${BASE_DIR}"

~/.conda/envs/elicitation/bin/python model/train_sae_numpy.py \
    --data_root "${DATA_ROOT}" \
    --model_family falcon \
    --output_root "${OUT_ROOT}" \
    --epochs 20 \
    --batch_size 512 \
    --lr 1e-3 \
    --hidden_dim 16384 \
    --l1_coef 1e-4 \
    --subsample 1.0 \
    --encode_subsample 1.0

echo "Finished SAE training for falcon"
date
