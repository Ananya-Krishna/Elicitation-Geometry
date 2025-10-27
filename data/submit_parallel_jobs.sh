#!/bin/bash

echo "🚀 Submitting PARALLEL DATA-SPLIT jobs for all 6 models..."
echo "Each job processes 1 model with 1/6th of the data"
echo "Expected completion time: ~1.5-2 hours (all running in parallel)"
echo ""

# Cancel any existing jobs first
echo "Cancelling existing jobs..."
scancel -u $USER

# Wait a moment for cancellation
sleep 5

# Submit all 6 parallel jobs
echo "Submitting parallel jobs..."

echo "1. Submitting llama2_base job..."
JOB1=$(sbatch data/run_parallel_llama2_base.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"

echo "2. Submitting llama2_aligned job..."
JOB2=$(sbatch data/run_parallel_llama2_aligned.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"

echo "3. Submitting falcon_base job..."
JOB3=$(sbatch data/run_parallel_falcon_base.slurm | awk '{print $4}')
echo "   Job ID: $JOB3"

echo "4. Submitting falcon_aligned job..."
JOB4=$(sbatch data/run_parallel_falcon_aligned.slurm | awk '{print $4}')
echo "   Job ID: $JOB4"

echo "5. Submitting mistral_base job..."
JOB5=$(sbatch data/run_parallel_mistral_base.slurm | awk '{print $4}')
echo "   Job ID: $JOB5"

echo "6. Submitting mistral_aligned job..."
JOB6=$(sbatch data/run_parallel_mistral_aligned.slurm | awk '{print $4}')
echo "   Job ID: $JOB6"

echo ""
echo "✅ All 6 parallel jobs submitted!"
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5, $JOB6"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "Watch logs:"
echo "  tail -f data/logs/parallel_llama2_base_$JOB1.out"
echo "  tail -f data/logs/parallel_llama2_aligned_$JOB2.out"
echo "  tail -f data/logs/parallel_falcon_base_$JOB3.out"
echo "  tail -f data/logs/parallel_falcon_aligned_$JOB4.out"
echo "  tail -f data/logs/parallel_mistral_base_$JOB5.out"
echo "  tail -f data/logs/parallel_mistral_aligned_$JOB6.out"
echo ""
echo "Expected completion: ~1.5-2 hours"
echo "After completion, run: python merge_parallel_data.py"
