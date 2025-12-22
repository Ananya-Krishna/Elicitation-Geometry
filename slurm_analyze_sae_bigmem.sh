#!/bin/bash
#SBATCH --job-name=analyze_sae_bigmem
#SBATCH --output=logs/analyze_sae_bigmem_%j.out
#SBATCH --error=logs/analyze_sae_bigmem_%j.err
#SBATCH --partition=bigmem          # CPU-only, very large RAM
#SBATCH --time=04:00:00              # Adjust if you want a shorter cap
#SBATCH --cpus-per-task=32           # TSNE benefits from more threads
#SBATCH --mem=512G                   # Plenty of headroom for full 9k + TSNE

# Use -e + pipefail; avoid -u to sidestep unbound vars from system bashrc
set -eo pipefail

cd /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry

PYTHON="$HOME/.conda/envs/elicitation/bin/python"
echo "Using python: $PYTHON"

# Run with overrides: no cap (use full 9000) and enable TSNE on all samples.
# This avoids editing analyze_sae_outputs.py; we set constants before calling main().
$PYTHON - <<'PY'
import analyze_sae_outputs as m

# Override defaults at runtime
m.MAX_ANALYSIS_SAMPLES = 9000       # full dataset
m.TSNE_SUBSAMPLE = 9000             # use all for TSNE (will double to 18k base+aligned)

m.main()
PY

