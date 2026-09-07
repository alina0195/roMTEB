#!/bin/bash
# Copy the current roster into legacy/ (no retrievers) and regenerate plots.
# Submit after the eval job (and after the report job, so bootstrap CSV exists):
#
#   sbatch --dependency=afterok:259309 scripts/sbatch_legacy_plots.sh
#
#SBATCH -A phd
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=romteb-legacy
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-legacy-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-legacy-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L || echo "WARNING: no GPU listed (this job is CPU-bound)"

RUNNER="$WS/apptainer-exec-romteb.sh"
export PYTHONUNBUFFERED=1

echo "=== 1. Collect roster into legacy/ ==="
"$RUNNER" python -u "$WS/scripts/collect_legacy_results.py" --copy --out_dir "$WS/legacy"

if [ -f "$WS/results/romteb_ir_bootstrap.csv" ]; then
  cp -f "$WS/results/romteb_ir_bootstrap.csv" "$WS/legacy/romteb_ir_bootstrap.csv"
  echo "Copied results/romteb_ir_bootstrap.csv -> legacy/"
fi
if [ -f "$WS/results/romteb_ir_pairwise.csv" ]; then
  cp -f "$WS/results/romteb_ir_pairwise.csv" "$WS/legacy/romteb_ir_pairwise.csv"
fi

echo "=== 2. Aggregate + plots in legacy/ ==="
"$RUNNER" "$WS/scripts/make_plots.sh" "$WS/legacy"

echo "Plots: $WS/legacy/plots/"
echo "Done at $(date)"
