#!/bin/bash
# Retry RoNLI on the rebalanced pin for models that failed in 260015:
# Qwen / e5-large-instruct (MTEB wrapper KeyError on custom task name)
# plus jina (flash-attn import). Loader=st; jina gets --no_flash_attn.
#
#   sbatch scripts/sbatch_ronli_missing.sh
#
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=ronli-miss
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-ronli-missing-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-ronli-missing-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L

RUNNER="$WS/apptainer-exec-romteb.sh"

run() {
  local model="$1"
  local loader="$2"
  shift 2
  echo
  echo "=== RoNLI  ${model}  loader=${loader} $* ==="
  "$RUNNER" romteb/run_benchmark.py \
    --model "${model}" \
    --output results \
    --overwrite_results \
    --loader "${loader}" \
    --tasks RoNLIPairClassification \
    "$@" || echo "[WARN] ${model} RoNLI failed; continuing."
}

run "Qwen/Qwen3-Embedding-0.6B" st
run "intfloat/multilingual-e5-large-instruct" st
run "Qwen/Qwen3-Embedding-8B" st
# jina has no results/ tree yet — a lone RoNLI JSON would drop categories
# from Overall for every model. Do not eval it here.

echo "Done RoNLI missing at $(date)"
