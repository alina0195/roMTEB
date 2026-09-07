#!/bin/bash
# Evaluate RoSTS on every dense model that already has a results/ tree.
# Do not add models with no other tasks (lone STS JSON would drop categories
# from Overall for everyone).
#
#   sbatch scripts/sbatch_rosts.sh
#
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=rosts-eval
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-rosts-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-rosts-%j.out

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
  echo "=== RoSTS  ${model}  loader=${loader} $* ==="
  "$RUNNER" romteb/run_benchmark.py \
    --model "${model}" \
    --output results \
    --overwrite_results \
    --loader "${loader}" \
    --tasks RoSTS \
    "$@" || echo "[WARN] ${model} RoSTS failed; continuing."
}

run "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" st
run "Alibaba-NLP/gte-multilingual-base" st --trust_remote_code
run "ibm-granite/granite-embedding-311m-multilingual-r2" st --no_flash_attn
run "intfloat/multilingual-e5-small" auto
run "intfloat/multilingual-e5-base" auto
run "google/embeddinggemma-300m" auto
# Instruct wrappers KeyError on custom task names — ST loader.
run "Qwen/Qwen3-Embedding-0.6B" st
run "nomic-ai/nomic-embed-text-v2-moe" st --trust_remote_code
run "intfloat/multilingual-e5-large-instruct" st
run "Qwen/Qwen3-Embedding-8B" st
run "BAAI/bge-multilingual-gemma2" st
run "BAAI/bge-m3" auto
run "alina0195/bge-m3-ro-msmarco-v2" auto
run "intfloat/multilingual-e5-large" auto
run "tencent/KaLM-Embedding-Gemma3-12B-2511" st --batch_size 1
# Smoke the transformers pin fix; retrieval prompt-rerun still skipped.
run "nvidia/llama-embed-nemotron-8b" auto --batch_size 1

echo "Done RoSTS at $(date)"
