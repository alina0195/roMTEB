#!/bin/bash
# Re-eval RoNLIPairClassification on the rebalanced test split (A1, Sep 2026)
# for every dense model except the three prompt-rerun models (they already
# overwrite RoNLI in sbatch_prompt_rerun.sh).
#
#   sbatch scripts/sbatch_ronli_rerun.sh
# Independent of the prompt-rerun (nemotron skipped).
#
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=ronli-rerun
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-ronli-rerun-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-ronli-rerun-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L

RUNNER="$WS/apptainer-exec-romteb.sh"

# model  loader  extra...
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

run "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" st
run "Alibaba-NLP/gte-multilingual-base" st --trust_remote_code
run "ibm-granite/granite-embedding-311m-multilingual-r2" st --no_flash_attn
run "intfloat/multilingual-e5-small" auto
run "intfloat/multilingual-e5-base" auto
# MTEB instruct wrappers KeyError on custom RoNLI task name — use ST.
run "Qwen/Qwen3-Embedding-0.6B" st
run "nomic-ai/nomic-embed-text-v2-moe" st --trust_remote_code
run "intfloat/multilingual-e5-large-instruct" st
run "Qwen/Qwen3-Embedding-8B" st
run "jinaai/jina-embeddings-v3" st --trust_remote_code --no_flash_attn
run "tencent/KaLM-Embedding-Gemma3-12B-2511" st --batch_size 1
# e5-large may already be in results/ from older roster runs
run "intfloat/multilingual-e5-large" auto
# Fine-tune vs BAAI/bge-m3 (base is covered by job 259999, which has no old RoNLI JSON).
run "alina0195/bge-m3-ro-msmarco-v2" auto

echo "Done RoNLI rerun at $(date)"
