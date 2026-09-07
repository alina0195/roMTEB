#!/bin/bash
# Re-run Retrieval + RoNLI for models whose query prompts were wrong in
# the legacy v1 snapshot (docs/task_audit.md, §1.3).
# Nemotron skipped (Sep 2026): too slow at batch=1; excluded from plots.
#
#   sbatch scripts/sbatch_prompt_rerun.sh
#
# embeddinggemma / nemotron: loader=auto → MTEB wrapper (PREFER_MTEB_LOADER).
# bge-multilingual-gemma2: loader=st so MODEL_PROMPTS query instruction applies.
#
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=prompt-rerun
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-prompt-rerun-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-prompt-rerun-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L
nvidia-smi

RUNNER="$WS/apptainer-exec-romteb.sh"
TASKS="WebFAQRetrieval WikipediaRetrievalMultilingual XQuADRetrieval RoDTALLawsRetrieval MQARoCQARetrieval RoNLIPairClassification"

run() {
  local model="$1"
  shift
  echo
  echo "=== ${model}  $*  tasks=${TASKS} ==="
  "$RUNNER" romteb/run_benchmark.py \
    --model "${model}" \
    --output results \
    --overwrite_results \
    --tasks ${TASKS} \
    "$@"
}

# Native ST / MTEB task-type prompts.
run "google/embeddinggemma-300m" --loader auto

# Query instruction from MODEL_PROMPTS (MTEB registry has use_instructions=False).
run "BAAI/bge-multilingual-gemma2" --loader st

# Not run: nvidia/llama-embed-nemotron-8b (batch=1 encode of MQA ~216k).
# Job 259996 is cancelled once gemma2 MQA finishes.

echo "Done prompt rerun at $(date)"
