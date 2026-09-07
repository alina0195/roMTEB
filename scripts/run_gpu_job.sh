#!/bin/bash
# Run any RoMTEB command on one H200 GPU. Do not use this on the login node
# (fep*) without sbatch — there is no GPU and dense encodes look hung.
#
#   sbatch --job-name=smoke-e5 scripts/run_gpu_job.sh \
#     romteb/run_benchmark.py --model intfloat/multilingual-e5-large --loader auto \
#     --tasks RoSTS --output results/_smoke_one
#
#   sbatch --job-name=romteb scripts/run_gpu_job.sh bash scripts/run_all_models.sh results
#
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=romteb-gpu
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-gpu-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-gpu-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch scripts/run_gpu_job.sh <command> [args...]" >&2
  echo "  sbatch scripts/run_gpu_job.sh romteb/run_benchmark.py --model ..." >&2
  echo "  sbatch scripts/run_gpu_job.sh bash scripts/run_all_models.sh results" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
if ! nvidia-smi -L; then
  echo "ERROR: no GPU on $(hostname). Submit with sbatch --gres=gpu:1 --partition=h200." >&2
  exit 1
fi
nvidia-smi

RUNNER="$WS/apptainer-exec-romteb.sh"

# bash scripts/... is passed through; *.py / other commands go through the SIF.
if [ "$1" = "bash" ]; then
  echo "=== $* ==="
  "$@"
else
  echo "=== apptainer $* ==="
  "$RUNNER" "$@"
fi

echo "Done at $(date)"
