#!/bin/bash
# Aggregate + bootstrap + plots after the eval job. GPU is unused (CPU
# tables/figures) but the cluster requires a partition; use A100 as asked.
#
#   sbatch --dependency=afterok:259253 scripts/sbatch_post_eval.sh
#
#SBATCH -A phd
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=romteb-report
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-report-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-romteb-report-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L || echo "WARNING: no GPU listed (report pipeline is CPU-bound)"

bash "$WS/scripts/post_eval_report.sh" "$WS/results"

echo "Wrapper done at $(date)"
