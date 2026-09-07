#!/bin/bash
# Smoke E5 on the new tasks. Always GPU (H200).
#
#   sbatch scripts/sbatch_smoke_e5.sh
#
# Logs: logs/slurm-smoke-e5-<jobid>.out
#SBATCH -A hria
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH -c 10
#SBATCH --mem-per-cpu=27G
#SBATCH --job-name=smoke-e5
#SBATCH --output=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-smoke-e5-%j.out
#SBATCH --error=/export/projects/nlp/users/alina_gheorghe/roMTEB/logs/slurm-smoke-e5-%j.out

set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
cd "$WS"
mkdir -p "$WS/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L
nvidia-smi

./apptainer-exec-romteb.sh romteb/run_benchmark.py \
  --model intfloat/multilingual-e5-large \
  --loader auto \
  --tasks RoABSAClassification HistNERoMentionClassification \
          SaRoCoClassification SciTechBanROClassification \
          RoMathDomainClassification \
          GrileGrammarReranking JuRoLegalExamReranking \
          WWTBMRoQAReranking RoMedQAv2Reranking \
          MQARoCQARetrieval RoDTALLawsRetrieval \
  --output results/_smoke_new

echo "Done at $(date)"
