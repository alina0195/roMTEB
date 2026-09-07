#!/bin/bash
# After bge-multilingual-gemma2 MQA JSON is overwritten, cancel 259996
# so nemotron never starts, then submit post-eval (after RoNLI if given).
#
#   bash scripts/skip_nemotron_then_report.sh <prompt_jobid> [ronli_jobid]
set -euo pipefail

WS=/export/projects/nlp/users/alina_gheorghe/roMTEB
PROMPT_JOB=${1:?prompt job id}
RONLI_JOB=${2:-}
MQA="$WS/results/BAAI__bge-multilingual-gemma2/8365500e19bc804136f4ad12c2494430239a2d72/MQARoCQARetrieval.json"
DTAL="$WS/results/BAAI__bge-multilingual-gemma2/8365500e19bc804136f4ad12c2494430239a2d72/RoDTALLawsRetrieval.json"
LOG="$WS/logs/skip-nemotron-watch.log"
mkdir -p "$WS/logs"

echo "$(date) watching MQA vs RoDTAL; will scancel ${PROMPT_JOB}" | tee -a "$LOG"

while true; do
  if [[ ! -f "$MQA" || ! -f "$DTAL" ]]; then
    echo "$(date) missing json, retry" | tee -a "$LOG"
    sleep 20
    continue
  fi
  mqa=$(stat -c %Y "$MQA")
  dtal=$(stat -c %Y "$DTAL")
  if (( mqa >= dtal )); then
    echo "$(date) MQA mtime=${mqa} >= RoDTAL ${dtal}; draining then scancel ${PROMPT_JOB}" | tee -a "$LOG"
    sleep 25
    scancel "$PROMPT_JOB" && echo "$(date) scancel ${PROMPT_JOB} ok" | tee -a "$LOG"
    break
  fi
  sleep 20
done

cd "$WS"
if [[ -n "$RONLI_JOB" ]]; then
  echo "$(date) sbatch post_eval afterok:${RONLI_JOB}" | tee -a "$LOG"
  sbatch --dependency=afterok:"$RONLI_JOB" "$WS/scripts/sbatch_post_eval.sh" | tee -a "$LOG"
else
  echo "$(date) sbatch post_eval immediately" | tee -a "$LOG"
  sbatch "$WS/scripts/sbatch_post_eval.sh" | tee -a "$LOG"
fi
echo "$(date) watcher done" | tee -a "$LOG"
