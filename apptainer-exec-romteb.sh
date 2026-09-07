#!/bin/bash
#
# RoMTEB runner (no-rebuild path).
#
# Canonical writable tree (code + results + logs + tmp + caches):
#   /export/projects/nlp/users/alina_gheorghe/roMTEB
#
# Container $HOME, TMPDIR, and Python/torch caches are forced onto that
# tree so jobs never write into quota-limited /home or $HOME.
#
# One-time setup:
#   bash /export/projects/nlp/users/alina_gheorghe/roMTEB/scripts/install_romteb_libs.sh
#
# Use (from anywhere):
#   /export/projects/nlp/users/alina_gheorghe/roMTEB/apptainer-exec-romteb.sh romteb/scripts/smoke_test.py
#   /export/projects/nlp/users/alina_gheorghe/roMTEB/apptainer-exec-romteb.sh romteb/romteb/run_benchmark.py --model intfloat/multilingual-e5-large
#
# Override default paths via env vars:
#   ROMTEB_ROOT  -- project tree (default: path above)
#   ROMTEB_SIF   -- image path
#   ROMTEB_LIBS  -- host-side dir holding mteb
#   HF_HOME      -- Hugging Face cache (default: shared /export/projects/nlp/.cache)

set -euo pipefail

echo "Welcome Alina - RoMTEB!"

ROMTEB_ROOT="${ROMTEB_ROOT:-/export/projects/nlp/users/alina_gheorghe/roMTEB}"
CONTAINER_HOME="${ROMTEB_ROOT}/container_home"

mkdir -p \
  "${ROMTEB_ROOT}/tmp/apptainer" \
  "${ROMTEB_ROOT}/logs" \
  "${ROMTEB_ROOT}/cache/torch" \
  "${ROMTEB_ROOT}/cache/matplotlib" \
  "${ROMTEB_ROOT}/cache/triton" \
  "${ROMTEB_ROOT}/cache/numba" \
  "${ROMTEB_ROOT}/cache/pycache" \
  "${ROMTEB_ROOT}/cache/apptainer" \
  "${CONTAINER_HOME}/.config" \
  "${ROMTEB_ROOT}/results" \
  "${ROMTEB_ROOT}/results_new"

# Redirect every well-known write location off $HOME / /home.
export HOME="${CONTAINER_HOME}"
export TMPDIR="${ROMTEB_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${ROMTEB_ROOT}/cache"
export XDG_CONFIG_HOME="${CONTAINER_HOME}/.config"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
# Keep Apptainer scratch on local /tmp when possible; fall back to project storage.
if [ -z "${APPTAINER_TMPDIR:-}" ]; then
  if mkdir -p "/tmp/${USER:-alina.gheorghe2505}/apptainer" 2>/dev/null; then
    export APPTAINER_TMPDIR="/tmp/${USER:-alina.gheorghe2505}/apptainer"
  else
    export APPTAINER_TMPDIR="${ROMTEB_ROOT}/tmp/apptainer"
  fi
fi
export APPTAINER_CACHEDIR="${XDG_CACHE_HOME}/apptainer"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/export/projects/nlp/.cache}"
# Tokens come from the environment. Do not hardcode them in this script.
: "${HF_TOKEN:=}"
: "${WANDB_API_KEY:=}"
export HF_TOKEN WANDB_API_KEY

ROMTEB_SIF="${ROMTEB_SIF:-/export/home/proiecte/aux/alina.gheorghe2505/transformers_flash_attn.sif}"
ROMTEB_LIBS="${ROMTEB_LIBS:-${ROMTEB_ROOT}/pylibs}"
if [ ! -d "${ROMTEB_LIBS}" ] || [ -z "$(ls -A "${ROMTEB_LIBS}" 2>/dev/null || true)" ]; then
  ROMTEB_LIBS=/export/home/proiecte/aux/alina.gheorghe2505/romteb_pylibs
fi

# Python package lives at ${ROMTEB_ROOT}/romteb (this directory is on PYTHONPATH).
ROMTEB_SRC="${ROMTEB_ROOT}"
ENCODER_DIR=/export/home/proiecte/aux/alina.gheorghe2505/tmp/encoder

if [ ! -f "${ROMTEB_SIF}" ]; then
  echo "ERROR: image not found at ${ROMTEB_SIF}" >&2
  exit 2
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 <python_script.py [args...] | command [args...]>" >&2
  exit 1
fi

# Historical docs use romteb/scripts/...; the files live in scripts/.
if [[ "$1" == romteb/scripts/* ]]; then
  _alt="scripts/${1#romteb/scripts/}"
  if [ ! -f "${ROMTEB_ROOT}/$1" ] && [ -f "${ROMTEB_ROOT}/${_alt}" ]; then
    echo "[romteb] ${_alt}  (romteb/scripts/ → scripts/)"
    set -- "${_alt}" "${@:2}"
  fi
fi

APPTAINER_OPTS=(
  --nv
  --home "${CONTAINER_HOME}"
  --pwd "${ROMTEB_ROOT}"
  -B /export/projects/nlp:/export/projects/nlp
  -B /export/home:/export/home
  -B "${ROMTEB_ROOT}:${ROMTEB_ROOT}"
  --env "TMPDIR=${TMPDIR}"
  --env "TMP=${TMPDIR}"
  --env "TEMP=${TMPDIR}"
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  --env "XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
  --env "TORCH_HOME=${TORCH_HOME}"
  --env "MPLCONFIGDIR=${MPLCONFIGDIR}"
  --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  --env "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
  --env "HF_HOME=${HF_HOME}"
)

# Local dataset trees (readable from home; writes still go to ROMTEB_ROOT).
if [ -d "${ROMTEB_ROOT}/tmp_graf_repo" ]; then
  APPTAINER_OPTS+=(-B "${ROMTEB_ROOT}/tmp_graf_repo:${ROMTEB_ROOT}/tmp_graf_repo")
elif [ -d "${ENCODER_DIR}/tmp_graf_repo" ]; then
  APPTAINER_OPTS+=(-B "${ENCODER_DIR}/tmp_graf_repo:${ENCODER_DIR}/tmp_graf_repo")
fi
if [ -d "${ROMTEB_ROOT}/datasets" ]; then
  APPTAINER_OPTS+=(-B "${ROMTEB_ROOT}/datasets:${ROMTEB_ROOT}/datasets")
elif [ -d "${ENCODER_DIR}/datasets" ]; then
  APPTAINER_OPTS+=(-B "${ENCODER_DIR}/datasets:${ENCODER_DIR}/datasets")
fi

if [ -d "${ROMTEB_LIBS}" ] && [ -n "$(ls -A "${ROMTEB_LIBS}" 2>/dev/null || true)" ]; then
  APPTAINER_OPTS+=(
    -B "${ROMTEB_LIBS}:/opt/romteb_libs"
    --env "PYTHONPATH=${ROMTEB_SRC}:/opt/romteb_libs:${PYTHONPATH:-}"
  )
else
  echo "Note: ${ROMTEB_LIBS} missing or empty -- assuming mteb is baked into the SIF." >&2
  echo "      If imports fail, run: bash ${ROMTEB_ROOT}/scripts/install_romteb_libs.sh" >&2
  APPTAINER_OPTS+=(--env "PYTHONPATH=${ROMTEB_SRC}:${PYTHONPATH:-}")
fi

echo "[romteb] ROOT=${ROMTEB_ROOT}  HOME=${HOME}  HF_HOME=${HF_HOME}  TMPDIR=${TMPDIR}"

if [[ "$1" == *.py ]]; then
  apptainer exec "${APPTAINER_OPTS[@]}" "${ROMTEB_SIF}" python "$@"
else
  apptainer exec "${APPTAINER_OPTS[@]}" "${ROMTEB_SIF}" "$@"
fi
