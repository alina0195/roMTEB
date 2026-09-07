# RoMTEB evaluation worker.
#
# Build:
#   docker build -t romteb-eval:0.1.0 .
#
# Run (GPU, one job, one model directory or HF id):
#   docker run --gpus all --rm \
#     -v "$MODEL_DIR:/model:ro" \
#     -v "$OUT:/out" \
#     -v "$HF_CACHE:/hf" \
#     -e HF_HOME=/hf \
#     -e HF_TOKEN \
#     romteb-eval:0.1.0 \
#       --model /model --preset full --output /out --job-id "$JOB_ID"
#
# Smoke (minutes):
#   docker run --gpus all --rm ... romteb-eval:0.1.0 \
#       --model /model --preset smoke --output /out
#
# Untrusted uploads: add --no_trust_remote_code
#
# Same CLI without Docker, on the existing cluster SIF:
#   ./apptainer-exec-romteb.sh python -m romteb --list-tasks
#   ./apptainer-exec-romteb.sh python -m romteb --model ... --preset smoke --output /out

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf \
    PYTHONPATH=/opt/romteb

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/romteb

COPY pyproject.toml README.md ./
COPY romteb ./romteb
COPY scripts ./scripts
COPY tasks ./tasks

RUN pip install --upgrade pip \
    && pip install -e . \
    && python -c "import romteb, mteb; print('romteb', romteb.__version__, 'mteb', mteb.__version__)"

WORKDIR /out
ENTRYPOINT ["romteb-eval"]
CMD ["--help"]
