# Multi-stage build with `serve` and `train` targets.
#
# The split follows the dependency boundary #3 draws: the inference/serve path
# never imports `datasets` (contrastive pairs load lazily, and only the built-in
# templates are needed at server start), so the serve image carries neither
# `datasets` nor `peft`/`mlflow`. The train image is the superset and stays the
# DEFAULT target -- a bare `docker build` builds the last stage -- so the CI
# docker-build and train-smoke jobs keep working unchanged.
#
# Contrastive A/B datasets (truthful_qa, Anthropic/hh-rlhf) are NOT baked in.
# They live in the mounted HuggingFace cache volume shared across the dev, train
# and validation containers, warmed by scripts/warm_hf_datasets.py -- which keeps
# the image lean and avoids tying an image build to HuggingFace availability.

# ── Base: CUDA runtime, Python, torch, and the core inference deps ──────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip

# PyTorch 2.10 stable with CUDA 13.0 wheels (Blackwell sm_120 support).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

# Core ML deps that every path imports.
RUN pip install --no-cache-dir \
    "transformers>=4.44.0" \
    accelerate \
    sentencepiece \
    numpy

COPY . .

# ── serve: FastAPI inference API. No datasets/peft/mlflow. ──────────────────
FROM base AS serve

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── train: the full workflow image, and the DEFAULT target (built last). ────
# Superset of serve, plus the training and dataset deps and the chat UI deps, so
# it matches the pre-split single image the CI train jobs were built against.
FROM base AS train

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    "datasets>=2.14.0" \
    "peft>=0.7.0" \
    "tqdm>=4.66.0" \
    "mlflow>=2.10.0" \
    gradio \
    requests

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
