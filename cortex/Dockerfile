# CUDA base for Blackwell (RTX 50-series) support
# Using 12.6 base image; PyTorch cu130 wheel bundles its own CUDA runtime
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10 and system basics
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

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch 2.10.0 stable with CUDA 13.0 (best Blackwell sm_120 support)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

# Install ML dependencies (only what's actually imported)
RUN pip install --no-cache-dir \
    "transformers>=4.44.0" \
    accelerate \
    sentencepiece

# Install training dependencies (optional but included for full workflow)
RUN pip install --no-cache-dir \
    "datasets>=2.14.0" \
    "peft>=0.7.0" \
    "tqdm>=4.66.0"

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    numpy

# Install Chat UI dependencies
RUN pip install --no-cache-dir \
    gradio \
    requests

# Copy our TAME architecture code
COPY . .

# Launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]