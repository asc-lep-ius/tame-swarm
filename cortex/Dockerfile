# We use a PyTorch image with CUDA 12.1 support which matches your 4060 nicely
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system basics
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install transformer_lens first to avoid dependency conflicts
RUN pip install --no-cache-dir transformer_lens

# Install everything else - let pip resolve versions
# Only constraint: transformers must be recent for Qwen3 support
RUN pip install --no-cache-dir --upgrade \
    transformers \
    accelerate \
    bitsandbytes \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    tiktoken \
    einops

# Copy our application code
COPY . .

# Launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]