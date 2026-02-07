# PyTorch with CUDA 12.1 runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system basics
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install ML dependencies
# Using recent transformers for Mistral support
RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.36.0" \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    numpy

# Copy our TAME architecture code
COPY . .

# Launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]