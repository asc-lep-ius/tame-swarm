# We use a PyTorch image with CUDA 12.1 support which matches your 4060 nicely
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system basics
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# - transformer_lens: The key tool for "Steering Vectors" (Module 2)
# - bitsandbytes: For 8-bit quantization (saves VRAM)
# - accelerate: Optimizes inference on your single GPU
RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.39.0" \
    "accelerate>=0.28.0" \
    accelerate \
    bitsandbytes \
    transformer_lens \
    fastapi \
    uvicorn \
    pydantic \
    protobuf \
    sentencepiece \
    einops

# Copy our application code
COPY . .

# Launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]