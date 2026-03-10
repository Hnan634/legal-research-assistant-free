# server/Dockerfile — slim, CPU-only wheel, faster builds
FROM python:3.10-slim

# Keep image lean, non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed for some Python packages (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch first from PyTorch CPU index (speeds up and avoids CUDA wheels)
# The --index-url points to CPU wheels; pip will find a suitable CPU-only torch.
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app files
COPY . /app

# Avoid storing vector store or caches in the image
RUN mkdir -p /app/vector_store

# Expose a default port (Render provides $PORT at runtime)
EXPOSE 3000

# Use sh -c to allow $PORT expansion at runtime
CMD ["sh", "-c", "echo Binding to port $PORT && gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 1"]
