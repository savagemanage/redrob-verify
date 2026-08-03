# Forgery: PyTorch CUDA runtime (not python:3.11-slim — that has no NVIDIA libs).
# pip torch wheels still need the host to pass a GPU via compose `gpus: all`.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Base image already has torch/torchvision; install API deps only.
COPY services/forgery/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "python-multipart>=0.0.9" \
        "pydantic>=2.8.0" \
        "pyyaml>=6.0.2" \
        "numpy>=1.26.0,<2.3" \
        "opencv-python-headless>=4.10.0,<5" \
        "pillow>=10.0.0"

COPY services/common /app/services/common
COPY services/forgery /app/services/forgery
COPY config.yaml /app/config.yaml
RUN printf '' > /app/services/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8003
CMD ["python", "-m", "uvicorn", "services.forgery.app:app", "--host", "0.0.0.0", "--port", "8003"]
