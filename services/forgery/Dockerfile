# Forgery: PyTorch with Blackwell (sm_120) support via CUDA 12.8 wheels.
# Host must pass a GPU via compose `gpus: all`.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE=torch==2.7.1
ARG TORCHVISION_PACKAGE=torchvision==0.22.1

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks system Python as externally managed (PEP 668).
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir \
        --index-url "${TORCH_INDEX}" \
        "${TORCH_PACKAGE}" \
        "${TORCHVISION_PACKAGE}" \
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
