# This service uses PaddleOCR / PaddlePaddle ONLY. Do not install torch.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ocr/requirements.txt /tmp/requirements.txt
# Prefer GPU paddle when the host passes --gpus.
# Blackwell (sm_120 / RTX 50 / RTX PRO 6000) needs cu129+ wheels — not cu126/3.2.1.
ARG PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu129/
ARG PADDLE_PACKAGE=paddlepaddle-gpu==3.3.0
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir ${PADDLE_PACKAGE} -i ${PADDLE_INDEX} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/common /app/services/common
COPY services/ocr /app/services/ocr
COPY config.yaml /app/config.yaml
RUN printf '' > /app/services/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "services.ocr.app:app", "--host", "0.0.0.0", "--port", "8001"]
