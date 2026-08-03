# This service uses OpenCV (YuNet + SFace) ONLY. Do not install torch or paddle.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/face/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/common /app/services/common
COPY services/face /app/services/face
COPY config.yaml /app/config.yaml
RUN printf '' > /app/services/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "services.face.app:app", "--host", "0.0.0.0", "--port", "8002"]
