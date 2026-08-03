# This service uses httpx + pdfplumber ONLY. Do not install torch or paddle.
FROM python:3.11-slim

WORKDIR /app

COPY services/identity/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/common /app/services/common
COPY services/identity /app/services/identity
COPY config.yaml /app/config.yaml
RUN printf '' > /app/services/__init__.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8004
CMD ["python", "-m", "uvicorn", "services.identity.app:app", "--host", "0.0.0.0", "--port", "8004"]
