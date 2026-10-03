FROM python:3.14.3-slim-bookworm@sha256:c6f0b5b3a167963de3cc7cd97fe1a5d07105c8d27f47507ab31d648b533b05c7
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OLLAMA_HOST=http://127.0.0.1:11434 \
    CLOUDRAG_MODE=participant CLOUDRAG_DEMO_GPU=0
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 build-essential \
    && git config --system --add safe.directory /opt/cloudrag/repository
WORKDIR /opt/cloudrag/repository
COPY requirements-lock.txt /opt/cloudrag/requirements-lock.txt
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r /opt/cloudrag/requirements-lock.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/cloudrag/linux-installed.txt
COPY . .
ENTRYPOINT ["python", "scripts/cloud_entrypoint.py"]
