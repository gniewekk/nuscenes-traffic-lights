# CUDA runtime + PyTorch — YOLO detect uses the GPU when started with --gpus / compose gpus:
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # FiftyOne: do not try to bundle mongod; compose sets FIFTYONE_DATABASE_URI
    FIFTYONE_DATABASE_VALIDATION=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY traffic_lights/ traffic_lights/
COPY Makefile .

# Default: full pipeline without UI (override in compose)
CMD ["make", "PYTHON=python", "pipeline"]
