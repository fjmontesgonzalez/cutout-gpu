# syntax=docker/dockerfile:1.7
# Recorte de producto en GPU v3 para Alibaba Function Compute (custom container, puerto 9000).
# Sin libcuda.so en la imagen: la plataforma inyecta el driver (580.x, CUDA 13), compatible
# hacia atrás con el runtime CUDA 12.8 de las ruedas de torch (incluye sm_75 = Tesla T4).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/models \
    HF_HOME=/tmp/hf \
    HF_MODULES_CACHE=/tmp/hf_modules

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 \
    && pip uninstall -y triton

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Pesos dentro de la imagen. El token de Hugging Face (repos gated) llega como secreto de
# BuildKit y no queda en ninguna capa. CUTOUT_MODELS permite una imagen de producción más
# ligera (p.ej. --build-arg CUTOUT_MODELS=birefnet,sam3).
ARG CUTOUT_MODELS=birefnet,rmbg2,ben2,sam3
COPY download_models.py /app/download_models.py
RUN --mount=type=secret,id=hf_token \
    CUTOUT_MODELS=${CUTOUT_MODELS} HF_TOKEN_FILE=/run/secrets/hf_token python /app/download_models.py \
    && rm -rf /models/*/.cache /tmp/hf

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY postproc.py server.py /app/
WORKDIR /app
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--timeout-keep-alive", "75", "--no-access-log"]
