# Recorte de producto en GPU para Alibaba Function Compute (custom container, puerto 9000).
# Sin libcuda.so en la imagen: la plataforma inyecta el driver (580.x, CUDA 13), que es
# compatible hacia atrás con el runtime CUDA 12.4 que traen las ruedas de torch.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/models \
    HF_HOME=/tmp/hf \
    HF_MODULES_CACHE=/tmp/hf_modules

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 \
    && pip uninstall -y triton

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY download_models.py /app/download_models.py
RUN python /app/download_models.py && rm -rf /models/*/.cache /tmp/hf

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY postproc.py server.py /app/
WORKDIR /app
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--timeout-keep-alive", "75", "--no-access-log"]
