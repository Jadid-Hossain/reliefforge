# ReliefForge prediction app (course requirement: Dockerized app).
# CPU-only on purpose: ViT-S at 518 runs in a few seconds per image on CPU.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf_cache \
    GRADIO_SERVER_PORT=7860

# CPU torch first so requirements.txt does not pull the CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY app /app/app
COPY configs /app/configs
COPY .dockerignore /app/.dockerignore

# warm the zero-shot weights at build time so the container starts offline-capable
RUN python -c "import sys; sys.path.insert(0, '/app/src'); from reliefkit.model import ReliefModel; ReliefModel('depth-anything/Depth-Anything-V2-Small-hf')"

EXPOSE 7860
ENV RELIEFFORGE_FINETUNED=/models/finetuned
# mount a volume with the fine-tuned export at /models/finetuned, or set
# RELIEFFORGE_FINETUNED_HF to a Hugging Face repo id to download it at startup.
CMD ["python", "/app/app/app.py"]
