# Basis-Image: Offizielles Python 3.11 (enthält bereits die meisten Tools)
FROM python:3.11

LABEL maintainer="Gemeinderatsprotokolle-Transkriptionsservice"

# ===== CUDNN INSTALLATION VOR System-Packages =====
# Kopiere cuDNN-Bibliotheken ins Image (verhindert Mount-Konflikte)
COPY cudnn/lib64/* /usr/local/lib/cudnn/
COPY cudnn/include/* /usr/local/include/cudnn/

# Erstelle CUDA-Verzeichnisse und setze ldconfig
RUN mkdir -p /usr/local/cuda/lib64 /usr/local/cuda/include && \
    # Direkte Kopien statt Symlinks (robuster)
    # Fail fast if essential cuDNN files are not found from the COPY step
    echo "Copying cuDNN libraries to /usr/local/cuda/lib64..." && \
    cp /usr/local/lib/cudnn/libcudnn* /usr/local/cuda/lib64/ && \
    echo "Verifying cuDNN libraries in /usr/local/cuda/lib64..." && \
    ls -1 /usr/local/cuda/lib64/libcudnn* && \
    echo "Copying cuDNN headers to /usr/local/cuda/include..." && \
    cp /usr/local/include/cudnn/cudnn*.h /usr/local/cuda/include/ && \
    echo "Verifying cuDNN headers in /usr/local/cuda/include..." && \
    ls -1 /usr/local/cuda/include/cudnn*.h && \
    # Update library cache
    echo "/usr/local/lib/cudnn" > /etc/ld.so.conf.d/cudnn.conf && \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig && \
    # Verifikation (original)
    echo "Final verification listing of copied cuDNN files:" && \
    ls -la /usr/local/lib/cudnn/ && \
    ls -la /usr/local/cuda/lib64/libcudnn* || echo "cuDNN files listing after ldconfig"

# Umgebungsvariablen
ENV PIP_DEFAULT_TIMEOUT=100 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    TORCH_HOME=/app/.cache/torch \
    WHISPER_CACHE_DIR=/app/.cache/whisper \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    LD_LIBRARY_PATH="/usr/local/lib/cudnn:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Nexus-Einstellungen für pip
ARG PIP_INDEX
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST

# Arbeitsverzeichnis
WORKDIR /app

# ✅ CACHE-VERZEICHNISSE ERSTELLEN
RUN mkdir -p /app/uploads /app/results /app/protocols \
    /app/.cache/huggingface/hub \
    /app/.cache/huggingface/transformers \
    /app/.cache/torch \
    /app/.cache/whisper \
    /app/config \
    /app/services \
    /app/frontend

# Requirements kopieren
COPY requirements.txt .

# ✅ Python-Abhängigkeiten installieren (über Nexus)
RUN if [ -n "$PIP_INDEX_URL" ] && [ -n "$PIP_TRUSTED_HOST" ]; then \
        pip install --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" --default-timeout=600 -r requirements.txt --no-cache-dir; \
    else \
        pip install --default-timeout=600 -r requirements.txt --no-cache-dir; \
    fi

# ✅ TESTE Python Audio-Bibliotheken + cuDNN
RUN python -c "import soundfile as sf; import librosa; import numpy as np; print('✅ Python Audio-Stack bereit'); print('SoundFile:', sf.__version__); print('Librosa:', librosa.__version__)"

# ✅ TESTE WhisperX + cuDNN Integration
RUN python -c "import whisperx; import torch; print('✅ WhisperX-Stack bereit'); print('PyTorch:', torch.__version__); print('CUDA verfügbar:', torch.cuda.is_available()); print('cuDNN verfügbar:', torch.backends.cudnn.is_available() if torch.cuda.is_available() else 'N/A (kein CUDA)')"

# ✅ Anwendungscode kopieren
COPY app.py .
COPY config/ ./config/
COPY services/ ./services/
COPY frontend/ ./frontend/


# ✅ Entrypoint kopieren und ausführbar machen
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Volumes definieren
VOLUME ["/app/uploads", "/app/results", "/app/protocols", "/app/.cache"]

# Port freigeben
EXPOSE 8005

# Zeitzone setzen
ENV TZ=Europe/Berlin

# ✅ ERWEITERTE CUDNN-HEALTH-CHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.is_available() if torch.cuda.is_available() else False); import whisperx; print('WhisperX ready')" || exit 1

# Anwendung starten
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]