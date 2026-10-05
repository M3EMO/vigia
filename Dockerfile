# syntax=docker/dockerfile:1.7
#
# Imagen de la plataforma Vigía (núcleo Apache 2.0 + vigia_ee comercial), solo CPU.
#
#   docker build -t vigia-platform .
#   docker build -t vigia-platform --build-arg PREFETCH_MODEL=ibm-granite/granite-3.1-1b-a400m-instruct .
#
# - torch sale del índice CPU de PyTorch; el resto de las dependencias sale de pyproject.toml
#   (extras hf, serve y ee), así pyproject sigue siendo la única lista.
# - El código se copia como fuente en /opt/vigia/app y corre con `python -m`: no depende de que el
#   wheel empaquete schema.sql, plantillas y estáticos.
# - Usuario sin privilegios (uid 10001). El único directorio escribible que la app necesita es
#   VIGIA_DATA_DIR (/var/lib/vigia, volumen) más /tmp (tmpfs). Funciona con `--read-only`.
# - Sin red saliente en runtime: HF_HUB_OFFLINE=1. El modelo va pre-descargado en la imagen
#   (PREFETCH_MODEL) o montado en HF_HOME (/opt/vigia/hf). Ver docs/operacion.md.
# - Secretos por archivo: VIGIA_PEPPERS_FILE y VIGIA_OWNER_PASSWORD_FILE apuntan a /run/secrets.
#
# Esta imagen no se construyó en la máquina de desarrollo (sin Docker). Ver docs/operacion.md.

ARG PYTHON_IMAGE=python:3.12-slim

# --- builder: venv con dependencias -----------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC="torch>=2.4"

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml ./

# torch primero desde el índice CPU: accelerate y transformers lo encuentran instalado y no traen
# la variante CUDA de PyPI.
RUN pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"

# Dependencias del proyecto (base + hf + serve + ee) leídas de pyproject.toml, sin torch.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; reqs = p['dependencies'] + [r for e in ('hf', 'serve', 'ee') for r in p['optional-dependencies'][e]]; print('\n'.join(r for r in reqs if not r.startswith('torch')))" \
        > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && python -c "import torch, transformers, fastapi, jinja2; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# --- model: pre-descarga opcional del checkpoint en el cache de Hugging Face ---------------------------
FROM builder AS model

ARG PREFETCH_MODEL=""
ENV HF_HOME=/opt/vigia/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# Sin PREFETCH_MODEL, la carpeta queda vacía y el modelo se monta en runtime.
RUN mkdir -p /opt/vigia/hf \
    && if [ -n "${PREFETCH_MODEL}" ]; then \
         python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "${PREFETCH_MODEL}"; \
       fi

# --- runtime ----------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="vigia-platform" \
      org.opencontainers.image.description="Vigía: confianza por token para LLM MoE autoalojados (plataforma)" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-Vigia-Commercial"

RUN groupadd --system --gid 10001 vigia \
    && useradd --system --uid 10001 --gid vigia --home-dir /nonexistent --no-create-home \
         --shell /usr/sbin/nologin vigia \
    && mkdir -p /var/lib/vigia /opt/vigia/app \
    && chown vigia:vigia /var/lib/vigia \
    && chmod 0750 /var/lib/vigia

COPY --from=builder /opt/venv /opt/venv
COPY --from=model /opt/vigia/hf /opt/vigia/hf

WORKDIR /opt/vigia/app
COPY LICENSE pyproject.toml ./
COPY vigia ./vigia
COPY vigia_ee ./vigia_ee

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/vigia/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/opt/vigia/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    VIGIA_DATA_DIR=/var/lib/vigia \
    VIGIA_PEPPERS_FILE=/run/secrets/vigia_peppers \
    VIGIA_OWNER_PASSWORD_FILE=/run/secrets/vigia_owner_password

USER 10001:10001

VOLUME ["/var/lib/vigia"]
EXPOSE 8000

# /livez solo mira el proceso. El modelo carga antes de abrir el puerto: start-period lo cubre.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM

CMD ["python", "-m", "vigia_ee.app", "serve", "--host", "0.0.0.0"]
