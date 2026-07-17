# syntax=docker/dockerfile:1.7
# Multi-stage build para Hermes (Asistente Jarvis)
# Sprint 4 MVP-2: incluye Agent-Reach (yt-dlp, gh CLI, Node.js, mcporter)
# - BuildKit cache mounts para pip (acelera re-builds)
# - amd64 only; arm64 remains unverified
#
# Stage 1: builder - instala dependencias Python en /install
# Stage 2: runtime - imagen final con system deps + Python deps

ARG PYTHON_VERSION=3.13

FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build

# Cache mount: pip guarda cache en /root/.cache/pip (persiste entre builds)
# --mount=type=cache solo funciona con BuildKit (DOCKER_BUILDKIT=1)
#
# Python 3.13 bump (2026-07-04) — sustituye el force-reinstall + cleanup
# defensivo de PRs #53/#54. python:3.13-slim trae wheel >=0.46 y
# jaraco.context >=6.x preinstalados (Trixie base + setuptools 81+), asi
# que no hay que parchearlos post-install. Las iteraciones previas en 3.11
# parcheaban cada path vulnerable uno a uno; cambiar a 3.13 elimina el
# problema de raiz en lugar de ir tapando cada manifest residual.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="oroimen" \
      org.opencontainers.image.description="Oroimen - local-first personal AI assistant" \
      org.opencontainers.image.source="https://github.com/adrianmedico/oroimen" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

# Sprint 4 MVP-2: instalar gh CLI (GitHub search) + Node.js (mcporter/Exa)
# + ffmpeg (audio transcripts) + deno (JS runtime para yt-dlp-ejs).
# yt-dlp y feedparser vienen como deps de agent-reach (pip). mcporter
# via npm global. deno via binario prebuilt (oficial de deno.land).
#
# deno + yt-dlp-ejs son ALTAMENTE RECOMENDADOS por yt-dlp para YouTube:
# - deno es el JS runtime que yt-dlp usa para resolver el JS challenge
# - yt-dlp-ejs provee las librerias JS externas que YouTube requiere
# Sin ellos, yt-dlp puede fallar con "Sign in to confirm you're not a bot"
# o HTTP 429. Con ellos, funciona en la mayoria de videos publicos.
#
# Se hace aqui (build time) porque:
# - apt-get install requiere root (somos root en build, no en runtime)
# - npm install -g deja binarios en /usr/local/bin (accesible para hermes)
# - El primer arranque del container no necesita red a GitHub/npm
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
        unzip \
        ffmpeg \
        # Trivy CVE bump (2026-07-04): libssh2-1t64 1.11.1-1 → 1.11.1-1+deb13u1
        # (CVE-2026-55199/55200/7598: DoS, OOB Write, int overflow).
        # Anadirlo al install (no upgrade separado) garantiza el fix tanto si
        # esta preinstalado en la base (upgrade) como si no (install).
        libssh2-1t64 \
        # Sprint 19 Slice 4b: Tesseract OCR engine for JPG/PNG extraction.
        # tesseract-ocr: el binario (paquete Debian oficial).
        # -spa/-deu/-eng: language packs for multilingual OCR.
        #   These cover the public demo languages; operators may add others.

        # Sin tesseract instalado, pytesseract.image_to_data() lanza
        # TesseractNotFoundError; el extractor lo captura y queuea el
        # archivo a ocr_pending con error='tesseract_not_installed' (Sprint 19
        # §4.4.1 fallback contract).
        tesseract-ocr tesseract-ocr-spa tesseract-ocr-deu tesseract-ocr-eng \
    && install -d -m 0755 /usr/local/lib/apt-keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/local/lib/apt-keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/local/lib/apt-keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends \
        gh nodejs \
    && rm -rf /var/lib/apt/lists/* \
    # Trivy CVE bump v3 (2026-07-04): picomatch 4.0.3 → 4.0.4 (ReDoS) +
    # sigstore 3.1.0 → 4.1.1 (cert verification drop).
    # Iteración 1 (PR #51): `npm install -g mcporter --omit=dev` NO los
    # excluye — son runtime deps, no devDeps.
    # Iteración 2 (PR #53): `npm install -g --force picomatch@4.0.4
    # sigstore@4.1.1` tampoco sirve — los nested dentro de npm-cli
    # (/usr/lib/node_modules/npm/node_modules/) y mcporter ganan al
    # global top-level por resolución de node_modules.
    # Iteración 3 (PR #54 build fallido): `cd` DENTRO de npm-cli y
    # `npm install` dispara resolución de TODO el dep tree de npm-cli.
    # npm-cli's package.json depende de `@npmcli/docs@^1.0.0` que ya
    # no existe en el registry → npm error E404.
    # Iteración 4 (este bloque): `npm pack <pkg>@<ver>` baja el tarball
    # SIN resolver deps, y `tar -xzf` lo extrae directamente sobre el
    # destino nested. Evita el resolver de npm y no rompe npm-cli.
    && npm install -g mcporter@latest --omit=dev \
    && cd /tmp \
        && npm pack picomatch@4.0.4 --silent --pack-destination=/tmp \
        && npm pack sigstore@4.1.1 --silent --pack-destination=/tmp \
        && MCPORTER_DIR="$(npm root -g)/mcporter" \
        && rm -rf "$MCPORTER_DIR/node_modules/picomatch" \
        && mkdir -p "$MCPORTER_DIR/node_modules/picomatch" \
        && tar -xzf /tmp/picomatch-4.0.4.tgz \
            -C "$MCPORTER_DIR/node_modules/picomatch" --strip-components=1 \
        && rm -rf "$MCPORTER_DIR/node_modules/sigstore" \
        && mkdir -p "$MCPORTER_DIR/node_modules/sigstore" \
        && tar -xzf /tmp/sigstore-4.1.1.tgz \
            -C "$MCPORTER_DIR/node_modules/sigstore" --strip-components=1 \
        && rm -rf /usr/lib/node_modules/npm/node_modules/picomatch \
        && mkdir -p /usr/lib/node_modules/npm/node_modules/picomatch \
        && tar -xzf /tmp/picomatch-4.0.4.tgz \
            -C /usr/lib/node_modules/npm/node_modules/picomatch --strip-components=1 \
        && rm -rf /usr/lib/node_modules/npm/node_modules/sigstore \
        && mkdir -p /usr/lib/node_modules/npm/node_modules/sigstore \
        && tar -xzf /tmp/sigstore-4.1.1.tgz \
            -C /usr/lib/node_modules/npm/node_modules/sigstore --strip-components=1 \
        && rm /tmp/picomatch-4.0.4.tgz /tmp/sigstore-4.1.1.tgz \
        && cd / \
    && npm cache clean --force \
    # deno: binario prebuilt desde GitHub releases (oficial).
    # Solo ~30MB, no requiere npm. Lo ponemos en /usr/local/bin para
    # que yt-dlp lo encuentre via PATH.
    && curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
        -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin/ \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && deno --version

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash --uid 1000 hermes

WORKDIR /app

# Copiar dependencias pre-instaladas y código
COPY --from=builder /install /usr/local
COPY hermes/ ./hermes/
# Sprint 4 MVP-2: setup script idempotente de Agent-Reach.
# Crea ~/.agent-reach/config.yaml template, configura yt-dlp JS runtime
# y Exa MCP. Se ejecuta en cada arranque via __main__.py.
COPY scripts/__init__.py scripts/setup_agent_reach.py ./scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/bin:$PATH

# Python 3.13 bump: la purga de wheel/jaraco vendoreados que teniamos
# en 3.11 ya no aplica — el base image 3.13-slim no trae esas versiones
# vulnerables. Si Trivy volviera a reportarlas seria senal de que
# una dep transitiva las esta metiendo, pero los rangos de requirements.txt
# debe cubrirlas.

# Empty named volumes inherit these mount-point permissions on first use.
# The tracked drop/ directory keeps the bind mount deterministic for the
# public quickstart; operators may replace it with their own writable mount.
RUN mkdir -p /app/data /app/backups /app/drop \
    && chown -R hermes:hermes /app/data /app/backups /app/drop /home/hermes

USER hermes

# Healthcheck: 200 si /health responde
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

EXPOSE 8000

CMD ["python", "-m", "hermes"]
