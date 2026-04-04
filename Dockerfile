# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials needed by some native extensions (rapidfuzz, numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY pharmatrials_env/ ./pharmatrials_env/
COPY baseline/         ./baseline/
COPY inference.py      ./
COPY openenv.yaml      ./
COPY README.md         ./

RUN pip install --no-cache-dir -e .

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="PharmaTrials-Env" \
      org.opencontainers.image.description="OpenEnv: Clinical-Trial Document Intelligence" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      openenv="true"

# Non-root user required by HuggingFace Spaces
RUN adduser --disabled-password --gecos "" --uid 1000 envuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# LLM configuration (override at runtime)
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV TASK_NAME=EASY
ENV SEED=42

RUN chown -R envuser:envuser /app

USER envuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT','7860') + '/health')"

CMD ["sh", "-c", "uvicorn pharmatrials_env.api.server:app --host ${HOST} --port ${PORT} --workers 1 --log-level info"]
