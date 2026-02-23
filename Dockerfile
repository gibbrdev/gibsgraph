# Multi-stage build — minimal, non-root, secure
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && python -m build --wheel

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

# Non-root user
RUN groupadd --gid 1001 gibsgraph && \
    useradd --uid 1001 --gid gibsgraph --shell /bin/bash --create-home gibsgraph

WORKDIR /app

# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install only runtime deps (no dev)
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy app files
COPY app/ ./app/
COPY .env.example ./.env.example

USER gibsgraph

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
