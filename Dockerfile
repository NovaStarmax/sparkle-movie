# ── Stage 1 : builder ────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Resolve and install production dependencies only.
# Copying lockfiles first lets Docker cache this layer as long as they don't change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-extra notebooks

# Copy source code after deps so the dep-install layer stays cached on code changes
COPY . .


# ── Stage 2 : runtime ────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy the pre-built virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy only the runtime artifacts — no notebooks, no dev tooling
COPY --from=builder /app/api       ./api
COPY --from=builder /app/frontend  ./frontend
COPY --from=builder /app/models    ./models
COPY --from=builder /app/tests     ./tests

ENV PATH="/app/.venv/bin:$PATH"
ENV MODELS_PATH="/app/models"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
