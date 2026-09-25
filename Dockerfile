# syntax=docker/dockerfile:1.7
#
# Client Research & Qualification Agent - container image.
#
#   docker build -t client-research-agent .
#   docker run --rm client-research-agent research --company "Microsoft Corporation" --ticker MSFT
#
# Builder stage installs pinned dependencies (requirements.txt) and the wheel
# into /opt/venv; the runtime stage copies only that venv, so no compilers,
# uv or build backends ship in the final image.

ARG PYTHON_IMAGE=python:3.12-slim

FROM ghcr.io/astral-sh/uv:0.12.18 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv
WORKDIR /build

RUN uv venv --python "$(command -v python3)" /opt/venv

# Dependencies first for layer caching. The OTLP exporter is pinned through
# requirements-dev.txt used as a constraints file.
COPY requirements.txt requirements-dev.txt ./
RUN uv pip install --requirement requirements.txt \
    && uv pip install --constraint requirements-dev.txt opentelemetry-exporter-otlp-proto-http

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv build --wheel --out-dir /build/dist \
    && uv pip install --no-deps /build/dist/*.whl

FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=1.0.0
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="client-research-qualification-agent" \
      org.opencontainers.image.description="Agentic RAG over public company evidence producing cited, scored client briefs." \
      org.opencontainers.image.source="https://github.com/kogunlowo123/client-research-qualification-agent" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CRA_ENVIRONMENT=local

RUN groupadd --system --gid 10001 cra \
    && useradd --system --uid 10001 --gid cra --home-dir /app --no-create-home --shell /usr/sbin/nologin cra \
    && mkdir -p /app/var \
    && chown -R cra:cra /app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER 10001:10001
VOLUME ["/app/var"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["cra", "serve-check"]

ENTRYPOINT ["cra"]
CMD ["--help"]
