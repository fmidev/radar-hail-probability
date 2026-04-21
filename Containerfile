FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY .git/ .git/

# hatch-vcs requires a git checkout to derive the version; supply a fallback.
ENV HATCH_VCS_PRETEND_VERSION=0.0.0
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENTRYPOINT ["python", "/app/src/scripts/entrypoint.py"]
