FROM python:3.14-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY .git/ .git/

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV

RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip \
    && $VIRTUAL_ENV/bin/pip install --no-cache-dir .


FROM python:3.14-slim

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
