# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

WORKDIR /app

COPY translator/src translator/src
COPY translator/.python-version translator/.python-version
COPY translator/pyproject.toml translator/pyproject.toml
COPY translator/uv.lock translator/uv.lock

COPY extension/backend/resources extension/backend/resources
COPY extension/backend/config.yaml extension/backend/config.yaml
COPY extension/backend/src extension/backend/src
COPY extension/backend/.env extension/backend/.env
COPY extension/backend/main.py extension/backend/main.py
COPY extension/backend/.python-version extension/backend/.python-version
COPY extension/backend/pyproject.toml extension/backend/pyproject.toml
COPY extension/backend/uv.lock extension/backend/uv.lock


RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    ffmpeg libsm6 libxext6  --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app/extension/backend

RUN uv sync

CMD ["uv", "run","poe","prod"]


