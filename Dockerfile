#Build ui using node
FROM node:18.12.1

WORKDIR /app

COPY ui ui

WORKDIR /app/ui

RUN npm install

RUN npm run build


# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

COPY --from=0 /app/ui/build /app/ui/build

WORKDIR /app

COPY translator translator
COPY server.py .
COPY fonts fonts
COPY models models
COPY .env .
COPY poetry.lock .
COPY pyproject.toml .


RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    libffi-dev libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx


RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update -n base -c defaults conda

RUN conda create -n translator python=3.10

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "translator", "/bin/bash", "-c"]

RUN conda install -c conda-forge poetry

RUN poetry install 

RUN poe force-cuda

EXPOSE 5000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "translator", "python3", "server.py"]


