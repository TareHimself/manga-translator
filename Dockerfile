# Build ui using node
# FROM node:18.12.1

# WORKDIR /app

# COPY ui ui

# WORKDIR /app/ui

# RUN npm install

# RUN npm run build


# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04


WORKDIR /app

# COPY --from=0 /app/ui/build /ui/build
# Set the working directory to /app
#WORKDIR /app

# Update package lists and install required packages

# RUN apt-get update
# RUN apt-get remove python
# RUN apt-get remove python-pip
# RUN apt-get -y install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get -y install python3.9
# RUN apt-get -y install python3-pip

COPY translator translator
COPY server.py .
COPY fonts fonts
COPY models models
COPY requirements.txt .

# # Install base utilities
# RUN apt-get update \
#     && apt-get install -y build-essential \
#     && apt-get install -y wget \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH

# # Create symbolic links to set Python 3.9 as the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
#     update-alternatives --config python3
# RUN apt-get update && apt-get install -y python3.9 python3.9-dev
# RUN conda create -n translator python=3.9 -y

# SHELL ["conda", "run", "-n", "translator", "/bin/bash", "-c"]

# RUN pip install -r requirements.txt
# RUN pip uninstall -y torch torchvision torchaudio
# RUN pip install opencv-python
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
#     curl python3-pip

# RUN pip3 install --upgrade pip

# RUN pip3 install -r requirements.txt
# RUN pip3 uninstall -y torch torchvision torchaudio
# RUN pip3 install opencv-python
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl python3-pip

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update -n base -c defaults conda

COPY conda.yml conda.yml

RUN conda env create -f conda.yml --name translator

RUN conda activate translator

# RUN python3.9 -m pip install -r requirements.txt

# RUN python3.9 -m 

# RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

CMD ["python3","server.py"]


