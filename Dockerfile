# Build ui using node
FROM node:18.12.1

COPY package.json .
COPY package-lock.json .
COPY public public
COPY src src
COPY tsconfig.json .
COPY .eslintrc.json .

RUN npm install

RUN npm run build


# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04


COPY --from=0 build build
# Set the working directory to /app
#WORKDIR /app

# Update package lists and install required packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y python3.9 python3-pip

# # Make sure Python 3.9 is the default python3 version
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Create a symbolic link for pip (optional)
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Verify Python and pip versions
RUN python3 --version && pip --version

# # Create symbolic links to set Python 3.9 as the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
#     update-alternatives --config python3
# RUN apt-get update && apt-get install -y python3.9 python3.9-dev

COPY translator translator
COPY server.py .
COPY fonts fonts
COPY models models
COPY requirements.txt .


RUN pip install -r requirements.txt

RUN pip uninstall -y torch torchvision torchaudio

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

CMD ["python","server.py"]


