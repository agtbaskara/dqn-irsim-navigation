FROM nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pyqt5 \
    python3-tk \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    ir-sim \
    ir-sim[all] \
    gymnasium \
    pygame \
    tqdm

# Install PyTorch with CUDA 12.9 support
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Set working directory
WORKDIR /app
