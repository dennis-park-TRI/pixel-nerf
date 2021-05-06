ARG BASE_DOCKER_IMAGE
FROM $BASE_DOCKER_IMAGE

ARG python=3.6
ENV PYTHON_VERSION=${python}

# -----------------------------------
# TRI-specific environment variables.
# -----------------------------------
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}

ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

ARG WANDB_ENTITY
ENV WANDB_ENTITY=${WANDB_ENTITY}

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# -------------------------
# Install core APT packages.
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      # essential
      build-essential \
      cmake \
      ffmpeg \
      g++-4.8 \
      git \
      curl \
      docker.io \
      vim \
      wget \
      unzip \
      ca-certificates \
      htop \
      libjpeg-dev \
      libpng-dev \
      libavdevice-dev \
      pkg-config \
      # python
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python3-tk \
      python${PYTHON_VERSION}-distutils \
      # opencv
      python3-opencv \
    # set python
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------------------------
# MPI backend for pytorch distributed training (covers both single- and multi-node training).
# -------------------------------------------------------------------------------------------
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# -------------------------
# Install core PIP packages.
# -------------------------
# Upgrade pip.
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Core tools.
RUN pip install -U \
        numpy scipy pandas matplotlib seaborn boto3 requests tenacity tqdm awscli scikit-image \
        wandb mpi4py onnx==1.5.0 onnxruntime coloredlogs pycuda

# # Install pytorch 1.8 (CUDA 10.1)
# RUN pip uninstall -y torch
# RUN pip install torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# Install pytorch 1.7 (CUDA 10.1)
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install fvcore and detectron2.
ENV FVCORE_CACHE="/tmp"
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# RUN python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

#-------------------------------------------------------
# Copy working directory, and optionally install package.
#-------------------------------------------------------
ARG WORKSPACE
COPY . ${WORKSPACE}
WORKDIR ${WORKSPACE}
# # (Optional)
# RUN python setup.py build develop

# Add tri-det to PYTHONPATH.
# Assumption: `tri-det` added as submodule from the top-level.
ENV PYTHONPATH "${PYTHONPATH}:${WORKSPACE}/tri-det/"


# -------------------------------------
# Install project-specific PIP packages.
# -------------------------------------
# cocoapi
RUN pip install cython && pip install -U pycocotools

# For panoptic segmentation experiments (and for preparing panoptic dataset)
# RUN pip install git+https://github.com/cocodataset/panopticapi.git
# This branch fix the the bug: safely close multiprocessing pools, therefore prevent memory leaking.
RUN pip install git+https://github.com/dennis-park-TRI/panopticapi.git

# Pre-build pytorch3d
RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt170/download.html
# Install pytorch3d
# # https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md
# # CUB is pytorch3d dependency when built from source.
# RUN mkdir -p ${WORKSPACE}/../CUB
# WORKDIR ${WORKSPACE}/../CUB
# RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
# RUN tar xzf 1.10.0.tar.gz
# ENV CUB_HOME=${WORKSPACE}/../CUB/cub-1.10.0
# # ENV FORCE_CUDA="1"
# # RUN pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
# RUN pip install 'git+https://github.com/dennis-park-TRI/pytorch3d.git@prevent-cuda-error'

# To use detectron2 Cityscapes dataset (data preparation, eval)
RUN pip install cityscapesscripts shapely
RUN pip install nuscenes-devkit
# Required to run nuscenes 3D det eval script.
RUN pip install motmetrics

# Install packnet-sfm
# For  DDAD depth data (visualization, evaluation)
WORKDIR ${WORKSPACE}/../
RUN git clone https://github.com/TRI-ML/packnet-sfm.git
ENV PYTHONPATH "${PYTHONPATH}:${WORKSPACE}/../packnet-sfm"

# # Install mmdet3d (for BEV NMS)
# WORKDIR ${WORKSPACE}/../
# RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
# RUN pip install git+https://github.com/open-mmlab/mmdetection.git

# RUN git clone https://github.com/open-mmlab/mmdetection3d.git \
#     && cd mmdetection3d \
#     && git checkout ac9a3e8cbad85b8f893fc526e900fc005c176a19 \
#     && pip install --no-cache-dir -e .

RUN pip install pyhocon dotmap imageio-ffmpeg

# -----------
# Final steps
# -----------
WORKDIR ${WORKSPACE}

# # For eGPU on Lenovo P52
# ENV CUDA_VISIBLE_DEVICES=0
