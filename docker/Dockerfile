# ===
# Dockerfile to train ReFiNe
#   => Base Image :: Python 3.11, Pytorch 2.2, CUDA 12.1.0
# ===

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel


RUN apt-get update
RUN apt-get install -y software-properties-common

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
      build-essential tar xz-utils cmake curl docker.io ffmpeg \
      libxkbcommon-x11-0 libxkbcommon-dev xorg libx11-xcb-dev xcb xvfb \
      git htop tmux && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Install dependencies
# -------------------------
RUN pip install open3d==0.16.0
RUN pip install tinyobjloader==2.0.0rc8
RUN pip install scikit-image==0.19.3
RUN pip install wandb==0.13.9
RUN pip install einops==0.6.1
RUN pip install Werkzeug==2.2.2
RUN pip install kornia==0.6.12
RUN pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html

# -------------------------
# Install kaolin-wisp
# -------------------------
RUN cd /tmp && git clone https://github.com/NVIDIAGameWorks/kaolin-wisp.git
RUN cd /tmp/kaolin-wisp && git checkout cb47e10 && export FORCE_CUDA=1 && python setup.py develop
