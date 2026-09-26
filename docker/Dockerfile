FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget curl ffmpeg libgl1 libglib2.0-0 \
    libopenblas-dev libsparsehash-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Cài đặt PyTorch chuẩn CUDA 11.8
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện xe tự hành và trực quan hóa
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["/bin/bash"]
