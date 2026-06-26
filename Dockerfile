# Chat from Scratch — Docker dev environment
# Server: NVIDIA driver 525.60.11 → CUDA 12.0 → PyTorch cu118 (compatible)
FROM docker.1ms.run/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3 python3-pip git curl wget vim htop \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch 2.5.1 + CUDA 11.8 (compatible with driver 525)
RUN pip install --no-cache-dir \
        torch==2.5.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# Project deps
RUN pip install --no-cache-dir \
        tokenizers>=0.19.0 \
        datasets>=2.20.0 \
        pyarrow \
        pyyaml \
        tqdm \
        pytest \
        transformers \
        tensorboard \
        accelerate

WORKDIR /workspace
CMD ["/bin/bash"]
