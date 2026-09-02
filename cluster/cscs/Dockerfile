# JAX 0.4.33, CUDA 12.4/12.6-era libraries, cuDNN, and NCCL on aarch64.
# This matches Terra's tested JAX 0.4 API more closely than current JAX 0.9+ images.
ARG BASE_IMAGE=nvcr.io/nvidia/jax:24.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/terra-cscs-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/terra-cscs-requirements.txt \
    && python -m pip check

RUN mkdir -p /workspace/terra /workspace/terra-baselines /workspace/run
WORKDIR /workspace/run

CMD ["/bin/bash"]
