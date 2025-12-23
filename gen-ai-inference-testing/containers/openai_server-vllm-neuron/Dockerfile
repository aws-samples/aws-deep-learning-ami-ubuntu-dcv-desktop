ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.27.0-ubuntu24.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=NEURON \
    LD_LIBRARY_PATH="/opt/conda/lib:/opt/aws/neuron/lib:${LD_LIBRARY_PATH}" \
    PATH="/opt/program:/opt/conda/bin:/opt/aws/neuron/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            build-essential \
            git \
            libssl-dev \
            libcurl4-openssl-dev \
            libgoogle-perftools-dev \
            libnuma-dev \
            pkg-config \
            unzip \
            wget \
            nginx \
      && rm -rf /var/lib/apt/lists/*

# Upgrade core tools in the existing environment
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel virtualenv build

# vLLM Installation
ARG VLLM_NEURON="git+https://github.com/vllm-project/vllm.git@v0.9.0.1"
ENV VLLM_TARGET_DEVICE=neuron
RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com && \
      pip3 install --no-cache-dir $VLLM_NEURON

RUN sed -i 's/AutoConfig\.register("aimv2", AIMv2Config)/# AutoConfig\.register("aimv2", AIMv2Config)/' \
 /opt/conda/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py

# App Setup
COPY resources /opt/program
RUN chmod u+x /opt/program/serve
WORKDIR /opt/program
