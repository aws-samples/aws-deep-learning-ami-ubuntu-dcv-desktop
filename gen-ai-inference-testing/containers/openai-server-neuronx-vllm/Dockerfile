ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
# vLLM build flags for Neuron
ENV VLLM_TARGET_DEVICE=neuron
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PJRT_DEVICE=NEURON
ENV PATH="/opt/program:${PATH}"

# 1. Consolidated APT Install
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

# 2. Upgrade core tools in the existing environment
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 3. Boost Headers
RUN wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
      && tar -xzf /tmp/boost.tar.gz -C /tmp \
      && mv /tmp/boost_1_80_0/boost /usr/include/boost \
      && rm -rf /tmp/boost*

# 4. Install Latest CMake (Required for vLLM builds)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
      && . /etc/os-release \
      && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
      && apt-get update \
      && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7* \
      && rm -rf /var/lib/apt/lists/*

# 5. vLLM Installation
RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
ARG USE_NEURON_VLLM=false
# Added VLLM_TARGET_DEVICE=neuron inline to ensure the build script sees it
RUN if [ "$USE_NEURON_VLLM" = "true" ]; then \
      VLLM_TARGET_DEVICE=neuron pip3 install git+https://github.com/aws-neuron/upstreaming-to-vllm.git@2.26.1; \
    else \
      VLLM_TARGET_DEVICE=neuron pip3 install git+https://github.com/vllm-project/vllm.git@v0.9.0.1; \
    fi

# 6. Final Python dependencies
RUN pip3 install --no-cache-dir pydantic pandas

# 7. App Setup
COPY resources /opt/program
RUN chmod u+x /opt/program/serve
WORKDIR /opt/program
