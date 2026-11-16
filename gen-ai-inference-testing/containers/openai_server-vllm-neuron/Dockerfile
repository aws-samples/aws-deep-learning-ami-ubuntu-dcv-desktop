ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.13.0-neuronx-py312-sdk2.28.0-ubuntu24.04
FROM $BASE_IMAGE

# legacy: torch-neuronx/XLA stack (inf2, trn1, and the default elsewhere).
# native: vllm-neuron plugin on native TorchNeuron (trn2/trn3); built with
#         --build-arg BASE_IMAGE=<0.29 DLC> NEURON_VLLM_STACK=native PJRT_DEVICE=
#         by scripts/openai_server-vllm-neuron.sh. PJRT_DEVICE=NEURON selects the
#         legacy XLA path, so the native build leaves it empty.
ARG NEURON_VLLM_STACK=legacy
ARG PJRT_DEVICE=NEURON
LABEL org.opencontainers.image.neuron-vllm-stack=$NEURON_VLLM_STACK

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=${PJRT_DEVICE} \
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

# Native images double as the SLO benchmark client (vllm bench serve ships with
# vLLM 0.29): add the Locust cross-check and plotting dependencies.
RUN if [ "$NEURON_VLLM_STACK" = "native" ]; then \
        python -m pip install --no-cache-dir "locust>=2.32" "pandas>=2.2" "matplotlib>=3.9" \
        && python -m pip freeze > /opt/pip-freeze.txt; \
    fi

# App Setup
COPY resources /opt/program
RUN chmod u+x /opt/program/serve && echo "$NEURON_VLLM_STACK" > /opt/program/neuron-stack
WORKDIR /opt/program
