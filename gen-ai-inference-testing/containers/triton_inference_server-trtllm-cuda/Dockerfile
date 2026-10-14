ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://github.com/NVIDIA/TensorRT-LLM.git /opt/TensorRT-LLM
RUN cd /opt/TensorRT-LLM && git fetch origin 376f7e1bd8ed543f75014309e3fd4b237e9b0e73
RUN cd /opt/TensorRT-LLM && GIT_LFS_SKIP_SMUDGE=1 git reset --hard 376f7e1bd8ed543f75014309e3fd4b237e9b0e73

RUN pip3 install datasets==4.8.5 evaluate~=0.4.3 rouge_score~=0.1.2 sentencepiece~=0.2.0

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
