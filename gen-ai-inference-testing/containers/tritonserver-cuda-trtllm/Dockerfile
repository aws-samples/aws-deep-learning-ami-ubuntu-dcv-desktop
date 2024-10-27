ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /opt/TensorRT-LLM
RUN cd /opt/TensorRT-LLM && git fetch origin 250d9c293d5edbc2a45c20775b3150b1eb68b364
RUN cd /opt/TensorRT-LLM && git reset --hard 250d9c293d5edbc2a45c20775b3150b1eb68b364

RUN pip install --upgrade pip
RUN pip install datasets==2.20.0 evaluate~=0.4.2 rouge_score~=0.1.2 sentencepiece~=0.2.0

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /opt/tensorrtllm_backend
RUN cd /opt/tensorrtllm_backend && git fetch origin 76464e9be06600f3979acad9c14857938a66ff9f
RUN cd /opt/tensorrtllm_backend && git reset --hard 76464e9be06600f3979acad9c14857938a66ff9f

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:opt/tritonserver/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
