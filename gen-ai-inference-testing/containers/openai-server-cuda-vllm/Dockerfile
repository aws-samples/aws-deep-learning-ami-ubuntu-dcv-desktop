ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3"
FROM ${BASE_IMAGE}

RUN pip install vllm==0.9.2

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:opt/tritonserver/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
