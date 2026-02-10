ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:26.01-pyt-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.9.0 torchaudio torchvision jinja2
RUN pip3 install transformers==4.57.3 datasets==4.4.2

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
