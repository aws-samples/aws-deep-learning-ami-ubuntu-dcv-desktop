ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:26.09-pyt-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The -pyt- base image ships only the C++ PyTorch backend, no Python torch package,
# so the Python backend models need torch installed explicitly. The cu130 wheel
# matches the pinned 580 driver. torchaudio/torchvision are not imported by any
# of the encoder models and are left out.
RUN pip3 install torch==2.14.1 jinja2
RUN pip3 install transformers==4.57.6 datasets==4.8.5

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
