ARG BASE_IMAGE=public.ecr.aws/lts/ubuntu:22.04_stable
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update \
      && apt-get install -y --no-install-recommends \
            ca-certificates \
            autoconf \
            automake \
            build-essential \
            git \
            gperf \
            libre2-dev \
            libssl-dev \
            libtool \
            libcurl4-openssl-dev \
            libb64-dev \
            libgoogle-perftools-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            rapidjson-dev \
            scons \
            software-properties-common \
            pkg-config \
            unzip \
            zlib1g-dev \
            libarchive-dev \
            libxml2-dev \
            libnuma-dev \
            wget \
            nginx \
      && rm -rf /var/lib/apt/lists/*

      RUN pip3 install --upgrade pip \
      && pip3 install --upgrade \
          wheel \
          setuptools \
          docker \
          virtualenv

RUN wget -O /tmp/boost.tar.gz \
          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
      && (cd /tmp && tar xzf boost.tar.gz) \
      && mv /tmp/boost_1_80_0/boost /usr/include/boost

RUN apt update -q=2 \
      && apt install -y gpg wget \
      && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
      && . /etc/os-release \
      && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
      && apt-get update -q=2 \
      && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7*

RUN git clone https://github.com/triton-inference-server/server.git /server
RUN cd /server && git fetch origin bf86c27169ad4027202c76cf609ce42a2ec7b533
RUN cd /server && git reset --hard bf86c27169ad4027202c76cf609ce42a2ec7b533
RUN cd /server && ./build.py -v --no-container-build --build-dir=/server/build --backend=python --enable-metrics --enable-logging --enable-stats --endpoint="http" --endpoint="grpc" --filesystem="s3"

RUN apt-get -y install gnupg2
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB > ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg --import  ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg  --export >  ./aws_neuron.gpg
RUN mv ./aws_neuron.gpg /etc/apt/trusted.gpg.d/
RUN rm ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

RUN add-apt-repository -y  "deb https://apt.repos.neuron.amazonaws.com jammy main"
RUN apt-get -y update

RUN apt-get -y install aws-neuronx-collectives=2.*
RUN apt-get -y install aws-neuronx-runtime-lib=2.*
RUN apt-get -y install aws-neuronx-tools=2.*

RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN pip3 install awscli
RUN pip3 install neuronx-cc==2.18.* torch-neuronx==2.6.* torchvision \
      transformers-neuronx neuronx-distributed neuronx-distributed-inference \
      git+https://github.com/huggingface/optimum-neuron.git@v0.1.0

RUN pip3 install ray

RUN cp -r /server/build/opt/* /opt/
RUN rm -rf /server

ENV VLLM_TARGET_DEVICE=neuron
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
RUN cd /opt/vllm && git fetch origin 5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
RUN cd /opt/vllm && git reset --hard 5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
RUN pip3 install -r /opt/vllm/requirements/neuron.txt
RUN cd /opt/vllm && pip3 install -e .

RUN git clone https://github.com/triton-inference-server/vllm_backend.git /opt/vllm_backend
RUN cd /opt/vllm_backend && git fetch origin 507e4dccabf85c3b7821843261bcea7ea5828802
RUN cd /opt/vllm_backend && git reset --hard 507e4dccabf85c3b7821843261bcea7ea5828802

RUN mkdir -p /opt/tritonserver/backends/vllm
RUN cp -r /opt/vllm_backend/src/* /opt/tritonserver/backends/vllm/

RUN pip3 install pynvml==11.5.3 transformers==4.48.3

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:/opt/aws/neuron/bin:/opt/tritonserver/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PJRT_DEVICE=NEURON

WORKDIR /opt/program
