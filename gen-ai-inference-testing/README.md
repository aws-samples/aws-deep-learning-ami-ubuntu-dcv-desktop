# Gen AI Inference Testing

Comprehensive inference testing framework for Generative AI models using multiple inference servers and backends on [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop).

## Overview

This directory provides Docker containers, scripts, and Jupyter notebooks for testing LLM and embedding model inference performance using:

**Inference Servers:**
* [Triton Inference Server](https://github.com/triton-inference-server)
* [DJL Serving](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) with LMI (Large Model Inference)
* OpenAI-compatible Server

**Backends:**
* [vLLM](https://github.com/vllm-project/vllm) - GPU and Neuron
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - GPU only
* DJL-LMI - Neuron only
* Custom Python backend for embeddings

**Supported Hardware:**
* NVIDIA GPUs (CUDA)
* AWS Trainium/Inferentia (Neuron)

## Quick Start

### LLM Inference Testing

Launch Visual Studio Code and open [load_testing.ipynb](./locust-testing/load_testing.ipynb) to:
1. Build Docker containers for your target hardware
2. Download and cache Hugging Face models
3. Launch inference servers locally
4. Run Locust load tests with configurable concurrency
5. Visualize performance metrics

### Embeddings Model Testing

Open [load_testing_embeddings.ipynb](./locust-testing/load_testing_embeddings.ipynb) for testing embedding models with Triton Inference Server.

### LiteLLM Testing

Open [litellm_testing.ipynb](./locust-testing/litellm_testing.ipynb) for testing with LiteLLM proxy.

## Directory Structure

### containers/

Docker container definitions for all inference server and backend combinations:

* **djl-serving-neuronx-lmi/** - DJL Serving with LMI for Neuron
* **openai-server-cuda-vllm/** - OpenAI-compatible server with vLLM on CUDA
* **openai-server-neuronx-vllm/** - OpenAI-compatible server with vLLM on Neuron (includes vLLM patches)
* **tritonserver-cuda-trtllm/** - Triton with TensorRT-LLM on CUDA
* **tritonserver-cuda-vllm/** - Triton with vLLM on CUDA
* **tritonserver-neuronx-djl-lmi/** - Triton with DJL-LMI backend on Neuron
* **tritonserver-neuronx-vllm/** - Triton with vLLM on Neuron (includes vLLM patches)

### scripts/

Build scripts for Docker containers:
* `build-djl-serving-neuronx-lmi.sh`
* `build-openai-server-cuda-vllm.sh`
* `build-openai-server-neuronx-vllm.sh`
* `build-tritonserver-cuda-trtllm.sh`
* `build-tritonserver-cuda-vllm.sh`
* `build-tritonserver-neuronx.sh`
* `build-tritonserver-neuronx-djl-lmi.sh`
* `build-tritonserver-neuronx-vllm.sh`

### djl-serving/

DJL Serving deployment configurations and scripts:

* **compose/** - Docker Compose files for different Neuron device counts (1, 6, 12, 16) and CUDA
* **tensorrt-llm/** - TensorRT-LLM backend scripts and compose files
* **vllm/** - vLLM backend scripts for CUDA and Neuron
* **tests/** - Test scripts for DJL-LMI endpoints

### openai-server/

OpenAI-compatible server deployment:

* **compose/** - Docker Compose files for different Neuron device counts (1, 6, 12, 16) and CUDA
* **vllm/** - vLLM backend scripts for CUDA and Neuron

### triton-server/

Triton Inference Server deployment configurations:

* **compose/** - Docker Compose files for different Neuron device counts (1, 6, 12, 16) and CUDA
* **djl-lmi/** - DJL-LMI backend with custom Python backend (`tnx_lmi_backend.py`)
* **embeddings/** - Custom Python backend for embeddings (`triton_embeddings_backend.py`)
* **tensorrt-llm/** - TensorRT-LLM backend with engine build scripts
* **vllm/** - vLLM backend scripts for CUDA and Neuron
* **tests/** - Test scripts for Triton endpoints (vLLM, TensorRT-LLM, DJL-LMI)

### locust-testing/

Locust-based load testing framework:

**Notebooks:**
* `load_testing.ipynb` - Main LLM inference testing notebook
* `load_testing_embeddings.ipynb` - Embeddings model testing notebook
* `litellm_testing.ipynb` - LiteLLM proxy testing notebook

**Configuration:**
* **config/** - YAML configurations for different server/backend combinations
  * **code-gen/** - Configs for code generation models
  * **multi-modal/** - Configs for multi-modal models
  * Base configs for text generation and embeddings

**Modules:**
* **modules/inst-semeval2017/** - Embedding and reranking prompt generators
* **modules/mminstruction-m3it/** - Multi-modal prompt generators
* **modules/nicholasKluge-toxic-text/** - Llama Guard prompt generators
* **modules/ronneldan_tinystories/** - Tiny stories prompt generators
* **modules/sahil2801-codealpca20k/** - Code generation prompt generators
* **modules/thudm-longbench/** - Long context prompt generators

**Scripts:**
* `build-containers.sh` - Build all LLM inference containers
* `build-containers-embeddings.sh` - Build embeddings containers
* `run_locust.sh` - Execute Locust load tests
* `run_llm_perf_openai.sh` - Run LLM performance benchmarks
* `endpoint_user.py` - Locust user implementation
* `custom_endpoint_handler.py` - Custom endpoint handler
* `litellm_config.yaml` - LiteLLM configuration

## Usage Examples

### Testing with Triton + vLLM on GPU

```python
# In load_testing.ipynb
inference_server = 'triton_inference_server'
backend = 'vllm'
hf_model_id = 'meta-llama/Llama-3.1-8B-Instruct'
```

### Testing with DJL Serving + TensorRT-LLM

```python
# In load_testing.ipynb
inference_server = 'djl_serving'
backend = 'trtllm'
hf_model_id = 'meta-llama/Llama-3.1-70B-Instruct'
```

### Testing with OpenAI Server + vLLM on Neuron

```python
# In load_testing.ipynb
inference_server = 'openai_server'
backend = 'vllm'
hf_model_id = 'meta-llama/Llama-3.1-8B-Instruct'
# Automatically detects Neuron hardware
```

### Testing Embeddings Models

```python
# In load_testing_embeddings.ipynb
inference_server = 'triton_inference_server'
backend = 'embeddings'
hf_model_id = 'BAAI/bge-large-en-v1.5'
```

## Key Features

* **Automated Model Caching** - Downloads and caches Hugging Face models to EFS
* **Hardware Auto-Detection** - Automatically detects CUDA GPUs or Neuron devices
* **Flexible Configuration** - YAML-based configs for different workloads
* **Dynamic Prompt Generation** - Pluggable prompt generators for various datasets
* **Concurrent Load Testing** - Locust-based testing with configurable concurrency
* **Performance Metrics** - CSV output with latency, throughput, and error rates
* **Docker Compose Orchestration** - Easy deployment with multiple device configurations

## Requirements

* AWS Deep Learning Desktop with GPU or Neuron instances
* Docker and Docker Compose
* Hugging Face account and access token (for gated models)
* EFS mounted at `/home/ubuntu/efs` for model caching

## Notes

* Neuron vLLM containers support both stock vLLM and [Neuron Upstreaming to vLLM](https://github.com/aws-neuron/upstreaming-to-vllm)
* TensorRT-LLM requires model-specific conversion scripts
* Tensor parallel size is auto-configured based on available device cores
* All containers mount `/snapshots` for model access