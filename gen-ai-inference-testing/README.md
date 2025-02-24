# Gen AI Inference Testing

This tutorial shows how to do Gen AI inference testing with [Deep Java Library (DJL) Large Model Inference (LMI) Server](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html), or [Triton Inference Server](https://github.com/triton-inference-server), on [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop). 

This solution is general purpose, but it has only been tested with `MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct`. You will need to adapt the test prompts in the [djl-serving/tests](./djl-serving/tests/) and [triton-server/tests](./triton-server/tests/) so they are consistent with the Hugging Face model you specify in `MODEL_ID` below. The default tests will only work for `MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct`.

## Tutorial Steps

### Step 1. Launch Deep Learning Ubuntu Desktop

This tutorial assumes a `trn1.32xlarge` machine for Neuron examples, and a `g5.48xlarge` for CUDA examples. You may want to launch the [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) with  `trn1.32xlarge` instance type for neuron, or `g5.48xlarge` for gpu.

### Step 2. Get Hugging Face Token For Gated Models

To access [Hugging Face gated models](https://huggingface.co/docs/hub/en/models-gated), get a Hugging Face token. You will need to specify it in the `HF_TOKEN` below. 

### Step 3. Build containers

To build the container for triton inference server with neuronx, execute this on `trn1.32xlarge` machine:

    ./scripts/build-tritonserver-neuronx.sh

To build the container for triton inference server with cuda and vLLM, execute this on gpu machine:

    ./scripts/build-tritonserver-cuda-vllm.sh

To build the container for triton inference server with cuda and TensorRT-LLM, execute this on gpu machine::

    ./scripts/build-tritonserver-cuda-trtllm.sh
    

### Step 4. Run Testing
### DJL Serving

#### TensorRT-LLM LMI Engine

This test should be run on a `g5.48xlarge` instance. To launch:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./djl-serving/tensorrt-llm/compose-djl-lmi-tensorrt-llm.sh up

To test:

    ./djl-serving/tests/test-djl-lmi.sh
    ./djl-serving/tests/test-djl-lmi-concurrent.sh

To stop:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./djl-serving/tensorrt-llm/compose-djl-lmi-tensorrt-llm.sh down

#### Transformers Neuronx LMI Engine

This test should be run on a `trn1.32xlarge` instance. To launch:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./djl-serving/transformers-neuronx/compose-djl-lmi-transformers-neuronx.sh up

To test:

    ./djl-serving/tests/test-djl-lmi.sh
    ./djl-serving/tests/test-djl-lmi-concurrent.sh

To stop:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./djl-serving/transformers-neuronx/compose-djl-lmi-transformers-neuronx.sh down

### Triton Inference Server

#### TensorRT-LLM Backend

To launch, execute this on `g5.48xlarge`:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/tensorrt-llm/compose-triton-tensorrt-llm.sh up

To test:

    ./triton-server/tests/test-triton-trtllm.sh
    ./triton-server/tests/test-triton-trtllm-concurrent.sh

To stop:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/tensorrt-llm/compose-triton-tensorrt-llm.sh down

#### vLLM Backend on CUDA

To launch, execute this on `g5.48xlarge`:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/vllm/compose-triton-vllm-cuda.sh up

To test:

    ./triton-server/tests/test-triton-vllm.sh
    ./triton-server/tests/test-triton-vllm-concurrent.sh

To stop:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/vllm/compose-triton-vllm-cuda.sh down

#### vLLM Backend on Neuronx

To launch Triton, execute this on `trn1.32xlarge`:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/vllm/compose-triton-vllm-neuronx.sh up

To test:

    ./triton-server/tests/test-triton-vllm.sh
    ./triton-server/tests/test-triton-vllm-concurrent.sh

To stop:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/vllm/compose-triton-vllm-neuronx.sh down


#### DJL-LMI Neuronx Backend

To launch:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/djl-lmi/compose-triton-djl-lmi-neuronx.sh up

To test:

    ./triton-server/tests/test-triton-djl-lmi-neuronx.sh
    ./triton-server/tests/test-triton-djl-lmi-neuronx-concurrent.sh

To stop the server:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/djl-lmi/compose-triton-djl-lmi-neuronx.sh down

