# Gen AI Inference Testing

A comprehensive testing framework for evaluating generative AI inference performance across multiple inference servers, engines, and hardware accelerators. This project provides automated testing capabilities for both decoder (text generation) and encoder (embeddings, classification) models using Locust load testing and LiteLLM proxy integration.

## Overview

This framework supports testing of Large Language Models (LLMs) and encoder models on both NVIDIA GPUs and AWS AI chips (Neuron) using containerized inference servers. The main entry points are Jupyter notebooks that guide you through the complete testing workflow.

## Key Features

- **Multiple Inference Servers**: Triton Inference Server, OpenAI Server (vLLM), and DJL Serving
- **Multiple Engines**: vLLM, TensorRT-LLM, and Python backends
- **Hardware Support**: NVIDIA CUDA GPUs and AWS Neuron chips
- **Testing Frameworks**: Locust load testing and LiteLLM proxy testing
- **Model Types**: Text generation, multimodal, code generation, embeddings, classification, and reranking
- **Automated Containerization**: Docker-based deployment with automated container building

## Project Structure

```
gen-ai-inference-testing/
├── litellm.ipynb              # LiteLLM testing notebook (main entry point)
├── locust_decoder.ipynb       # Locust testing for decoder models
├── locust_encoder.ipynb       # Locust testing for encoder models
├── launch.sh                  # Main launcher script for inference servers
├── run_locust.sh             # Locust test execution script
├── endpoint_user.py          # Locust user implementation
├── custom_endpoint_handler.py # LiteLLM custom endpoint handler
├── config/                   # Configuration files
│   ├── decoder/             # Decoder model configurations
│   │   ├── text_only/       # Text-only model configs
│   │   ├── text_image/      # Multimodal model configs
│   │   └── code_gen/        # Code generation model configs
│   ├── encoder/             # Encoder model configurations
│   │   ├── embeddings/      # Text embedding configs
│   │   ├── reranker/        # Reranking model configs
│   │   ├── sequence_classification/
│   │   ├── token_classification/
│   │   └── masked_lm/       # Masked language model configs
│   └── litellm/             # LiteLLM proxy configuration
├── modules/                  # Test prompt generators
│   ├── inst-semeval2017/    # Text generation prompts
│   ├── ms-marco/            # Reranking prompts
│   ├── squad-context/       # Context-based prompts
│   ├── sahil2801-codealpca20k/ # Code generation prompts
│   └── [other datasets]/   # Various specialized prompt generators
├── scripts/                 # Container build scripts
├── containers/              # Docker container definitions
├── compose/                 # Docker Compose files
├── triton_inference_server/ # Triton server configurations
├── openai_server/           # OpenAI server configurations
└── djl_serving/             # DJL serving configurations
```

## Supported Configurations

### Inference Servers
- **Triton Inference Server**: NVIDIA's inference server with multiple backend support
- **OpenAI Server**: vLLM's OpenAI-compatible API server
- **DJL Serving**: Deep Java Library serving framework

### Inference Engines
- **vLLM**: High-performance LLM inference engine
- **TensorRT-LLM**: NVIDIA's optimized inference engine
- **Python**: Custom Python backend for specialized models

### Model Types

#### Decoder Models
- **Text Only**: Standard text generation models (Llama, Mistral, etc.)
- **Text + Image**: Multimodal models supporting both text and image inputs
- **Code Generation**: Specialized code generation models

#### Encoder Models
- **Embeddings**: Text embedding models for semantic search
- **Reranker**: Document reranking models
- **Sequence Classification**: Text classification models
- **Token Classification**: Named entity recognition models
- **Masked Language Models**: BERT-style models

## Configuration System

The project uses YAML configuration files that specify:
- API endpoints and request templates
- Prompt generator modules and classes
- Template keys for dynamic content injection

Example configuration:
```yaml
endpoint_url: "http://localhost:8080/v1/chat/completions"
module_name: "llama3_prompt_generator"
module_dir: "modules/inst-semeval2017"
prompt_generator: "PromptGenerator"
template:
  model: ""
  max_tokens: 2048
  messages:
    - role: "user"
      content:
        - type: "text"
          text: ""
template_keys: ["model", "messages.[0].content.[0].text"]
```

## Testing Frameworks

### Locust Testing
- Concurrent load testing with configurable users and workers
- Real-time performance metrics
- CSV output for analysis
- Customizable test duration and spawn rates

### LiteLLM Testing
- OpenAI-compatible API standardization
- Custom endpoint handler for non-standard APIs
- Proxy-based testing architecture
- Integration with existing OpenAI tooling

## Prompt Generators

The `modules/` directory contains specialized prompt generators for different datasets and use cases:

- **inst-semeval2017**: Technical writing and keyphrase generation
- **ms-marco**: Document reranking pairs
- **squad-context**: Question-answering contexts
- **sahil2801-codealpca20k**: Code generation prompts
- **thudm-longbench**: Long context evaluation
- **mminstruction-m3it**: Multimodal instruction following

Each module provides dataset-specific prompt generation with appropriate formatting for different model types.

## Container Management

The framework automatically builds and manages Docker containers for different inference configurations:

```bash
# Build containers for detected hardware
bash scripts/build-containers.sh

# Launch inference server
bash launch.sh up

# Stop inference server
bash launch.sh down
```

## Performance Optimization

- **Tensor Parallelism**: Automatic configuration based on available hardware
- **Dynamic Batching**: Configurable batch sizes for optimal throughput
- **Model Caching**: EFS-based model storage for faster startup
- **Hardware Detection**: Automatic CUDA/Neuron detection and configuration

## Output and Analysis

Test results are saved in structured formats:
- **Locust**: CSV files with detailed performance metrics
- **LiteLLM**: JSON responses with timing information
- **Logs**: Comprehensive logging for debugging and analysis

## Advanced Usage

### Custom Prompt Generators
Create custom prompt generators by implementing the generator interface. The generator must return a list of items that map 1:1 to the template keys defined in your configuration file (excluding "model"). These items are injected into the template in the order specified by `template_keys`.

```python
class CustomPromptGenerator:
    def __init__(self):
        # Initialize your dataset or prompt source
        pass
    
    def __call__(self):
        # Yield lists of values that correspond to template_keys
        # Example: if template_keys = ["messages.[0].content.[0].text"]
        # then yield a single-item list with the prompt text
        for prompt in your_prompts:
            yield [prompt]  # Single item for single template key
            
        # For multiple template keys (excluding "model"):
        # if template_keys = ["messages.[0].content.[0].text", "max_tokens"]
        # then yield [prompt_text, max_tokens_value]
```

**Important Notes**:
- The number of items in the returned list must match the number of `template_keys` (excluding "model")
- Each item will be injected into the corresponding template key position
- **Model injection is handled automatically**: If "model" appears in `template_keys`, it is injected externally from the `MODEL` environment variable. Your prompt generator should NOT return a model value.

### Custom Configurations
Add new model configurations by creating YAML files in the appropriate `config/` subdirectory following the existing template structure.

### Environment Variables
Key environment variables for customization:
- `HF_TOKEN`: Hugging Face access token
- `MAX_MODEL_LEN`: Maximum model context length
- `TENSOR_PARALLEL_SIZE`: Tensor parallelism configuration
- `MAX_NUM_SEQS`: Maximum batch size

## Troubleshooting

- **Container Build Failures**: Check `/tmp/build.log` for detailed error messages
- **Model Download Issues**: Verify HF_TOKEN and model access permissions
- **Memory Issues**: Adjust tensor parallel size or use smaller models
- **Network Timeouts**: Increase timeout values in configuration files

## Contributing

When adding new inference servers, engines, or model types:
1. Add appropriate configuration files in `config/`
2. Create container definitions in `containers/`
3. Add build scripts in `scripts/`
4. Update documentation and examples

This framework provides a comprehensive foundation for evaluating generative AI inference performance across diverse hardware and software configurations.