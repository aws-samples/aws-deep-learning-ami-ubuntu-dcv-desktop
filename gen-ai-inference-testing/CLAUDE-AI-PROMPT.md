# Claude AI Prompt Templates for Model Testing

These prompt templates are designed to be used with the **Claude AI agent running on the AWS Deep Learning Desktop**. Copy and paste a prompt into the Claude AI agent's chat interface on the desktop, and the agent will autonomously execute the inference testing workflow — building containers, launching servers, running load tests, and reporting results.

**Prerequisites:**
- You are connected to an AWS Deep Learning Desktop instance (via DCV or similar)
- The Claude AI agent is running on the desktop with the `gen-ai-inference-testing` folder open as its workspace
- Models are accessible via Hugging Face Hub or pre-downloaded to the local filesystem
- For gated models (e.g., Llama, Gemma): accept the model EULA on Hugging Face, and include your Hugging Face Access Token in the prompt

**How to use:**
1. Open the Claude AI agent on your Deep Learning Desktop
2. Copy one of the prompt templates below
3. Fill in the placeholders (model ID, token lengths, etc.) for your specific model
4. Paste the prompt into the agent's chat
5. The agent will analyze the notebook, configure the infrastructure, and run the tests

Use this template to test models with the inference testing framework. Fill in the placeholders with your specific values.


## For Encoder Models

Encoder models include embeddings, rerankers, sequence classification, token classification, and masked language models. To adapt the examples below for a different encoder model type, customize:
- The **model ID** (e.g., `BAAI/bge-m3`, `BAAI/bge-reranker-large`)
- The **maximum input token length** and **batch size** based on the model's specifications
- The notebook reference stays `locust_encoder.ipynb` for all encoder types

Supported inference configurations for encoder models:

| Hardware | Server Options | Backend Options |
|----------|----------------|-----------------|
| CUDA     | Triton Inference Server, OpenAI Server | Python, vLLM |
| Neuron   | Triton Inference Server, OpenAI Server | Python, vLLM |

### Example 1: Embeddings

```
Read and analyze the locust_encoder.ipynb notebook and dependent configuration 
YAML files and scripts to understand the inference testing workflow. 
Confirm your understanding of all the steps in the workflow.

Then, test the BAAI/bge-m3 embeddings model on this machine.

Model details:
   - Maximum input token length = 512 
   - Maximum number of sequences in a batch = 8

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Create new temporary files as needed for testing (notebooks, configs, scripts)
- Create all temporary files under tmp sub-folder for easy identification
- Use optimal configurations for the inference server (e.g., config.pbtxt, 
  compose/*.yaml) and backend (e.g., vLLM config, Python backend settings) 
  based on model type and detected hardware
- For encoder models: Set TP=1 (encoder models don't benefit from tensor 
  parallelism). The infrastructure will automatically utilize available devices.
- Follow the workflow from the locust_encoder.ipynb, exactly, adapting only parameters
- Verify ALL Backend servers are running before Locust performance testing

WHAT TO REPORT:
1. Does the model work? (Yes/No)
2. Any errors encountered (include root cause analysis and which cell failed)
3. Locust performance metrics (throughput, latency) if testing completes

NOTE: First run on Neuron requires model compilation which may take 30-60 minutes.
Time to compile depends on maximum sequence length and maximum number of sequences in a batch. 
The infrastructure creates multiple inference server instances based on available hardware - this is normal.
```

### Example 2: Reranker

```
Read and analyze the locust_encoder.ipynb notebook and dependent configuration 
YAML files and scripts to understand the inference testing workflow. 
Confirm your understanding of all the steps in the workflow.

Then, test the BAAI/bge-reranker-large reranker model on this machine.

Model details:
   - Maximum input token length = 512 
   - Maximum number of sequences in a batch = 8

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Create new temporary files as needed for testing (notebooks, configs, scripts)
- Create all temporary files under tmp sub-folder for easy identification
- Use optimal configurations for the inference server (e.g., config.pbtxt, 
  compose/*.yaml) and backend (e.g., vLLM config, Python backend settings) 
  based on model type and detected hardware
- For encoder models: Set TP=1 (encoder models don't benefit from tensor 
  parallelism). The infrastructure will automatically utilize available devices.
- Follow the workflow from the locust_encoder.ipynb, exactly, adapting only parameters
- Verify ALL Backend servers are running before Locust performance testing

WHAT TO REPORT:
1. Does the model work? (Yes/No)
2. Any errors encountered (include root cause analysis and which cell failed)
3. Locust performance metrics (throughput, latency) if testing completes

NOTE: First run on Neuron requires model compilation which may take 30-60 minutes.
Time to compile depends on maximum sequence length and maximum number of sequences in a batch. 
The infrastructure creates multiple inference server instances based on available hardware - this is normal.
```

---

## Decoder Models

Decoder models include text generation, multimodal (text+image), and code generation models. To adapt the example below for a different decoder model, customize:
- The **model ID** (e.g., `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`)
- The **maximum input token length** and **batch size** based on the model's specifications
- The notebook reference stays `locust_decoder.ipynb` for all decoder types

Supported inference configurations for decoder models:

| Hardware | Server Options | Backend Options |
|----------|----------------|-----------------|
| CUDA     | Triton Inference Server, OpenAI Server, DJL Serving | vLLM, TensorRT-LLM |
| Neuron   | Triton Inference Server, OpenAI Server, DJL Serving | vLLM |

### Example 1: Qwen/Qwen3-8B

```
Read and analyze the locust_decoder.ipynb notebook and dependent configuration 
YAML files and scripts to understand the inference testing workflow. 
Confirm your understanding of all the steps in the workflow.

Then, test the Qwen/Qwen3-8B model on this machine.

Model details:
   - Maximum input token length = 8192 
   - Maximum number of sequences in a batch = 8

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Create new temporary files as needed for testing (notebooks, configs, scripts)
- Create all temporary files under tmp sub-folder for easy identification
- Use optimal configurations for the inference server (e.g., config.pbtxt, 
  compose/*.yaml) and backend (e.g., vLLM config, Python backend settings) 
  based on model type and detected hardware
- For decoder models: Auto-detect optimal TP size based on model size and 
  available hardware (use multiple devices as appropriate)
- Follow the workflow from the notebook exactly, adapting only parameters
- Verify ALL Backend servers are running before Locust performance testing

WHAT TO REPORT:
1. Does the model work? (Yes/No)
2. Any errors encountered (include root cause analysis and which cell failed)
3. Locust performance metrics (throughput, latency) if testing completes

NOTE: First run on Neuron requires model compilation which may take 30-60 minutes.
Time to compile depends on maximum sequence length and maximum number of sequences in a batch. 
The infrastructure creates multiple inference server instances based on available hardware - this is normal.
```



---

## Supported Hardware
- **NVIDIA GPUs**: Detected as `cuda` device
- **AWS Neuron (Inferentia/Trainium)**: Detected as `neuron` device

---
