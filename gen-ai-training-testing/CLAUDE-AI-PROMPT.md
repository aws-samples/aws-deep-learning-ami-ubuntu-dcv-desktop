# Claude AI Prompt Templates for Model Training

These prompt templates are designed to be used with the **Claude AI agent running on the AWS Deep Learning Desktop**. Copy and paste a prompt into the Claude AI agent's chat interface on the desktop, and the agent will autonomously execute the training workflow — building containers, configuring FSDP, launching distributed training, and reporting results.

**Prerequisites:**
- You are connected to an AWS Deep Learning Desktop instance (via DCV or similar)
- The Claude AI agent is running on the desktop with the `gen-ai-training-testing` folder open as its workspace
- Models are accessible via Hugging Face Hub or pre-downloaded to the local filesystem
- For gated models (e.g., Llama, Gemma): accept the model EULA on Hugging Face, and include your Hugging Face Access Token in the prompt

**How to use:**
1. Open the Claude AI agent on your Deep Learning Desktop
2. Copy one of the prompt templates below
3. Fill in the placeholders (model ID, dataset, hyperparameters) for your specific task
4. Paste the prompt into the agent's chat
5. The agent will build the container, configure the environment, and run training

---

## Text Training (Accelerate)

Text training supports PEFT/SFT (LoRA) and Continual Pre-Training (CPT) using HuggingFace Trainer with FSDP. To adapt the examples below, customize:
- The **model ID** and **dataset** for your use case
- The **max steps** or **num epochs** based on dataset size
- The **accelerate config** (`peft_accelerate_config.yaml` for PEFT, `cpt_accelerate_config.yaml` for CPT)
- The **training script** (`accelerate/text/peft_accelerate.py` for PEFT, `accelerate/text/cpt_accelerate.py` for CPT)

### Example 1: PEFT/SFT with Accelerate

```
Read and analyze the accelerate/text/peft_accelerate.py script and the 
peft_accelerate_config.yaml configuration to understand the SFT training workflow.
Confirm your understanding of the training pipeline, including data preparation,
model loading, LoRA configuration, FSDP settings, and checkpoint saving.

Then, run LoRA SFT training of Qwen/Qwen3-8B on this machine using the 
accelerate framework.

Training details:
   - Framework: accelerate
   - Training type: PEFT/SFT (LoRA)
   - Model: Qwen/Qwen3-8B
   - Dataset: cognitivecomputations/dolphin (flan1m-alpaca-uncensored)
   - Max sequence length: 2048
   - Max steps: 500 (for quick validation)

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Build the Docker container using containers/Dockerfile.accelerate
- Use the peft_accelerate_config.yaml for FSDP configuration
- Launch training with: accelerate launch --config_file peft_accelerate_config.yaml 
  accelerate/text/peft_accelerate.py
- Auto-detect available GPUs and adjust configuration accordingly
- Monitor GPU memory usage during training

WHAT TO REPORT:
1. Does training start successfully? (Yes/No)
2. Training loss after first 100 steps
3. Peak GPU memory usage per device
4. Any errors encountered (include root cause analysis)
5. Location of saved checkpoints
```

### Example 2: Continual Pre-Training with Accelerate

```
Read and analyze the accelerate/text/cpt_accelerate.py script and the 
cpt_accelerate_config.yaml configuration to understand the CPT training workflow.
Confirm your understanding of the training pipeline, including data preparation,
full model training (no LoRA), FSDP activation checkpointing, and epoch-based 
checkpoint saving.

Then, run continual pre-training of Qwen/Qwen3-8B on this machine using the 
accelerate framework.

Training details:
   - Framework: accelerate
   - Training type: Continual Pre-Training (full model, no LoRA)
   - Model: Qwen/Qwen3-8B
   - Dataset: wikimedia/wikipedia (20231101.en)
   - Max sequence length: 4096
   - Num epochs: 1 (for quick validation)
   - Save steps: 500

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Build the Docker container using containers/Dockerfile.accelerate
- Use the cpt_accelerate_config.yaml for FSDP configuration
- Launch training with: accelerate launch --config_file cpt_accelerate_config.yaml 
  accelerate/text/cpt_accelerate.py
- Auto-detect available GPUs and adjust configuration accordingly
- Monitor GPU memory usage during training

WHAT TO REPORT:
1. Does training start successfully? (Yes/No)
2. Training loss after first 100 steps
3. Peak GPU memory usage per device
4. Any errors encountered (include root cause analysis)
5. Location of saved checkpoints
```

---

## Text Training (Ray Train)

Ray Train provides the same training capabilities as accelerate but with Ray's distributed execution, fault tolerance, and checkpoint management. To adapt the examples below, customize:
- The **model ID** and **dataset** for your use case
- The **training script** (`ray_train/text/ray_train_sft.py` for PEFT, `ray_train/text/cpt_ray_train.py` for CPT)

### Example 3: PEFT/SFT with Ray Train

```
Read and analyze the ray_train/text/ray_train_sft.py script to understand the 
Ray Train SFT workflow. Confirm your understanding of the training pipeline, 
including Ray initialization, FSDP configuration, data preparation, LoRA setup,
and checkpoint management via Ray Train callbacks.

Then, run LoRA SFT training of Qwen/Qwen3-8B on this machine using Ray Train.

Training details:
   - Framework: ray_train
   - Training type: PEFT/SFT (LoRA)
   - Model: Qwen/Qwen3-8B
   - Dataset: cognitivecomputations/dolphin (flan1m-alpaca-uncensored)
   - Max sequence length: 2048
   - Max steps: 500 (for quick validation)

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Build the Docker container using containers/Dockerfile.ray_train
- Launch training with: python ray_train/text/ray_train_sft.py
- Auto-detect available GPUs — Ray Train will use all available GPUs automatically
- Monitor GPU memory usage during training

WHAT TO REPORT:
1. Does training start successfully? (Yes/No)
2. Training loss after first 100 steps
3. Peak GPU memory usage per device
4. Any errors encountered (include root cause analysis)
5. Location of saved checkpoints
```

### Example 4: Continual Pre-Training with Ray Train

```
Read and analyze the ray_train/text/cpt_ray_train.py script to understand the 
Ray Train CPT workflow. Confirm your understanding of the training pipeline, 
including FSDP activation checkpointing configuration in the fsdp_config dict,
full model training, and epoch-based checkpoint saving.

Then, run continual pre-training of Qwen/Qwen3-8B on this machine using Ray Train.

Training details:
   - Framework: ray_train
   - Training type: Continual Pre-Training (full model, no LoRA)
   - Model: Qwen/Qwen3-8B
   - Dataset: wikimedia/wikipedia (20231101.en)
   - Max sequence length: 4096
   - Num epochs: 1 (for quick validation)
   - Save steps: 500

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Build the Docker container using containers/Dockerfile.ray_train
- Launch training with: python ray_train/text/cpt_ray_train.py
- Auto-detect available GPUs — Ray Train will use all available GPUs automatically
- Monitor GPU memory usage during training

WHAT TO REPORT:
1. Does training start successfully? (Yes/No)
2. Training loss after first 100 steps
3. Peak GPU memory usage per device
4. Any errors encountered (include root cause analysis)
5. Location of saved checkpoints
```

---

## Multimodal Training (Vision-Language)

Multimodal training supports vision-language models (e.g., Qwen3-VL) for both PEFT/SFT and CPT. The adapter pattern handles model-specific image processing automatically. To adapt the examples below, customize:
- The **model ID** (e.g., `Qwen/Qwen3-VL-8B-Instruct`)
- The **HuggingFace dataset** for vision-language data
- The **framework** (`accelerate` or `ray_train`)
- The **training script** (under `accelerate/multimodal/vision_language/` or `ray_train/multimodal/vision_language/`)

### Example 5: Multimodal PEFT/SFT with Accelerate

```
Read and analyze the accelerate/multimodal/vision_language/peft_accelerate.py 
script, the adapter pattern (adapters/ and base/ directories), and the 
peft_accelerate_config.yaml configuration to understand the VLM SFT workflow.
Confirm your understanding of the training pipeline, including adapter-based 
model loading, image processing, LoRA configuration, and vision encoder freezing.

Then, run LoRA SFT training of Qwen/Qwen3-VL-8B-Instruct on this machine 
using the accelerate framework.

Training details:
   - Framework: accelerate
   - Training type: Multimodal PEFT/SFT (LoRA)
   - Model: Qwen/Qwen3-VL-8B-Instruct
   - Dataset: lmms-lab/LLaVA-NeXT-Data (auto-downloaded)
   - Max sequence length: 4096
   - Max steps: 500 (for quick validation)
   - Freeze vision encoder: yes
   - Per device batch size: 1 (required for dynamic resolution)

IMPORTANT CONSTRAINTS:
- Do NOT modify any existing files in the repository
- Build the Docker container using containers/Dockerfile.accelerate
- Use the peft_accelerate_config.yaml for FSDP configuration
- Launch training with: accelerate launch --config_file peft_accelerate_config.yaml 
  accelerate/multimodal/vision_language/peft_accelerate.py
- Auto-detect available GPUs and adjust configuration accordingly
- Dataset preparation may take 1-2 hours on first run (images are saved to disk)

WHAT TO REPORT:
1. Does training start successfully? (Yes/No)
2. Training loss after first 100 steps
3. Peak GPU memory usage per device
4. Any errors encountered (include root cause analysis)
5. Location of saved checkpoints

NOTE: First run requires downloading and preparing the dataset which includes 
saving images to disk. This is a one-time operation.
```

---

## Supported Configurations

### Frameworks
- **Accelerate**: HuggingFace Trainer with FSDP, configured via YAML config files
- **Ray Train**: Ray's distributed training with FSDP, configured via fsdp_config dict in code

### Training Types
| Type | Script (Accelerate) | Script (Ray Train) | Description |
|------|--------------------|--------------------|-------------|
| PEFT/SFT | `accelerate/text/peft_accelerate.py` | `ray_train/text/ray_train_sft.py` | LoRA fine-tuning on instruction data |
| CPT | `accelerate/text/cpt_accelerate.py` | `ray_train/text/cpt_ray_train.py` | Full model continual pre-training |
| VLM PEFT | `accelerate/multimodal/vision_language/peft_accelerate.py` | `ray_train/multimodal/vision_language/ray_train_vlm.py` | Vision-language LoRA fine-tuning |
| VLM CPT | `accelerate/multimodal/vision_language/cpt_accelerate.py` | `ray_train/multimodal/vision_language/cpt_ray_train_vlm.py` | Vision-language continual pre-training |
| DPO | `accelerate/text/dpo_accelerate.py` | — | Direct Preference Optimization |
| PPO | `accelerate/text/ppo_accelerate.py` | — | PPO-RLHF alignment |

### Supported Hardware
- **NVIDIA GPUs**: Detected automatically, FSDP distributes across all available GPUs
- Recommended: 8x A100 (40/80GB), 8x L40S (48GB), 8x H100 (80GB)

### Docker Containers
| Container | Dockerfile | Use Case |
|-----------|-----------|----------|
| accelerate | `containers/Dockerfile.accelerate` | Accelerate + FSDP training |
| ray_train | `containers/Dockerfile.ray_train` | Ray Train + FSDP training |

---
