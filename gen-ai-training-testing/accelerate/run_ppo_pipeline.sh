#!/bin/bash

set -e  # Exit on error

# RLHF Pipeline Script
# Runs: SFT -> Checkpoint Conversion -> Reward Model -> PPO Training

echo "================================================================================"
echo "RLHF Training Pipeline"
echo "================================================================================"
echo ""

# Default configuration
BASE_MODEL="Qwen/Qwen3-8B"
ACCELERATE_CONFIG="accelerate_config.yaml"
SKIP_SFT=false
SKIP_CONVERT_SFT=false
SKIP_REWARD=false
SKIP_CONVERT_REWARD=false
SKIP_PPO=false

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --skip-convert-sft)
            SKIP_CONVERT_SFT=true
            shift
            ;;
        --skip-reward)
            SKIP_REWARD=true
            shift
            ;;
        --skip-convert-reward)
            SKIP_CONVERT_REWARD=true
            shift
            ;;
        --skip-ppo)
            SKIP_PPO=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS] [EXTRA_ARGS]"
            echo ""
            echo "Pipeline Control:"
            echo "  --base-model MODEL          Base model ID (default: Qwen/Qwen3-8B)"
            echo "  --accelerate-config FILE    Accelerate config (default: accelerate_config.yaml)"
            echo "  --skip-sft                  Skip SFT training"
            echo "  --skip-convert-sft          Skip SFT checkpoint conversion"
            echo "  --skip-reward               Skip reward model training"
            echo "  --skip-convert-reward       Skip reward checkpoint conversion"
            echo "  --skip-ppo                  Skip PPO training"
            echo ""
            echo "All other arguments are passed to the Python scripts."
            echo ""
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Accelerate Config: $ACCELERATE_CONFIG"
echo "  Extra Args: ${EXTRA_ARGS[@]}"
echo ""

# Step 1: Supervised Fine-Tuning
if [ "$SKIP_SFT" = false ]; then
    echo "================================================================================"
    echo "Step 1/5: Supervised Fine-Tuning (SFT)"
    echo "================================================================================"
    accelerate launch --config_file "$ACCELERATE_CONFIG" peft_accelerate.py \
        --hf_model_id "$BASE_MODEL" \
        "${EXTRA_ARGS[@]}"
    echo ""
    echo "✓ SFT training completed"
    echo ""
else
    echo "⊘ Skipping SFT training"
    echo ""
fi

# Step 2: Convert SFT Checkpoint to HuggingFace Format
if [ "$SKIP_CONVERT_SFT" = false ]; then
    echo "================================================================================"
    echo "Step 2/5: Convert SFT Checkpoint to HuggingFace Format"
    echo "================================================================================"
    python convert_checkpoint_to_hf.py \
        --base_model "$BASE_MODEL"
    echo ""
    echo "✓ SFT checkpoint conversion completed"
    echo ""
else
    echo "⊘ Skipping SFT checkpoint conversion"
    echo ""
fi

# Step 3: Train Reward Model
if [ "$SKIP_REWARD" = false ]; then
    echo "================================================================================"
    echo "Step 3/5: Train Reward Model"
    echo "================================================================================"
    accelerate launch --config_file "$ACCELERATE_CONFIG" reward_model_accelerate.py \
        --hf_model_id "$BASE_MODEL" \
        "${EXTRA_ARGS[@]}"
    echo ""
    echo "✓ Reward model training completed"
    echo ""
else
    echo "⊘ Skipping reward model training"
    echo ""
fi

# Step 4: Convert Reward Model Checkpoint to HuggingFace Format
if [ "$SKIP_CONVERT_REWARD" = false ]; then
    echo "================================================================================"
    echo "Step 4/5: Convert Reward Model Checkpoint to HuggingFace Format"
    echo "================================================================================"
    python convert_checkpoint_to_hf.py \
        --base_model "$BASE_MODEL" \
        --checkpoints_dir "results/reward_$BASE_MODEL"
    echo ""
    echo "✓ Reward model checkpoint conversion completed"
    echo ""
else
    echo "⊘ Skipping reward model checkpoint conversion"
    echo ""
fi

# Step 5: PPO Training
if [ "$SKIP_PPO" = false ]; then
    echo "================================================================================"
    echo "Step 5/5: PPO Policy Training"
    echo "================================================================================"
    accelerate launch --config_file "$ACCELERATE_CONFIG" ppo_accelerate.py \
        --hf_model_id "$BASE_MODEL" \
        "${EXTRA_ARGS[@]}"
    echo ""
    echo "✓ PPO training completed"
    echo ""
else
    echo "⊘ Skipping PPO training"
    echo ""
fi

echo "================================================================================"
echo "RLHF Pipeline Completed Successfully!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  SFT Model: results/$BASE_MODEL/checkpoint-*.hf_model"
echo "  Reward Model: results/reward_$BASE_MODEL/checkpoint-*.hf_model"
echo "  PPO Policy: results/ppo_$BASE_MODEL/final"
echo ""
