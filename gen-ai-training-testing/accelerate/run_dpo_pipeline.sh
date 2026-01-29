#!/bin/bash

set -e  # Exit on error

# DPO Pipeline Script
# Runs: SFT -> Checkpoint Conversion -> DPO Training

echo "================================================================================"
echo "DPO Training Pipeline"
echo "================================================================================"
echo ""

# Default configuration
BASE_MODEL="Qwen/Qwen3-8B"
ACCELERATE_CONFIG="accelerate_config.yaml"
SKIP_SFT=false
SKIP_CONVERT_SFT=false
SKIP_DPO=false

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
        --skip-dpo)
            SKIP_DPO=true
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
            echo "  --skip-dpo                  Skip DPO training"
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
    echo "Step 1/3: Supervised Fine-Tuning (SFT)"
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
    echo "Step 2/3: Convert SFT Checkpoint to HuggingFace Format"
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

# Step 3: DPO Training
if [ "$SKIP_DPO" = false ]; then
    echo "================================================================================"
    echo "Step 3/3: DPO Policy Training"
    echo "================================================================================"
    accelerate launch --config_file "$ACCELERATE_CONFIG" dpo_accelerate.py \
        --hf_model_id "$BASE_MODEL" \
        "${EXTRA_ARGS[@]}"
    echo ""
    echo "✓ DPO training completed"
    echo ""
else
    echo "⊘ Skipping DPO training"
    echo ""
fi

echo "================================================================================"
echo "DPO Pipeline Completed Successfully!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  SFT Model: results/$BASE_MODEL/checkpoint-*.hf_model"
echo "  DPO Policy: results/dpo_$BASE_MODEL/final"
echo ""
