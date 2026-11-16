#!/bin/bash

# 1. Presence Validation
if [ -z "$INFERENCE_SERVER" ] || [ -z "$INFERENCE_ENGINE" ] || [ -z "$DEVICE" ]; then
    echo "Error: Required environment variables (INFERENCE_SERVER, INFERENCE_ENGINE, DEVICE) must be set."
    exit 1
fi

# 2. DEVICE Validation
if [[ "$DEVICE" != "cuda" && "$DEVICE" != "neuron" ]]; then
    echo "Error: Invalid DEVICE '$DEVICE'. Must be 'cuda' or 'neuron'."
    exit 1
fi

# 3. SERVER and ENGINE logic Validation
case "$INFERENCE_SERVER" in
    "triton_inference_server" | "openai_server")
        export IMAGE="docker.io/library/${INFERENCE_SERVER}-${INFERENCE_ENGINE}:${DEVICE}"
        ;;
    "djl_serving")
        case "$INFERENCE_ENGINE" in
            "vllm")
                case "$DEVICE" in
                    "cuda")
                        export IMAGE="deepjavalibrary/djl-serving:0.38.0-lmi"
                        ;;
                    "neuron") 
                        export IMAGE="djl_serving-lmi:neuron"
                        ;;
                esac
                ;;
            *)
                echo "Error: For djl_serving, INFERENCE_ENGINE must be 'vllm'."
                exit 1
                ;;
        esac
        ;;
    *)
        echo "INFERENCE_SERVER must be triton_inference_server, openai_server, or djl_serving."
        exit 1
        ;;
esac

# Usage check for argument (up/down)
[ $# -ne 1 ] && echo "Usage: $0 <up/down>" && exit 1

ACTION=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 4. Map DEVICE to specific Compose File
COMPOSE_FILE="./compose/${DEVICE}.yaml"


# 5. Environment configuration
# Logic for ENCODER_TYPE check
if [ -n "$ENCODER_TYPE" ]; then
    export COMMAND="/scripts/run_${ENCODER_TYPE}.sh"
else
    export COMMAND="/scripts/run.sh"
fi

export HF_HOME=/snapshots/huggingface
SCRIPT_SOURCE="$SCRIPT_DIR/$INFERENCE_SERVER/$INFERENCE_ENGINE"

# 6. Execution logic
if [ "$ACTION" == "up" ]; then
    if [ ! -d "$SCRIPT_SOURCE" ]; then
        echo "Error: Source directory not found at $SCRIPT_SOURCE"
        exit 1
    fi

    echo "Starting $DEVICE environment ($INFERENCE_SERVER/$INFERENCE_ENGINE)..."

    # Build the image, but only on "up". djl_serving on cuda uses a public image and has
    # nothing to build; on neuron its build script is named after the lmi engine.
    build_script=""
    if [ "$INFERENCE_SERVER" == "djl_serving" ]; then
        [ "$DEVICE" == "neuron" ] && build_script="$SCRIPT_DIR/scripts/djl_serving-lmi-neuron.sh"
    else
        build_script="$SCRIPT_DIR/scripts/${INFERENCE_SERVER}-${INFERENCE_ENGINE}-${DEVICE}.sh"
    fi

    if [ -n "$build_script" ]; then
        [ ! -f "$build_script" ] && echo "Build script $build_script not found!" && exit 1
        build_log="/tmp/build-$(basename "$build_script" .sh).log"
        echo "Building $(basename "$build_script"). See logs: $build_log"
        if ! bash "$build_script" 1>"$build_log" 2>&1; then
            echo "Error: build failed. Last 20 lines of $build_log:"
            tail -20 "$build_log"
            exit 1
        fi
    fi

    mkdir -p "$HOME/scripts"
    cp "$SCRIPT_SOURCE"/*.* "$HOME/scripts/"
    chmod u+x "$HOME/scripts"/*.sh
    [ "${DEVICE}" == "neuron" ] && mkdir -p "$HOME/cache"

    docker compose -f "$COMPOSE_FILE" up -d
elif [ "$ACTION" == "down" ]; then
    echo "Stopping $DEVICE environment..."
    docker compose -f "$COMPOSE_FILE" down 
else
    echo "Invalid action: $ACTION. Use 'up' or 'down'."
    exit 1
fi