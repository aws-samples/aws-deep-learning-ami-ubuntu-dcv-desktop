#!/bin/bash
# Build openai_server-vllm:neuron for the Neuron hardware this host has.
#
# Two vLLM-on-Neuron stacks share containers/openai_server-vllm-neuron:
#   legacy  vLLM 0.13 DLC (Neuron SDK 2.28, torch-neuronx/XLA). inf2, trn1 -- and the
#           default on any host whose instance family is not trn2/trn3.
#   native  vLLM 0.29 + vllm-neuron plugin on the native TorchNeuron backend
#           (Neuron SDK 2.33 DLC). Neuron documents this stack for trn2 and trn3 only.
#
# NEURON_VLLM_STACK=auto (default) picks native on trn2/trn3 and legacy elsewhere;
# set it to native or legacy to override (e.g. legacy on trn2, or native from a
# non-Neuron build host). The image records its stack in /opt/program/neuron-stack.

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

NATIVE_BASE_IMAGE="${NATIVE_BASE_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04}"
STACK="${NEURON_VLLM_STACK:-auto}"
if [ "$STACK" == "auto" ]; then
    family=$(neuron-ls 2>/dev/null | awk -F': ' '/instance-type/ {split($2, a, "."); print a[1]}')
    case "$family" in
        trn2*|trn3*) STACK=native ;;
        *)           STACK=legacy ;;
    esac
    # Encoder tasks (launch.sh passes ENCODER_TYPE) run openai_server/vllm/run_<task>.sh,
    # which only the legacy stack implements, so they keep the legacy stack everywhere.
    if [ "$STACK" == "native" ] && [ -n "$ENCODER_TYPE" ]; then
        STACK=legacy
        echo "ENCODER_TYPE=$ENCODER_TYPE: encoder tasks use the legacy stack"
    fi
    echo "NEURON_VLLM_STACK=auto: instance family '${family:-unknown}' -> $STACK"
elif [ "$STACK" == "native" ] && [ -n "$ENCODER_TYPE" ]; then
    # An explicit native request for an encoder task would otherwise fail later, deep in
    # vLLM: the encoder run scripts only set up the legacy stack.
    echo "NEURON_VLLM_STACK=native is not supported for encoder tasks (ENCODER_TYPE=$ENCODER_TYPE)."
    echo "Encoder tasks use the legacy stack: unset NEURON_VLLM_STACK or set it to legacy."
    exit 1
fi

cd $DIR/containers/openai_server-vllm-neuron

case "$STACK" in
    legacy)
        docker buildx build -t openai_server-vllm:neuron .
        ;;
    native)
        docker buildx build -t openai_server-vllm:neuron \
            --build-arg BASE_IMAGE="$NATIVE_BASE_IMAGE" \
            --build-arg NEURON_VLLM_STACK=native \
            --build-arg PJRT_DEVICE= \
            .
        ;;
    *)
        echo "NEURON_VLLM_STACK must be auto, native or legacy (got '$STACK')"
        exit 1
        ;;
esac
