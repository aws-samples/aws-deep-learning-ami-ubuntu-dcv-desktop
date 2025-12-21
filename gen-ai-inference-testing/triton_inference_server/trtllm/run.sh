#!/bin/bash


export BUILD_SCRIPT="/scripts/build-engine.sh"
export MODEL_NAME="model_tensorrt_llm"

export TRITON_LAUNCH_SCRIPT=${TRITON_LAUNCH_SCRIPT:-"/opt/TensorRT-LLM/triton_backend/scripts/launch_triton_server.py"}

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"