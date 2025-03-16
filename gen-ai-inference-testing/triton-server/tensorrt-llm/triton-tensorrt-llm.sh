#!/bin/bash


export BUILD_SCRIPT="/scripts/build-engine.sh"
export MODEL_NAME="model_tensorrt_llm"
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"