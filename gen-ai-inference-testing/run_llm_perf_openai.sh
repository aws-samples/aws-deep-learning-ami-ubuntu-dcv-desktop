#!/bin/bash

# You need to create a conda environment with Python 3.10 
# conda create -n llmperf python=3.10
# You need to git clone https://github.com/ray-project/llmperf.git
# and specify the location of the llmperf directory in the LLM_PERF_DIR environment variable.
# cd $LLM_PERF_DIR
# pip install -e .
# conda activate llmperf
# This script will create a RESULTS_DIR directory where the results will be stored.

# Set defaults for environment variables
[  -z "$MODEL"  ] && echo "MODEL environment variable must exist" && exit 1
[  -z "$LLM_PERF_DIR"  ] && echo "LLM_PERF_DIR environment variable must exist" && exit 1
[  -z "$RESULTS_DIR"  ] && echo "RESULTS_DIR environment variable must exist" && exit 1

# Set defaults for environment variables
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8080/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-key}"

MEAN_INPUT_TOKENS="${MEAN_INPUT_TOKENS:-550}"
STDDEV_INPUT_TOKENS="${STDDEV_INPUT_TOKENS:-150}"
MEAN_OUTPUT_TOKENS="${MEAN_OUTPUT_TOKENS:-350}"
STDDEV_OUTPUT_TOKENS="${STDDEV_OUTPUT_TOKENS:-10}"
MAX_NUM_COMPLETED_REQUESTS="${MAX_NUM_COMPLETED_REQUESTS:-100}"
TIMEOUT="${TIMEOUT:-600}"
NUM_CONCURRENT_REQUESTS="${NUM_CONCURRENT_REQUESTS:-10}"
LLM_API="${LLM_API:-openai}"

if [ -d "$RESULTS_DIR" ]; then
    echo "Error: RESULTS_DIR '$RESULTS_DIR' already exists" >&2
    exit 1
fi

mkdir -p $RESULTS_DIR
LOGFILE="${RESULTS_DIR}/llm_perf.log"

cd $LLM_PERF_DIR


python token_benchmark_ray.py   \
    --model $MODEL   \
    --mean-input-tokens $MEAN_INPUT_TOKENS   \
    --stddev-input-tokens $STDDEV_INPUT_TOKENS   \
    --mean-output-tokens $MEAN_OUTPUT_TOKENS   \
    --stddev-output-tokens $STDDEV_OUTPUT_TOKENS   \
    --max-num-completed-requests $MAX_NUM_COMPLETED_REQUESTS   \
    --timeout $TIMEOUT   \
    --num-concurrent-requests $NUM_CONCURRENT_REQUESTS   \
    --llm-api $LLM_API   \
    --results-dir $RESULTS_DIR 2>&1 | tee -a ${LOGFILE}
