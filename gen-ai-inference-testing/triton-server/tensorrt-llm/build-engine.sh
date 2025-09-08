#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_NUM_SEQS:=8}
: ${MAX_MODEL_LEN:=8192}
: ${OMP_NUM_THREADS:=16}
: ${TRTLLM_CONVERT_CKPT_SCRIPT:=/opt/TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py}

TEMP_DIR=$(mktemp -d)
CKPT_PATH=$TEMP_DIR/trtllm_ckpt
[[ -d $CKPT_PATH ]] && rm -rf $CKPT_PATH

echo "Converting Hugging Face ckpt to TensorRT-LLM ckpt" 
python3 \
${TRTLLM_CONVERT_CKPT_SCRIPT} \
--model_dir=$MODEL_ID \
--dtype=auto \
--output_dir=$CKPT_PATH \
--tp_size=$TENSOR_PARALLEL_SIZE

ENGINE_DIR=$TEMP_DIR/trtllm_engine
[[ -d $ENGINE_DIR ]] && rm -rf $ENGINE_DIR

echo "Build TensorRT-LLM engine"

trtllm-build \
--checkpoint_dir ${CKPT_PATH} \
--max_num_tokens ${MAX_MODEL_LEN} \
--gpus_per_node ${TENSOR_PARALLEL_SIZE} \
--remove_input_padding enable \
--paged_kv_cache enable \
--context_fmha enable \
--output_dir ${ENGINE_DIR} \
--max_batch_size ${MAX_NUM_SEQS}

MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO

echo "Build Triton TensorRT-LLM model"
cd /opt/TensorRT-LLM/triton_backend

TOKENIZER_DIR=$MODEL_ID
DECOUPLED_MODE=false
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=100
FILL_TEMPLATE_SCRIPT=tools/fill_template.py
encoder_input_features_data_type=TYPE_BF16

MODEL_NAME=model
cp -r all_models/inflight_batcher_llm/preprocessing $MODEL_REPO/${MODEL_NAME}_preprocessing
cp -r all_models/inflight_batcher_llm/postprocessing $MODEL_REPO/${MODEL_NAME}_postprocessing
cp -r all_models/inflight_batcher_llm/tensorrt_llm_bls $MODEL_REPO/${MODEL_NAME}_tensorrt_llm_bls
cp -r all_models/inflight_batcher_llm/ensemble $MODEL_REPO/${MODEL_NAME}
cp -r all_models/inflight_batcher_llm/tensorrt_llm $MODEL_REPO/${MODEL_NAME}_tensorrt_llm
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${MAX_NUM_SEQS},preprocessing_instance_count:${INSTANCE_COUNT} 
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${MAX_NUM_SEQS},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_NUM_SEQS},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},tensorrt_llm_model_name:${MODEL_NAME}_tensorrt_llm
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt triton_max_batch_size:${MAX_NUM_SEQS}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_NUM_SEQS},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,encoder_input_features_data_type:${encoder_input_features_data_type},triton_backend:tensorrtllm
sed -i 's/name: "preprocessing"/name: "model_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt
sed -i 's/name: "postprocessing"/name: "model_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt
sed -i 's/name: "tensorrt_llm_bls"/name: "model_tensorrt_llm_bls"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt
sed -i 's/name: "ensemble"/name: "model"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/name: "tensorrt_llm"/name: "model_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt
sed -i 's/model_name: "preprocessing"/model_name: "model_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/model_name: "postprocessing"/model_name: "model_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/model_name: "tensorrt_llm"/model_name: "model_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/${logits_datatype}/TYPE_FP32/g' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/${logits_datatype}/TYPE_FP32/g' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt
sed -i 's/${logits_datatype}/TYPE_FP32/g' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt

