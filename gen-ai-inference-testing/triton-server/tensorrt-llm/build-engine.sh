#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ ! -d /cache ] && echo "/cache dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

MODEL_PATH=/snapshots/$MODEL_ID
if [ ! -d $MODEL_PATH ]
then
pip3 install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type model \
    --local-dir $MODEL_PATH \
    --token $HF_TOKEN  $MODEL_ID
fi

TP_SIZE=8
PP_SIZE=1

CKPT_PATH=$MODEL_PATH/trtllm_ckpt

if [ ! -d $CKPT_PATH ]
then
echo "Convert HF ckpt to TensorRT-LLM ckpt" 
cd /opt/TensorRT-LLM
python3 \
examples/llama/convert_checkpoint.py \
--model_dir=$MODEL_PATH \
--output_dir=$CKPT_PATH \
--dtype=float16 \
--tp_size=$TP_SIZE 
fi

cd /opt/tensorrtllm_backend
mkdir -p /cache/$MODEL_ID
ENGINE_DIR=/cache/$MODEL_ID/trtllm_engine

if [ ! -d $ENGINE_DIR ]
then
echo "Build TensorRT-LLM engine"

trtllm-build \
--checkpoint_dir ${CKPT_PATH} \
--max_num_tokens 32768 \
--tp_size ${TP_SIZE} \
--pp_size ${PP_SIZE} \
--gpus_per_node 8 \
--remove_input_padding enable \
--gemm_plugin float16 \
--gpt_attention_plugin float16 \
--paged_kv_cache enable \
--context_fmha enable \
--output_dir ${ENGINE_DIR} \
--max_batch_size 8 \
--use_custom_all_reduce disable

mpirun --allow-run-as-root -np $TP_SIZE python3 examples/run.py --tokenizer_dir $MODEL_PATH --engine_dir $ENGINE_DIR --max_output_len 128
fi

MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO

echo "Build Triton TensorRT-LLM model"

TOKENIZER_DIR=$MODEL_PATH
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
MAX_BATCH_SIZE=8
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=100
FILL_TEMPLATE_SCRIPT=tools/fill_template.py

MODEL_NAME=model
cp -r all_models/inflight_batcher_llm/preprocessing $MODEL_REPO/${MODEL_NAME}_preprocessing
cp -r all_models/inflight_batcher_llm/postprocessing $MODEL_REPO/${MODEL_NAME}_postprocessing
cp -r all_models/inflight_batcher_llm/tensorrt_llm_bls $MODEL_REPO/${MODEL_NAME}_tensorrt_llm_bls
cp -r all_models/inflight_batcher_llm/ensemble $MODEL_REPO/${MODEL_NAME}
cp -r all_models/inflight_batcher_llm/tensorrt_llm $MODEL_REPO/${MODEL_NAME}_tensorrt_llm
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT} 
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},tensorrt_llm_model_name:${MODEL_NAME}_tensorrt_llm
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching
sed -i 's/name: "preprocessing"/name: "model_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt
sed -i 's/name: "postprocessing"/name: "model_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt
sed -i 's/name: "tensorrt_llm_bls"/name: "model_tensorrt_llm_bls"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt
sed -i 's/name: "ensemble"/name: "model"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/name: "tensorrt_llm"/name: "model_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt
sed -i 's/model_name: "preprocessing"/model_name: "model_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/model_name: "postprocessing"/model_name: "model_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt
sed -i 's/model_name: "tensorrt_llm"/model_name: "model_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}/config.pbtxt

