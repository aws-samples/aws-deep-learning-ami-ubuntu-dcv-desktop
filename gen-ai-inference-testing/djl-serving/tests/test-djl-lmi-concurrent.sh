#!/bin/bash

mkdir -p output
STARTTIME=$(date +%s)
for i in {1..8};
do
curl -X POST http://localhost:8080/invocations \
  --connect-timeout 120 \
  -H "Content-type: application/json" \
  -d '{"inputs":" <|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI applications in Mobile apps. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>","parameters":{ "do_sample": true, "max_new_tokens": 2048, "top_k": 50 }}' \
1>output/test-djl-concurrent.0-$i.out &
curl -X POST http://localhost:8080/invocations \
  --connect-timeout 120 \
  -H "Content-type: application/json" \
  -d '{"inputs":" <|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI in computer visions. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>","parameters":{ "do_sample": true, "max_new_tokens": 2048, "top_k": 50 }}' \
1>output/test-djl-concurrent.1-$i.out &
curl -X POST http://localhost:8080/invocations \
  --connect-timeout 120 \
  -H "Content-type: application/json" \
  -d '{"inputs": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on the ethics of Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>","parameters":{ "do_sample": true, "max_new_tokens": 2048, "top_k": 50 }}' \
1>output/test-djl-concurrent.2-$i.out &
curl -X POST http://localhost:8080/invocations \
  --connect-timeout 120 \
  -H "Content-type: application/json" \
  -d '{"inputs": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on the dangers of Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>","parameters":{ "do_sample": true, "max_new_tokens": 2048, "top_k": 50 }}' \
1>output/test-djl-concurrent.3-$i.out &
done

wait
ENDTIME=$(date +%s)
echo "Elapsed time: $((ENDTIME - STARTTIME)) seconds for 32 requests"