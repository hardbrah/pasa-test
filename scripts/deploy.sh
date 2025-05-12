#!/bin/bash
export VLLM_DEVICE=cuda

# export VLLM_LOGGING_LEVEL=DEBUG
vllm serve /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
    --dtype bfloat16 \
    --lora-dtype bfloat16 \
    --max-num-seqs 1 \
    --max-model-len 1024 \
    --device cuda \
    --enable-lora \
    --lora-modules '{"name":"selector", "path": "/root/pasa/results/sft_selector/checkpoint-4957"}'