#!/bin/bash

cd trl
accelerate launch \
    --config_file examples/accelerate_configs/single_gpu.yaml \
    --num_processes 1 \
    --machine_rank 0 \
    examples/scripts/sft.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
    --dataset_name /root/autodl-tmp/pasa-dataset/sft_crawler/train.jsonl \
    --learning_rate 1.0e-4 \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_steps 2000 \
    --max_seq_length 1024 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --output_dir ../results/sft_crawler \
    --attn_implementation "flash_attention_2" \
    --use_peft \
    --lora_task_type CAUSAL_LM \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj
