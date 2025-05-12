#!/bin/bash

cd trl
accelerate launch \
    --config_file examples/accelerate_configs/single_gpu.yaml \
    --main_process_port 2501 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name /root/autodl-tmp/pasa-dataset/AutoScholarQuery/train.jsonl \
    --dataset_test_split validation \
    --output_dir ../results/ppo_crawler \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --total_episodes 160 \
    --paper_db /root/autodl-tmp/pasa-dataset/paper_database/cs_paper_2nd.zip \
    --paper_id /root/autodl-tmp/pasa-dataset/paper_database/id2paper.json \
    --model_name_or_path /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
    --sft_model_path /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
    --reward_model_path /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
    --local_rollout_forward_batch_size 4 \
    --num_sample_generations 0 \
    --attn_implementation "flash_attention_2" \
    --response_length 512 \
    --stop_token eos \
    --gamma1 0.1 \
    --save_steps 10 \
    --rounds 3 \
    --use_vm True \
    --use_selector True \
    --vf_coef 10.0 \
    --expand_select_score 1.5 \
    --expand_cost 0.1 \
    --search_select_score 1.5 \
    --search_cost 0.1 \
    --num_ppo_epochs 2 \
    --kl_coef 0.1 \
    --lora_path /root/pasa/results/sft_crawler/checkpoint-3248