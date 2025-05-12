#!/bin/bash

MODEL_DIR=/root/autodl-tmp/data
DATA_DIR=/root/autodl-tmp/pasa-dataset
python run_paper_agent.py \
    --input_file    ${DATA_DIR}/RealScholarQuery/test.jsonl \
    --crawler_path   ${MODEL_DIR}/pasa-7b-crawler \
    --selector_path  ${MODEL_DIR}/pasa-7b-selector \
    --threads_num 2    \
    # --search_papers 4 \
    # --expand_papers  4
