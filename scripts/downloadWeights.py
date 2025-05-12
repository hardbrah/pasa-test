from huggingface_hub import snapshot_download

# 指定模型 ID，下载 Hugging Face 仓库中的模型
snapshot_download(repo_id="bytedance-research/pasa-7b-crawler", local_dir="/root/autodl-tmp/models/pasa-7b-crawler")
snapshot_download(repo_id="bytedance-research/pasa-7b-selector", local_dir="/root/autodl-tmp/models/pasa-7b-selector")