from modelscope.hub.snapshot_download import snapshot_download
import os

os.makedirs("/root/autodl-tmp/models",exist_ok=True)

model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',           # 模型ID（可以是任意魔搭上的模型，如 qwen/Qwen1.5-7B-Chat）
    cache_dir='/root/autodl-tmp/models' # 你想要保存模型的路径
)
print(f"模型已保存到: {model_dir}")

