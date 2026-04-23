from huggingface_hub import snapshot_download
import os

# 【关键：开启国内镜像源】
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 自动下载模型到 ~/models/bge-base-zh-v1.5
model_path = os.path.expanduser("~/models/bge-base-zh-v1.5")

snapshot_download(
    repo_id="BAAI/bge-base-zh-v1.5",  # 模型官方名称
    local_dir=model_path,             # 保存到你代码里的路径
    local_dir_use_symlinks=False      # 直接保存，不搞快捷方式
)

print("模型下载完成！路径：", model_path)
