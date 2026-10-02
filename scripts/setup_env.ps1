# 在项目根目录执行（Anaconda Prompt / PowerShell）

conda create -n segformer-building python=3.10 -y
conda activate segformer-building

# CUDA 12.4 轮子（与驱动 616 兼容；若失败可改 cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
