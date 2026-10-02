# SegFormer 遥感建筑区域语义分割

本科毕设 / 可复用模板项目：基于 **SegFormer-B0** 对遥感航拍图做**建筑像素级分割**，并加入**边缘感知损失**，改善建筑轮廓与边界粘连问题。

> **给新手**：下面「从零跑通」按顺序做即可。所有命令默认在**项目根目录**执行（就是本 README 所在文件夹）。

---

## 你需要准备什么

| 项目 | 建议 | 说明 |
|------|------|------|
| 系统 | Windows 10/11 | 本仓库脚本按 Windows 写的，Linux/macOS 命令类似 |
| GPU | NVIDIA，显存 ≥ 6GB | 无 GPU 也能跑，但训练会非常慢 |
| 软件 | [Anaconda / Miniconda](https://www.anaconda.com/download) | 用来创建 Python 3.10 环境 |
| 驱动 | 较新的 NVIDIA 驱动 | 在命令行能识别到 GPU |
| 磁盘 | 建议预留 10GB+ | 数据 + 权重 + 依赖 |

---

## 从零跑通（推荐按 1→7）

### 第 1 步：拿到代码

任选一种方式：

```bash
# 方式 A：从 GitHub 克隆（把地址换成你的仓库）
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>

# 方式 B：直接下载 ZIP 解压，用资源管理器进入解压后的根目录
```

确认根目录里能看到这些文件/文件夹：

```text
README.md
requirements.txt
configs/
scripts/
src/
tools/
app/
data/
outputs/
```

---

### 第 2 步：创建并激活 Python 环境

1. 打开 **Anaconda Prompt**（或已装好 conda 的 PowerShell）。
2. 进入项目根目录，例如：

```powershell
cd /d E:\path\to\本项目
```

3. 创建环境并安装依赖（**只需做一次**）：

```bash
conda create -n segformer-building python=3.10 -y
conda activate segformer-building

# 先装 PyTorch（CUDA 12.4 示例；若失败可把 cu124 改成 cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 再装项目依赖
pip install -r requirements.txt
```

也可以直接运行仓库自带脚本（效果等价）：

```powershell
.\scripts\setup_env.ps1
```

> **注意**：以后每次重新开终端，都要先执行  
> `conda activate segformer-building`  
> 再跑后面的 `python` / `streamlit` 命令。

#### 检查环境是否成功

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

期望看到类似：

```text
torch 2.x.x
cuda True
NVIDIA GeForce RTX ....
```

若 `cuda False`：先更新显卡驱动；或确认安装的是带 CUDA 的 PyTorch，而不是 CPU 版。

---

### 第 3 步：准备数据集（最容易卡住的一步）

本项目默认使用 **Massachusetts Buildings Dataset**（航拍建筑标注）。

#### 3.1 下载

任选其一：

1. 官方：<https://www.cs.toronto.edu/~vmnih/data/>
2. Kaggle 搜索：`Massachusetts Buildings Dataset`（常见镜像包名如 `balraj98/massachusetts-buildings-dataset`）

更细说明见：`scripts/download_notes.md`

#### 3.2 按下面结构放好（名字必须对上）

在项目根目录创建/整理为：

```text
data/raw/massachusetts/
  ├── train/
  │   ├── images/     # 原图：png/jpg/tif 均可，约 1500×1500
  │   └── masks/      # 掩膜：与 images 同名（扩展名可不同）
  ├── val/
  │   ├── images/
  │   └── masks/
  └── test/
      ├── images/
      └── masks/
```

要点：

- `images` 与 `masks` 文件名主体（不含扩展名）要一致，例如 `22828930_15.tiff` 对应 `22828930_15.tif`
- 掩膜里建筑一般为白色/非零，背景为黑/零
- 如果下载包只有 `images` + `labels` 两大文件夹，请先按官方划分手动整理进 `train/val/test`

#### 3.3 快速自检

```bash
# PowerShell：看 train 图像数量是否 > 0
dir data\raw\massachusetts\train\images
dir data\raw\massachusetts\train\masks
```

---

### 第 4 步：切片（必须做，训练读的是切片后的数据）

原图约 1500×1500，训练默认读 **512×512** 切片。在项目根目录执行：

```bash
python scripts/slice_dataset.py
python scripts/split_check.py
```

成功后会生成：

```text
data/processed/massachusetts/
  ├── train/images  train/masks
  ├── val/images    val/masks
  ├── test/images   test/masks
  └── slice_summary.json
```

默认参数：`tile=512`，`overlap=64`，丢弃建筑像素占比 < 1% 的切片。  
若想全部保留切片：

```bash
python scripts/slice_dataset.py --min-building-ratio 0
```

---

### 第 5 步：开始训练（跑通一个模型）

第一次建议直接训「边缘损失改进版」：

```bash
python tools/train.py --config configs/segformer_b0_edge.yaml
```

训练时你会看到：

- `Device: cuda` / GPU 名称
- 每个 epoch 的 loss、mIoU 等
- 权重保存到：`outputs/segformer_b0_edge/`
  - `best.pth`：验证集最好权重（后面评估/Demo 用这个）
  - `last.pth`：最近一轮

参考耗时（RTX 4050 6GB，早停）：大约数小时量级，具体取决于数据量与 patience。

#### 显存不够怎么办？

打开对应 yaml，例如 `configs/segformer_b0_edge.yaml`，把：

```yaml
batch_size: 4
```

改成 `2` 或 `1`。DeepLabV3+ 默认已是 `batch_size: 2`。

---

### 第 6 步：评估与单图推理

```bash
# 验证集评估
python tools/eval.py --config configs/segformer_b0_edge.yaml --checkpoint outputs/segformer_b0_edge/best.pth

# 测试集评估
python tools/eval.py --config configs/segformer_b0_edge.yaml --checkpoint outputs/segformer_b0_edge/best.pth --split test

# 单张图推理（把路径换成你的图片）
python tools/infer.py --config configs/segformer_b0_edge.yaml --checkpoint outputs/segformer_b0_edge/best.pth --image data/processed/massachusetts/val/images/某张图.png
```

推理结果默认写到 `outputs/` 相关可视化目录（以命令输出为准）。

---

### 第 7 步：打开可视化 Demo

先确保已经训出权重 `outputs/segformer_b0_edge/best.pth`，然后：

```bash
streamlit run app/demo.py
```

浏览器会自动打开本地页面。在左侧：

1. 确认配置：`configs/segformer_b0_edge.yaml`
2. 确认权重：`outputs/segformer_b0_edge/best.pth`
3. 点 **加载模型**
4. 上传一张遥感图，查看掩膜 / 轮廓叠加 / 面积统计

---

## 一页纸命令清单（复制用）

```bash
conda activate segformer-building

# 数据切片
python scripts/slice_dataset.py
python scripts/split_check.py

# 训练 → 评估 → Demo
python tools/train.py --config configs/segformer_b0_edge.yaml
python tools/eval.py --config configs/segformer_b0_edge.yaml --checkpoint outputs/segformer_b0_edge/best.pth
streamlit run app/demo.py
```

---

## 可选：换模型 / 做对比实验

| 配置文件 | 做什么 |
|----------|--------|
| `configs/segformer_b0_edge.yaml` | 主实验：SegFormer + 边缘损失（λ=0.5） |
| `configs/segformer_b0_baseline.yaml` | SegFormer 无边缘损失（对照） |
| `configs/segformer_b0_edge_l01.yaml` | λ=0.1 消融 |
| `configs/segformer_b0_edge_l10.yaml` | λ=1.0 消融 |
| `configs/unet_resnet34.yaml` | U-Net 对比 |
| `configs/deeplabv3plus_resnet50.yaml` | DeepLabV3+ 对比 |

把训练命令里的 `--config` 换成上表路径即可，例如：

```bash
python tools/train.py --config configs/unet_resnet34.yaml
```

---

## 常见报错与处理

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `No module named 'torch'` | 没激活环境 / 没装依赖 | `conda activate segformer-building` 后重装 `requirements.txt` |
| `cuda False` | 装成 CPU 版 PyTorch / 驱动问题 | 按第 2 步重装 cu124/cu121 轮子，更新驱动 |
| `FileNotFoundError` 找不到 images/masks | 数据目录结构不对 | 对照第 3 步检查 `data/raw/massachusetts/...` |
| 训练立刻报错缺数据 | 没切片 | 先跑 `scripts/slice_dataset.py` |
| `CUDA out of memory` | batch 太大 | yaml 里把 `batch_size` 改小 |
| Demo 提示找不到权重 | 还没训练完 | 先确认存在 `outputs/.../best.pth` |
| HuggingFace 下载很慢/失败 | 首次会下 `mit-b0` 预训练 | 开代理，或配置镜像后再训 |
| `num_workers` 相关报错 | Windows 多进程问题 | yaml 里把 `num_workers` 改成 `0` |

---

## 项目简介（背景）

| 项目 | 说明 |
|------|------|
| 主模型 | SegFormer-B0（HuggingFace `nvidia/mit-b0`） |
| 损失 | CE + Dice + 边缘感知损失 |
| 数据 | Massachusetts Buildings → 512×512，overlap=64 |
| Demo | Streamlit 上传 / 掩膜 / 轮廓 / 面积 |
| 硬件参考 | RTX 4050 Laptop 6GB + AMP |

### 参考实验结果（验证集，本仓库作者环境）

| 指标 | 数值 |
|------|------|
| mIoU | **0.790** |
| IoU_building | 0.630 |
| Dice | 0.873 |
| Pixel Accuracy | 0.954 |
| Precision / Recall | 0.773 / 0.773 |
| Boundary IoU | 0.529 |

> 配置：`configs/segformer_b0_edge.yaml`（λ=0.5）。你本机结果可能因驱动、随机种子、数据切片过滤略有差异。

---

## 目录结构

```text
├── app/demo.py           # Streamlit 演示
├── configs/              # 各模型训练配置（改超参主要看这里）
├── data/
│   ├── raw/              # 你下载的原图（不入库）
│   └── processed/        # 切片结果（不入库）
├── scripts/
│   ├── setup_env.ps1     # 一键装环境
│   ├── slice_dataset.py  # 切片
│   ├── split_check.py    # 检查划分是否泄漏
│   └── download_notes.md # 数据下载说明
├── src/                  # 数据 / 模型 / 损失 / 指标
├── tools/
│   ├── train.py          # 训练
│   ├── eval.py           # 评估
│   └── infer.py          # 推理
├── outputs/              # 权重与日志（权重不入库）
└── requirements.txt
```

---

## 把本仓库当「模板」怎么用

1. Fork / 复制本仓库，改 README 里的项目名与介绍。
2. 换数据集时：保持 `images/` + `masks/` 成对结构，或改 `configs/*.yaml` 里的路径。
3. 换模型时：在 `src/models/builder.py` 增加构建逻辑，并新增一份 `configs/xxx.yaml`。
4. 调损失权重：改 yaml 中的 `ce_weight` / `dice_weight` / `edge_weight`。
5. 不要把 `data/raw`、`data/processed`、大权重提交到 Git（已在 `.gitignore` 忽略）。

---

## 上传到 GitHub（可选）

1. 运行 `init_git_repo.bat` 初始化仓库（若尚未 init）。
2. 在 GitHub 新建空仓库，复制地址。
3. 运行 `upload_github.bat`，按提示填写地址与 commit message。

> 默认代理端口 `65532`。若你本机 VPN/代理端口不同，请编辑 `upload_github.bat` 顶部的 `PROXY_PORT`。

---

## 致谢

- SegFormer：Xie et al., NeurIPS 2021  
- Massachusetts Buildings：Mnih, University of Toronto, 2013  
- 依赖：PyTorch、HuggingFace Transformers、segmentation_models_pytorch、Streamlit 等

代码主要用于学习与毕设复现；使用模型/数据时请遵守原作者许可。
