# Massachusetts Buildings 数据集准备说明（新手版）

## 1. 去哪里下载

任选其一：

1. 官方页面：https://www.cs.toronto.edu/~vmnih/data/
2. Kaggle 搜索：`Massachusetts Buildings Dataset`  
   常见镜像包：`balraj98/massachusetts-buildings-dataset`

下载后解压到任意临时目录，再整理进本项目。

## 2. 必须整理成的目录结构

请放到**项目根目录**下（与 `README.md` 同级）：

```text
data/raw/massachusetts/
  ├── train/
  │   ├── images/    # 原图 *.tiff / *.png / *.jpg，约 1500x1500
  │   └── masks/     # 与 images 同名的掩膜
  ├── val/
  │   ├── images/
  │   └── masks/
  └── test/
      ├── images/
      └── masks/
```

### 命名规则

- `images` 与 `masks` 的**文件名主体必须一致**（扩展名可以不同）
- 例如：`xxx.tiff` ↔ `xxx.png` 可以；`xxx.tiff` ↔ `yyy.png` 不行
- 掩膜：建筑区域为非零/白色，背景为 0/黑色

### 如果下载包不是 train/val/test 结构

有的压缩包只有 `images` + `labels`（或 `masks`）两个大文件夹。请：

1. 按官方划分（或 Kaggle 说明）拆成 train / val / test
2. 再分别放入上面的 `images/`、`masks/` 子目录

## 3. 整理完成后做什么

在项目根目录、已激活 `segformer-building` 环境时执行：

```bash
python scripts/slice_dataset.py
python scripts/split_check.py
```

切片结果会写到：

```text
data/processed/massachusetts/
```

**注意：训练脚本读取的是 `data/processed/...`，不是 `data/raw/...`。**  
不切片直接训练，通常会报找不到数据或路径错误。

## 4. 可选参数

```bash
# 不过滤低建筑占比切片
python scripts/slice_dataset.py --min-building-ratio 0

# 自定义路径
python scripts/slice_dataset.py --raw-root data/raw/massachusetts --out-root data/processed/massachusetts
```
