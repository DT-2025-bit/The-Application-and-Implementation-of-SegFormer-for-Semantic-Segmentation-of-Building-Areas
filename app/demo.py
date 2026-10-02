"""Streamlit Demo：上传遥感图 → 建筑掩膜 / 轮廓叠加 / 面积统计。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.utils import load_checkpoint, load_config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


def overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = np.zeros_like(image_bgr)
    color[mask > 0] = (0, 255, 0)
    out = image_bgr.copy()
    building = mask > 0
    out[building] = (image_bgr[building] * (1 - alpha) + color[building] * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


@st.cache_resource
def load_model(config_path: str, ckpt_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg["model_name"], cfg["encoder"], cfg.get("num_classes", 2), pretrained=False
    ).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    return model, cfg, device


def predict(model, cfg, device, image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x = preprocess(image_bgr, cfg["img_size"]).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    st.set_page_config(page_title="遥感建筑分割 Demo", layout="wide")
    st.title("遥感建筑区域语义分割 Demo")
    st.caption("SegFormer-B0 + 边缘感知损失 | 本科毕设演示")

    with st.sidebar:
        st.header("模型")
        config_path = st.text_input("配置文件", "configs/segformer_b0_edge.yaml")
        ckpt_path = st.text_input("权重路径", "outputs/segformer_b0_edge/best.pth")
        gsd = st.number_input("地面分辨率 GSD (米/像素)", min_value=0.01, value=1.0, step=0.01)
        load_btn = st.button("加载模型", type="primary")

    if "model_pack" not in st.session_state:
        st.session_state.model_pack = None

    if load_btn:
        if not Path(ckpt_path).exists():
            st.error(f"找不到权重: {ckpt_path}")
        else:
            st.session_state.model_pack = load_model(config_path, ckpt_path)
            st.success("模型已加载")

    uploaded = st.file_uploader("上传遥感影像", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("请先加载模型并上传图像。")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("无法读取图像")
        return

    if st.session_state.model_pack is None:
        st.warning("尚未加载模型，仅显示原图。")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
        return

    model, cfg, device = st.session_state.model_pack
    mask = predict(model, cfg, device, image)
    vis = overlay(image, mask)

    building_px = int((mask > 0).sum())
    total_px = mask.size
    area_m2 = building_px * (gsd ** 2)

    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
    c2.image(mask, caption="建筑掩膜", use_container_width=True)
    c3.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="轮廓叠加", use_container_width=True)

    st.subheader("统计")
    st.write(
        {
            "建筑像素数": building_px,
            "占比": f"{building_px / total_px * 100:.2f}%",
            "估算面积(m²)": round(area_m2, 2),
            "估算面积(km²)": round(area_m2 / 1e6, 6),
        }
    )


if __name__ == "__main__":
    main()
