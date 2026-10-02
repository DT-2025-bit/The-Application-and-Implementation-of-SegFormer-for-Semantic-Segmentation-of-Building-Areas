"""单图/目录推理与可视化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask > 0] = (0, 255, 0)
    return color


def overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask)
    out = image_bgr.copy()
    building = mask > 0
    out[building] = (image_bgr[building] * (1 - alpha) + color[building] * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


@torch.no_grad()
def predict_image(model, image_bgr, img_size, device):
    h, w = image_bgr.shape[:2]
    x = preprocess(image_bgr, img_size).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="图像文件或目录")
    parser.add_argument("--output", type=str, default="outputs/infer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg["model_name"], cfg["encoder"], cfg.get("num_classes", 2), pretrained=False
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [in_path] if in_path.is_file() else sorted(
        list(in_path.glob("*.png"))
        + list(in_path.glob("*.jpg"))
        + list(in_path.glob("*.tif"))
        + list(in_path.glob("*.tiff"))
    )

    for fp in tqdm(files):
        image = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask = predict_image(model, image, cfg["img_size"], device)
        vis = overlay(image, mask)
        stem = fp.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), vis)

    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
