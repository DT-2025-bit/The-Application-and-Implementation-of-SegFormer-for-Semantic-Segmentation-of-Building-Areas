"""评估：python tools/eval.py --config configs/segformer_b0_edge.yaml --checkpoint outputs/.../best.pth"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BuildingSegDataset, collate_fn
from src.metrics import SegmentationMeter
from src.models import build_model
from src.utils import load_checkpoint, load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--boundary-bandwidth", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.split == "val":
        image_dir, mask_dir = cfg["val_image_dir"], cfg["val_mask_dir"]
    else:
        image_dir = cfg.get("test_image_dir", cfg["val_image_dir"].replace("val", "test"))
        mask_dir = cfg.get("test_mask_dir", cfg["val_mask_dir"].replace("val", "test"))

    ds = BuildingSegDataset(image_dir, mask_dir, cfg["img_size"], train=False)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    model = build_model(
        cfg["model_name"], cfg["encoder"], cfg.get("num_classes", 2), pretrained=False
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    meter = SegmentationMeter(
        num_classes=cfg.get("num_classes", 2),
        boundary_bandwidth=args.boundary_bandwidth,
    )
    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        meter.update(logits, masks)

    metrics = meter.compute()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    out = Path(args.checkpoint).parent / f"metrics_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
